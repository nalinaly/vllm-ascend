"""CPU-only accounting of remaining differences in the unchanged step46 capture."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
d = torch.load(args.capture, map_location="cpu", weights_only=False)


def metrics(actual, expected, atol=0.0, rtol=0.0):
    expected = expected.reshape(actual.shape)
    delta = (actual.float() - expected.float()).abs()
    mask = actual != expected
    return {"elements": actual.numel(), "different": int(mask.sum()), "atol": atol, "rtol": rtol,
            "max_abs": float(delta.max()), "rmse": float(delta.square().mean().sqrt()),
            "over_tolerance": int((delta > atol + rtol * expected.float().abs()).sum()),
            "rows": mask.reshape(actual.shape[0], -1).any(-1).nonzero().flatten().tolist()}


report = {"capture": str(args.capture), "stages": {}, "outputs": {}, "query_coordinates": [],
          "unique_swa_errors": [], "attention_coordinates_given_native_inputs": [],
          "scope": "one saved B40 step46; diagnostic substitutions are not additive error counts"}
for stage, actual, expected in (
    ("qr", "pto.qr", "native.qr"), ("qr_scale", "pto.qr_scale", "native.qr_scale"),
    ("query", "pto.query", "native.query"),
    ("attention_and_inverse_rope", "pto.pto_query_pto_cache.projection_input", "native.projection_input"),
    ("attention_given_native_query", "pto.native_query_pto_cache.projection_input", "native.projection_input"),
    ("attention_given_native_query_cache", "pto.native_query_native_cache.projection_input", "native.projection_input"),
):
    report["stages"][stage] = metrics(d[actual], d[expected])
for label in ("pto.full_output", "pto.native_query_pto_cache.output", "pto.native_query_native_cache.output",
              "pto.given_native_projection_input"):
    report["outputs"][label] = metrics(d[label], d["native.output"], .01, .01)
for idx in (d["pto.query"] != d["native.query"]).nonzero().tolist():
    report["query_coordinates"].append({"coordinate": idx, "native": float(d["native.query"][tuple(idx)]),
                                         "pto": float(d["pto.query"][tuple(idx)])})
for kind, selection in (("swa", slice(0, 128)), ("compressed", slice(128, None))):
    actual = d["pto.selected_kv"][:, selection].clone()
    expected = d["native.selected_kv"][:, selection].clone()
    valid = d["selected_valid"][:, selection]
    actual[~valid] = 0
    expected[~valid] = 0
    report["stages"][kind] = metrics(actual, expected)

unique = {}
mask = (d["pto.selected_kv"][:, :128] != d["native.selected_kv"][:, :128]) & d["selected_valid"][:, :128, None]
for q, k, c in mask.nonzero().tolist():
    request = q // 6
    position = max(0, int(d["positions"][q, 0]) - 127) + k
    key = (request, position, c)
    record = unique.setdefault(key, {"request": request, "position": position, "column": c,
                                    "native": float(d["native.selected_kv"][q, k, c]),
                                    "pto": float(d["pto.selected_kv"][q, k, c]), "query_rows": []})
    record["query_rows"].append(q)
for record in unique.values():
    start = 131071
    for step in range(47):
        if start <= record["position"] < start + 6:
            record["last_write"] = {"step": step, "input_row": record["request"] * 6 + record["position"] - start,
                                    "request_start": start}
        start += (1, 6, 2, 5, 5)[(step + record["request"]) % 5]
    report["unique_swa_errors"].append(record)

actual = d["pto.native_query_native_cache.projection_input"]
expected = d["native.projection_input"]
for idx in (actual != expected).nonzero().tolist():
    report["attention_coordinates_given_native_inputs"].append({"coordinate": idx,
        "native": float(expected[tuple(idx)]), "pto": float(actual[tuple(idx)])})
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"stages": {k: {a: b for a,b in v.items() if a != "rows"} for k,v in report["stages"].items()},
                  "unique_swa_errors": report["unique_swa_errors"]}, indent=2))
