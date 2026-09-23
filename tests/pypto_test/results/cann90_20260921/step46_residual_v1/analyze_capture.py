"""CPU-only attribution of saved production boundaries, with exact coordinates."""
import argparse
import json
from pathlib import Path
import torch

torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument("capture", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
d = torch.load(args.capture, map_location="cpu", weights_only=False)

def compare(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {"max_abs": delta.max().item(), "rmse": delta.square().mean().sqrt().item(),
            "different": (actual != expected).sum().item(),
            "over_tolerance": (delta > 0.01 + 0.01 * expected.float().abs()).sum().item()}

native = d["native.output"]
pto = d["pto.full_output"]
diff = (pto.float() - native.float()).abs()
coordinates = torch.nonzero(diff == diff.max()).tolist()
report = {"capture": str(args.capture), "full_output": compare(pto, native),
          "max_coordinates": coordinates, "boundaries": {}, "rows": {}}
for key in ("pto.given_native_projection_input", "pto.pto_query_pto_cache.output",
            "pto.native_query_pto_cache.output", "pto.native_query_native_cache.output"):
    report["boundaries"][key] = compare(d[key], native)
for row in sorted({r for r, _ in coordinates}):
    record = {"max_columns": [c for r, c in coordinates if r == row], "values": {}, "stages": {}}
    for col in record["max_columns"]:
        record["values"][col] = {key: float(d[key][row, col]) for key in
            ("native.output", "pto.full_output", "pto.given_native_projection_input",
             "pto.native_query_pto_cache.output", "pto.native_query_native_cache.output")}
    for name, actual, expected in (
        ("query", d["pto.query"][row], d["native.query"][row]),
        ("swa", d["pto.selected_kv"][row, :128], d["native.selected_kv"][row, :128]),
        ("compressed", d["pto.selected_kv"][row, 128:], d["native.selected_kv"][row, 128:]),
        ("projection_input", d["pto.pto_query_pto_cache.projection_input"][row], d["native.projection_input"][row]),
        ("projection_input_native_query_cache", d["pto.native_query_native_cache.projection_input"][row], d["native.projection_input"][row]),
    ):
        record["stages"][name] = compare(actual, expected)
    report["rows"][row] = record
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
