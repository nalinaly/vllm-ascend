"""Observe saved failing candidates using exact INT32 QK and Native coefficients."""

import json
from pathlib import Path

import torch

torch.set_num_threads(8)
root = Path(__file__).resolve().parent
saved = torch.load(root / "query_boundary.pt", map_location="cpu", weights_only=True)
native_indices = saved["native_topk"].reshape_as(saved["pto_topk"])
changed = (native_indices != saved["pto_topk"]).nonzero()
report = {"scope": "CPU observation; FP64 head reduction is not the Native Cube accumulation oracle", "queries": {}}
for query in changed[:, 0].unique().tolist():
    candidates = saved["pto_topk"][query].long()
    pages = saved["block_table"][query // 6, candidates // 32].long()
    keys = saved["key_cache"][pages, candidates % 32, 0]
    scales = saved["scale_cache"][pages, candidates % 32, 0, 0].float()
    q = saved["native.query"][query].int()
    qk = (q @ keys.int().T).clamp_min(0)
    qk_half = (qk.float() / 1024).half()
    coefficients = (
        saved["native.query_scale"][query].float() * saved["native.weights"][query].half().float()
    ).half()
    weighted = (coefficients.double() @ qk_half.double()).float()
    scores = weighted * scales
    pto_scores = saved["pto_scores"][query]
    order = scores.argsort(descending=True, stable=True)
    wanted = native_indices[query].long()
    pairs = []
    for slot in changed[changed[:, 0] == query, 1].tolist():
        native_candidate = int(wanted[slot])
        native_in_pto = (candidates == native_candidate).nonzero().flatten().tolist()
        pairs.append({
            "slot": slot, "native_candidate": native_candidate, "pto_candidate": int(candidates[slot]),
            "pto_score": float(pto_scores[slot]), "cpu_score_for_pto_candidate": float(scores[slot]),
            "cpu_score_for_native_candidate": [float(scores[i]) for i in native_in_pto],
        })
    report["queries"][query] = {
        "cpu_vs_native_topk_positions": int((candidates[order] != wanted).sum()),
        "cpu_vs_pto_scores": {
            "different": int((scores != pto_scores).sum()),
            "max_abs": float((scores - pto_scores).abs().max()),
            "max_ulp": int((scores.view(torch.int32).long() - pto_scores.view(torch.int32).long()).abs().max()),
        },
        "coefficients": coefficients.float().tolist(), "pairs": pairs,
    }
    torch.save({"candidates": candidates, "qk_half": qk_half, "coefficients": coefficients,
                "key_scale": scales, "cpu_scores": scores, "pto_scores": pto_scores},
               root / f"score_cpu_query{query}.pt")
(root / "score_cpu.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
