"""CPU attribution from captured A3 QK, not an A3 numerical oracle."""
import json
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
torch.set_num_threads(4)
results = []
for batch, head in ((16, 51), (40, 34)):
    data = torch.load(root / f"b{batch}/sparse_boundary.pt", weights_only=True)
    scores = data["scores"] * (512 ** -0.5)
    outputs, ties = {}, []
    for mode in ("rint", "round"):
        maximum = data["sink"][:, None].clone()
        denominator = torch.ones_like(maximum)
        numerator = torch.zeros((64, 512))
        for first, last in ((0, 128), (128, 640)):
            new_maximum = torch.maximum(maximum, scores[:, first:last].max(-1, keepdim=True).values)
            probability = (scores[:, first:last] - new_maximum).exp()
            alpha = (maximum - new_maximum).exp()
            if mode == "round":
                # Probabilities are positive: add half a BF16 ULP, then truncate.
                bits = probability.view(torch.int32)
                rounded = ((bits + 32768) & -65536).view(torch.float32)
                for i, j in torch.nonzero((bits & 65535) == 32768).tolist():
                    ties.append(dict(head=i, column=first+j, probability=float(probability[i, j]),
                                     rint=float(probability[i, j].bfloat16()), round=float(rounded[i, j])))
            else:
                rounded = probability.bfloat16().float()
            numerator = numerator * alpha + (rounded.double() @ data["selected_kv"][first:last].double()).float()
            denominator = denominator * alpha + probability.sum(-1, keepdim=True)
            maximum = new_maximum
        output = (numerator / denominator).bfloat16()
        outputs[mode] = dict(head_nope_vs_native=int((output[head, :448] != data["native_attention_output"][head, :448]).sum()),
                             head_nope_vs_pto=int((output[head, :448] != data["projection_input"][head, :448]).sum()))
    results.append(dict(batch=batch, head=head, ties=ties, modes=outputs))
(root / "rounding_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
