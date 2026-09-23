"""CPU-only reconstruction of the eight differing SWA rows in B40/step46.

Run from the repository after sourcing ../env.sh, with
TORCH_DEVICE_BACKEND_AUTOLOAD=0. No checkpoint state or device work is needed.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

root = Path(__file__).resolve().parent
old = root.parent / "continuous_output_late_v1/b40"
checkpoint = Path("/data/model/dsv4-flash-0731-dspark-w8a8")
torch.set_num_threads(4)
data = torch.load(old / "output_boundary.pt", weights_only=True)
failure = torch.load(old / "failure.pt", weights_only=True)
query, step_limit, batch, seed = 185, 46, 40, 1024
request = query // 6
position = int(failure["positions"][query])
pto = data["pto.selected_kv"][query, :128]
native = data["native.selected_kv"][query, :128]
changed = torch.nonzero((pto != native).any(1)).flatten()
writes, history = {}, 131071
for step in range(step_limit + 1):
    for token in range(6):
        writes[history + token] = step, request * 6 + token
    history += (1, 6, 2, 5, 5)[(step + request) % 5]
inputs, origins = [], []
for row in changed.tolist():
    absolute = position - 127 + row
    step, input_row = writes[absolute]
    hidden = torch.randn((batch * 6, 4096), generator=torch.Generator().manual_seed(seed + step),
                         dtype=torch.bfloat16)
    inputs.append(hidden[input_row])
    origins.append(dict(selected_row=row, position=absolute, step=step, input_row=input_row))
weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
weights = {}
for name in ("layers.2.attn.wkv.weight", "layers.2.attn.kv_norm.weight"):
    with safe_open(checkpoint / weight_map[name], framework="pt", device="cpu") as reader:
        weights[name] = reader.get_tensor(name)
x = torch.stack(inputs)
weight, gamma = (weights[name] for name in ("layers.2.attn.wkv.weight", "layers.2.attn.kv_norm.weight"))
projection = (x.double() @ weight.double().T).float().bfloat16()
values = projection.float()
norm = (values * (values.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * gamma.float()).bfloat16()
for i, origin in enumerate(origins):
    row = origin["selected_row"]
    origin.update(cpu_norm_nope_vs_native=int((norm[i, :448] != native[row, :448]).sum()),
                  cpu_norm_nope_vs_pto=int((norm[i, :448] != pto[row, :448]).sum()),
                  original_nope_diffs=int((pto[row, :448] != native[row, :448]).sum()))
torch.save(dict(x=x, weight=weight, gamma=gamma, projection_cpu=projection, norm_cpu=norm,
                native_selected=native[changed], pto_selected=pto[changed], origins=origins),
           root / "swa_inputs.pt")
(root / "cpu_attribution.json").write_text(json.dumps(origins, indent=2) + "\n")
