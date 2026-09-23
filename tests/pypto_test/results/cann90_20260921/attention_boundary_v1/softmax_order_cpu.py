"""CPU attribution only: compare block probability rounding with captured NPU data."""
import json
from pathlib import Path
import torch

root = Path(__file__).parent
torch.set_num_threads(8)
results = []

def metrics(actual, expected):
    diff = actual.float() - expected.float()
    return {'different': int((actual != expected).sum()), 'elements': actual.numel(),
            'max_abs': float(diff.abs().max()), 'rmse': float(diff.square().mean().sqrt())}

def attention(q, kv, valid, sink, mode):
    # FP64 dot product is only an attribution aid; it does not model Cube's FP32 reduction.
    scores = (q.double() @ kv.double().T).float() * (512 ** -0.5)
    scores[:, ~valid] = -2e38
    if mode == 'global_float_probability':
        m = torch.maximum(scores.max(-1, keepdim=True).values, sink[:, None])
        p = (scores - m).exp()
        return (p @ kv.float() / (p.sum(-1, keepdim=True) + (sink[:, None]-m).exp())).bfloat16()
    blocks = [(0,128),(128,640)] if mode == 'running512' else [(i,i+128) for i in range(0,640,128)]
    m = sink[:, None].clone()
    l = torch.zeros_like(m) if mode == 'local128' else torch.ones_like(m)
    o = torch.zeros_like(q, dtype=torch.float32)
    for first,last in blocks:
        s = scores[:, first:last]
        block_m = s.max(-1, keepdim=True).values
        new_m = torch.maximum(m, block_m)
        p = (s - (block_m if mode == 'local128' else new_m)).exp()
        local_l = p.sum(-1, keepdim=True)
        pv = (p.bfloat16().double() @ kv[first:last].double()).float()
        alpha = (m-new_m).exp()
        beta = (block_m-new_m).exp() if mode == 'local128' else torch.ones_like(m)
        o = o * alpha + pv * beta
        l = l * alpha + local_l * beta
        m = new_m
    if mode == 'local128':
        l = l + (sink[:,None]-m).exp()
    return (o/l).bfloat16()

for batch,row in ((4,18),(32,191)):
    data = torch.load(root/f'b{batch}/output_boundary.pt', map_location='cpu', weights_only=True)
    expected = data['native.attention_output'][row]
    pto = data['pto.native_query_native_cache.projection_input'][row]
    item={'batch':batch,'row':row,'scope':'NoPE columns0:448; same captured Native query and selected cache',
          'native_vs_pto':metrics(pto[:,:448],expected[:,:448]),'modes':{}}
    for mode in ('local128','running128','running512','global_float_probability'):
        output=attention(data['native.query'][row],data['native.selected_kv'][row],
                         data['selected_valid'][row],data['sink'],mode)
        item['modes'][mode]={'vs_native':metrics(output[:,:448],expected[:,:448]),
                             'vs_pto':metrics(output[:,:448],pto[:,:448])}
    results.append(item)
    print(json.dumps(item),flush=True)
(root/'softmax_order_cpu.json').write_text(json.dumps(results,indent=2)+'\n')
