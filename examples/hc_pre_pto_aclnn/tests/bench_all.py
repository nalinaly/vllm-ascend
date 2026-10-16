# SPDX-License-Identifier: Apache-2.0
"""Compare all 14 cases from the 2026-10-12 baseline through Torch/ACLNN, with random weights by default."""
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu

from check_outputs import reference
from test_torch import compare


def profile(graph, directory, kernel):
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory), analyse_flag=False),
    ) as prof:
        for _ in range(3):
            graph.replay()
            torch.npu.synchronize()
            prof.step()
    raw=list(directory.glob('*/PROF_*'))
    if len(raw)!=1:raise RuntimeError(f'Expected one raw capture: {raw}')
    return {'kernel':kernel, 'raw':str(raw[0].relative_to(directory.parent.parent)),
            'status':'captured; awaiting offline export'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,help='Optional model directory; omitted for seeded random weights')
    p.add_argument('--weight-seed',type=int,default=20261009)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,nargs='+',default=[2,4,8,16,20,24,28])
    p.add_argument('--seqlen',type=int,nargs='+',default=[5,6])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.npu.set_device(0)
    torch.ops.load_library(str(args.binding.resolve()))
    torch.ops.load_library(str(args.native_library.resolve()))
    shapes = [('fn', (24, 16384)), ('scale', (3,)), ('base', (24,))]
    if args.checkpoint is not None:
        from safetensors import safe_open

        index = json.loads((args.checkpoint / 'quant_model_weights.safetensors.index.json').read_text())['weight_map']
        weights = []
        for suffix, shape in shapes:
            with safe_open(args.checkpoint / index[f'layers.2.hc_attn_{suffix}'], framework='pt', device='cpu') as f:
                weights.append(f.get_tensor(f'layers.2.hc_attn_{suffix}'))
        weight_source = {'mode': 'checkpoint', 'layer': 2, 'branch': 'attn'}
    else:
        generator = torch.Generator().manual_seed(args.weight_seed)
        weights = [torch.randn((24, 16384), generator=generator) * 0.01,
                   torch.ones(3, dtype=torch.float32),
                   torch.randn((24,), generator=generator) * 0.1]
        weight_source = {'mode': 'random', 'seed': args.weight_seed,
                         'distribution': 'fn: normal(0, 0.01); scale: ones; base: normal(0, 0.1)'}
    records = []
    for (suffix, shape), tensor in zip(shapes, weights):
        assert tensor.dtype == torch.float32 and tuple(tensor.shape) == shape
        records.append({'name': f'hc_attn_{suffix}', 'shape': list(shape),
                        'sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
    device_weights=[t.npu() for t in weights]
    report={'physical_device':os.environ.get('TASK_DEVICE'), 'weights':records,'weight_source':weight_source,'seed':'20261009 + B*100 + S','torch':torch.__version__,
            'torch_npu':torch_npu.__version__,'cases':[]}
    for b in args.batch_size:
        for s in args.seqlen:
            t=b*s; name=f'B{b:02}_S{s}_T{t:03}'
            host=torch.randn(t,4,4096,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(20261009+b*100+s))
            x=host.npu()
            runners={
                'native':lambda:torch.ops.custom.npu_hc_pre(x.view(b,s,4,4096),*device_weights,
                    hc_mult=4,hc_sinkhorn_iters=20,norm_eps=1e-6,hc_eps=1e-6),
                'pto':lambda:torch.ops.hcpre_pto.hc_pre(x,*device_weights)}
            row={'B':b,'S':s,'T':t,'name':name};report['cases'].append(row)
            expected=reference(host,*weights)
            eager={k:fn() for k,fn in runners.items()}
            row['pto_vs_native']=compare(eager['pto'],eager['native'],require_pass=False)
            row['cpu']={k:compare(v,expected,require_pass=False) for k,v in eager.items()}
            graphs={}; outputs={}
            for k,fn in runners.items():
                for _ in range(5):fn()
                torch.npu.synchronize()
                g=torch.npu.NPUGraph()
                with torch.npu.graph(g):outputs[k]=fn()
                g.replay();torch.npu.synchronize()
                row[k+'_graph_vs_eager']=compare(outputs[k],eager[k])
                graphs[k]=g
            changed=host*-0.5
            x.copy_(changed)
            new_expected=reference(changed,*weights)
            for k,g in graphs.items():
                g.replay();torch.npu.synchronize()
                row[k+'_updated_cpu']=compare(outputs[k],new_expected,require_pass=False)
                row[k+'_updated_eager']=compare(outputs[k],runners[k]())
            x.copy_(host);torch.npu.synchronize()
            # Alternate sides; events are pre-created/warmed to avoid first-use event cost.
            events=[torch.npu.Event(enable_timing=True) for _ in range(2)]
            for e in events:e.record()
            torch.npu.synchronize()
            for _ in range(10):
                for g in graphs.values():g.replay()
            torch.npu.synchronize()
            samples={k:[] for k in graphs}
            for i in range(20):
                for k in (('native','pto') if i%2==0 else ('pto','native')):
                    events[0].record();graphs[k].replay();events[1].record();events[1].synchronize()
                    samples[k].append(events[0].elapsed_time(events[1])*1000)
            row['replay']={k:{'median_us':statistics.median(v),'samples_us':v} for k,v in samples.items()}
            row['device']={}
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            for k,kernel in [('native','HcPre'),('pto','HcPrePto')]:
                for _ in range(5):graphs[k].replay()
                torch.npu.synchronize()
                row['device'][k]=profile(graphs[k],args.output/name/k,kernel)
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name, {k:v['status'] for k,v in row['device'].items()},flush=True)
            del outputs,graphs,eager,runners
    failures=[r['name'] for r in report['cases'] if not all(x['passed'] for x in r['pto_vs_native'].values())]
    print('PTO vs Native failed cases:',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
