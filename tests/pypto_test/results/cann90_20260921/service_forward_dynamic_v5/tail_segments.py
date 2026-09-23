import argparse, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'tests/pypto_test'))
from dsv4_csa_env import activate, load_native_extension, write_json
repo=activate()
import torch
import pypto.torch
from dsv4_csa_native_fixture import make_config, make_attention, native_session
from dsv4_csa_native_layout import load_layer_weights
from dsv4_csa_native_forward import make_numerical_fixture, execute_native
from dsv4_csa_precision_kernels import diagnose_main_query, diagnose_indexer_query, T_PAD
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_adapter import prepare_weights
p=argparse.ArgumentParser()
p.add_argument('--device',type=int,required=True)
p.add_argument('--stage',choices=['qkv','indexer'],required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args()
report={'status':'FAIL','stage':a.stage,'scope':'tail execution isolation only'}
try:
    checkpoint=Path('/data/model/DeepSeek-V4-Flash-0731-w8a8')
    load_native_extension(repo)
    config=make_config(checkpoint)
    device=torch.device(f'npu:{a.device}')
    with native_session(config,a.device),torch.inference_mode():
        attention=make_attention(config,device)
        load_layer_weights(attention,checkpoint)
        f=make_numerical_fixture(config,device,attention,1,131071,1024)
        execute_native(config,attention,f)
        md=f['metadata'][f['groups']['compressed']['prefix']].req_metadata
        layer=attention.dsa_attn.dsa_attn.layer_name
        cos=md.cos[layer][:6].view(6,64)
        sin=md.sin[layer][:6].view(6,64)
        h=f['metadata'][f['groups']['indexer']['prefix']].hadamard
        w=prepare_weights(attention,h)
        pypto.torch.init(device=a.device,platform='a2a3',runtime='tensormap_and_ringbuffer')
        qr=torch.empty((6,1024),dtype=torch.int8,device=device)
        scale=torch.empty((6,1),dtype=torch.float32,device=device)
        print('starting '+a.stage,flush=True)
        if a.stage=='qkv':
            q=torch.empty((6,64,512),dtype=torch.bfloat16,device=device)
            diagnose_main_query(f['hidden'],w['wq_a'],w['wq_b'],w['wq_b_scale'],w['wkv'],cos,sin,w['gamma_cq'],w['gamma_ckv'],q,qr,scale)
            torch.npu.synchronize()
            report['finite']=bool(torch.isfinite(q).all())
        else:
            qa=attention.wq_a(f['hidden'])[0]
            qr,scale=torch.ops._C_ascend.npu_rms_norm_dynamic_quant(qa,w['gamma_cq'],epsilon=1e-6)
            scale=scale.reshape(6,1)
            pre=torch.empty((T_PAD*64,128),dtype=torch.bfloat16,device=device)
            q=torch.empty((T_PAD*64,128),dtype=torch.int8,device=device)
            qs=torch.empty((T_PAD*64,1),dtype=torch.float32,device=device)
            # Indexer expects the signed Native sine used by the CSA producer.
            signed_sin=sin.clone()
            signed_sin[:,0::2].neg_()
            diagnose_indexer_query(f['hidden'],qr,scale,w['idx_wq_b'],w['idx_wq_b_scale'],cos,signed_sin,w['hadamard_idx'],pre,q,qs)
            torch.npu.synchronize()
            report['finite']=bool(torch.isfinite(pre[:6*64]).all())
        assert report['finite']
        report['status']='PASS'
        print(report,flush=True)
except BaseException:
    report['error']=traceback.format_exc()
    raise
finally:
    write_json(a.output_dir/'segments.json',report)
