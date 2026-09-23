"""Isolate production SWA projection and RMS with formal ModelSlim weights."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

from dsv4_csa_env import activate, write_json

REPO = activate()
import pypto.language as pl
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.qkv_proj_rope import kv_project_native_240, KV_T_DYN, QPROJ_MM_T_DYN

ROWS = pl.dynamic("RMS_BOUNDARY_ROWS")


@pl.jit(auto_scope=False)
def project_240(
    x: pl.Tensor[[KV_T_DYN, 4096], pl.BF16],
    w: pl.Tensor[[4096, 512], pl.BF16],
    output: pl.Out[pl.Tensor[[QPROJ_MM_T_DYN, 512], pl.FP32]],
):
    x.bind_dynamic(0, KV_T_DYN)
    output.bind_dynamic(0, QPROJ_MM_T_DYN)
    ready = pl.system.task_dummy(deps=[])
    output = kv_project_native_240(x, w, output, 0, ready)
    return output


@pl.jit(auto_scope=False)
def rms_candidates(
    x: pl.Tensor[[ROWS, 512], pl.BF16],
    stats: pl.Out[pl.Tensor[[6, ROWS], pl.FP32]],
):
    x.bind_dynamic(0, ROWS)
    stats.bind_dynamic(1, ROWS)
    for block in pl.spmd(pl.tensor.dim(x, 0) // 8):
        row = block * 8
        value = pl.cast(x[row:row + 8, :], pl.FP32)
        sq = pl.mul(value, value)
        half = pl.add(sq[:, :256], sq[:, 256:512])
        quarter = pl.add(half[:, :128], half[:, 128:256])
        eighth = pl.add(quarter[:, :64], quarter[:, 64:128])
        folded = pl.reshape(pl.row_sum(eighth), [1, 8])
        lane = pl.full([8, 64], dtype=pl.FP32, value=0.0)
        chunks = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        for offset in pl.range(0, 512, 64):
            part = sq[:, offset:offset + 64]
            lane = pl.add(lane, part)
            chunks = pl.add(chunks, pl.reshape(pl.row_sum(part), [1, 8]))
        sequential = pl.reshape(pl.row_sum(lane), [1, 8])
        ones = pl.full([1, 8], dtype=pl.FP32, value=1.0)
        stats[0:1, row:row + 8] = folded
        stats[1:2, row:row + 8] = sequential
        stats[2:3, row:row + 8] = pl.div(ones, pl.sqrt(pl.add(pl.mul(folded, 1.0 / 512), 1e-6)))
        stats[3:4, row:row + 8] = pl.div(ones, pl.sqrt(pl.add(pl.mul(sequential, 1.0 / 512), 1e-6)))
        stats[4:5, row:row + 8] = pl.rsqrt(pl.add(pl.mul(folded, 1.0 / 512), 1e-6), high_precision=True)
        stats[5:6, row:row + 8] = pl.rsqrt(pl.add(pl.mul(chunks, 1.0 / 512), 1e-6), high_precision=True)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[25, 28, 29, 39, 42, 44, 46])
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    report = {'status': 'FAIL', 'checkpoint': str(args.checkpoint),
              'scope': 'formal WKV full M240 projection and RMS boundary; no trajectory acceptance', 'steps': {}}
    try:
        import pypto.torch
        import torch
        import torch_npu
        from safetensors import safe_open
        from dsv4_csa_full_compare import compare_tensor
        from dsv4_csa_precision_kernels import diagnose_kv_identity_rope
        assert (args.checkpoint / 'quant_model_description.json').is_file()
        wm = json.loads((args.checkpoint / 'quant_model_weights.safetensors.index.json').read_text())['weight_map']
        def weight(suffix):
            key = 'layers.2.attn.' + suffix
            with safe_open(args.checkpoint / wm[key], framework='pt', device='cpu') as f:
                return f.get_tensor(key).bfloat16()
        torch.set_num_threads(4)
        torch.npu.set_device(args.device)
        device = torch.device(f'npu:{args.device}')
        pypto.torch.init(device=args.device, platform='a2a3', runtime='tensormap_and_ringbuffer')
        original_weight = weight('wkv.weight').to(device)
        w = original_weight.T.contiguous()
        gamma = weight('kv_norm.weight').to(device)
        cos = torch.ones((240, 64), dtype=torch.float32, device=device)
        sin = torch.zeros_like(cos)
        dump = {'gamma': gamma.cpu()}
        with torch.inference_mode():
            for step in args.steps:
                x = torch.randn((240, 4096), generator=torch.Generator().manual_seed(1024 + step), dtype=torch.bfloat16).to(device)
                projected = torch.nn.functional.linear(x, original_weight)
                norm, rstd = torch_npu.npu_rms_norm(projected, gamma, 1e-6)
                pto_projected = torch.empty((240, 512), dtype=torch.float32, device=device)
                project_240(x, w, pto_projected)
                pto = torch.empty_like(projected)
                diagnose_kv_identity_rope(x, w, gamma, cos, sin, pto)
                given_projection, _ = torch_npu.npu_rms_norm(pto_projected.bfloat16(), gamma, 1e-6)
                stats = torch.empty((6, 240), dtype=torch.float32, device=device)
                rms_candidates(projected, stats)
                torch.npu.synchronize()
                v = {'native.projected': projected.cpu(), 'native.normalized': norm.cpu(), 'native.rstd': rstd.cpu(),
                     'pto.projected': pto_projected.cpu().bfloat16(), 'pto.normalized': pto.cpu(), 'stats': stats.cpu()}
                item = {'projected_exact': compare_tensor(v['pto.projected'], v['native.projected'], 0, 0),
                        'normalized_exact': compare_tensor(v['pto.normalized'], v['native.normalized'], 0, 0),
                        'native_rms_given_pto_projection': compare_tensor(given_projection.cpu(), norm.cpu(), 0, 0),
                        'candidates': {}}
                candidates = {name: v['stats'][i] for i, name in [(2, 'fold_div'), (3, 'lane_div'), (4, 'fold_rsqrt'), (5, 'chunk_rsqrt')]}
                candidates.update({name: (v['stats'][i] * (1 / 512) + 1e-6).sqrt().reciprocal()
                                   for i, name in [(0, 'fold_scalar_cpu'), (1, 'lane_scalar_cpu')]})
                for name, inverse in candidates.items():
                    normalized = ((v['native.projected'].float() * inverse[:, None]) * dump['gamma'].float()).bfloat16()
                    item['candidates'][name] = {'rstd': compare_tensor(inverse, v['native.rstd'].reshape(-1), 0, 0),
                                               'norm': compare_tensor(normalized, v['native.normalized'], 0, 0)}
                report['steps'][step] = item
                dump[str(step)] = v
                print(json.dumps({'step': step, 'projected': item['projected_exact']['mismatches'],
                                  'normalized': item['normalized_exact']['mismatches'],
                                  'candidates': {k: [v['rstd']['mismatches'], v['norm']['mismatches']] for k,v in item['candidates'].items()}}), flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dump, args.output_dir / 'swa_rms_boundary.pt')
        report['sources'] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [Path(__file__).resolve(), REPO/'vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/qkv_proj_rope.py']}
        report['status'] = 'PASS'
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / 'swa_rms_boundary.json', report)


if __name__ == '__main__':
    main()
