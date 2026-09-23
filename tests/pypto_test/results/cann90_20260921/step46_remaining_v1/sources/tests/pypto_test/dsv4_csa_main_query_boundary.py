"""Isolate Native Q dequantization and RMS/RoPE from a captured full CSA step."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path
from dsv4_csa_env import activate, load_native_extension, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--pto', action='store_true')
    args = p.parse_args()
    repo = activate()
    report = {'status': 'FAIL', 'scope': 'captured Native QR to Query boundary only'}
    try:
        import torch
        import torch_npu
        from safetensors import safe_open
        from dsv4_csa_full_compare import compare_tensor
        torch.set_num_threads(4)
        torch.npu.set_device(args.device)
        device = torch.device(f'npu:{args.device}')
        load_native_extension(repo)
        d = torch.load(args.capture, map_location='cpu', weights_only=False)
        wm = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
        def weight(suffix):
            key = 'layers.2.attn.' + suffix
            with safe_open(args.checkpoint / wm[key], framework='pt', device='cpu') as f:
                return f.get_tensor(key)
        w = weight('wq_b.weight').T.contiguous()
        # The reference loader materializes these channel scales as BF16 first.
        scale = weight('wq_b.scale').bfloat16().float().reshape(-1)
        qr = d['native.qr']; qs = d['native.qr_scale'].reshape(-1)
        acc = qr.int() @ w.int()
        shape = d['native.query'].shape
        cos = d['cos'].to(device).view(shape[0], 1, 1, 64)
        sin = d['sin'].to(device).view(shape[0], 1, 1, 64)
        gamma = torch.ones(512, dtype=torch.bfloat16, device=device)
        def finish(projected):
            norm, rstd = torch_npu.npu_rms_norm(projected.view(shape), gamma, 1e-6)
            normalized = norm.clone()
            torch.ops._C_ascend.inplace_partial_rotary_mul(norm.unsqueeze(1), cos, sin,
                rotary_mode='interleave', partial_slice=[448, 512])
            return norm, normalized, rstd
        with torch.inference_mode():
            native_projected = torch_npu.npu_quant_matmul(qr.to(device), w.to(device), scale.to(device),
                pertoken_scale=qs.to(device), output_dtype=torch.bfloat16)
            native_query, native_norm, rstd = finish(native_projected)
            torch.npu.synchronize()
            report['native_reproduces_capture'] = compare_tensor(native_query.cpu(), d['native.query'], 0, 0)
            assert report['native_reproduces_capture']['status'] == 'PASS', report
            variants = {'activation_then_weight': (acc.float() * qs[:, None]) * scale,
                        'weight_then_activation': (acc.float() * scale) * qs[:, None],
                        'merged_scales': acc.float() * (qs[:, None] * scale)}
            dump = {'native.projected': native_projected.cpu().view(shape), 'native.normalized': native_norm.cpu(),
                    'native.rstd': rstd.cpu(), 'native.query': native_query.cpu(), 'integer_dot': acc,
                    'weight_scale': scale, 'activation_scale': qs}
            report['orders'] = {}
            for key, value in variants.items():
                projected = value.bfloat16().to(device)
                query, normalized, _ = finish(projected)
                torch.npu.synchronize()
                report['orders'][key] = {'projected': compare_tensor(projected.cpu(), native_projected.cpu(), 0, 0),
                    'query': compare_tensor(query.cpu(), d['native.query'], 0, 0),
                    'rows': {r: compare_tensor(query.cpu()[r], d['native.query'][r], 0, 0) for r in (12,109,224)}}
                dump[key + '.query'] = query.cpu()
                dump[key + '.normalized'] = normalized.cpu()
            if args.pto:
                import pypto.torch
                from dsv4_csa_precision_kernels import diagnose_main_query
                pypto.torch.init(device=args.device, platform='a2a3', runtime='tensormap_and_ringbuffer')
                query = torch.empty(shape, dtype=torch.bfloat16, device=device)
                actual_qr = torch.empty_like(qr, device=device)
                actual_scale = torch.empty((shape[0], 1), dtype=torch.float32, device=device)
                diagnose_main_query(d['hidden'].to(device), weight('wq_a.weight').T.contiguous().to(device),
                    w.to(device), scale.to(device), weight('wkv.weight').T.contiguous().to(device),
                    d['cos'].to(device), d['sin'].to(device), weight('q_norm.weight').to(device),
                    weight('kv_norm.weight').to(device), query, actual_qr, actual_scale)
                torch.npu.synchronize()
                report['pto_candidate'] = compare_tensor(query.cpu(), d['native.query'], 0, 0)
                report['pto_rows'] = {r: compare_tensor(query.cpu()[r], d['native.query'][r], 0, 0) for r in (12,109,224)}
                dump['pto.query'] = query.cpu()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dump, args.output_dir/'query_boundary.pt')
        report['sources'] = {str(Path(__file__).resolve().relative_to(repo)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        report['status'] = 'PASS'
        print(json.dumps(report), flush=True)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir/'query_boundary.json', report)

if __name__ == '__main__':
    main()
