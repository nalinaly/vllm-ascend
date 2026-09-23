"""Replay the production main compressor from a captured continuous step."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path
from dsv4_csa_env import activate, write_json

REPO = activate()
import pypto.language as pl
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_compressor_ratio4 import (
    B_DYN, T_DYN, D, HEAD_DIM, OUT_DIM, BS_PAD, COMPRESS_RATIO, ROPE_HEAD_DIM,
    COMPRESS_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN, STATE_TABLE_COLUMNS_DYN,
    COMPRESSED_ROWS_DYN, CMP_BLOCK_NUM_DYN, BLOCK_SIZE,
    compressor_ratio4_project, compressor_ratio4_pool_projected, compressor_ratio4_cache_write,
)

@pl.jit(auto_scope=False)
def project(x: pl.Tensor[[T_DYN, D], pl.BF16], wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
            wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
            values: pl.Out[pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32]],
            scores: pl.Out[pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32]]):
    x.bind_dynamic(0, T_DYN)
    gate = pl.system.task_dummy(deps=[])
    compressor_ratio4_project(x, wkv, wgate, values, scores, gate)
    return values, scores

@pl.jit(auto_scope=False)
def postproject(
    state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
    tables: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32],
    values: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32], scores: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32], norm: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[COMPRESSED_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[COMPRESSED_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32], offsets: pl.Tensor[[B_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64], state_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    cache_slots: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32],
    cache: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    pooled: pl.Out[pl.Tensor[[BS_PAD, HEAD_DIM], pl.FP32]],
    normalized: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
):
    state.bind_dynamic(0, COMPRESS_STATE_BLOCK_NUM_DYN)
    state.bind_dynamic(1, STATE_PAGE_ELEMENTS_DYN)
    tables.bind_dynamic(0, B_DYN)
    tables.bind_dynamic(1, STATE_TABLE_COLUMNS_DYN)
    offsets.bind_dynamic(0, B_DYN)
    positions.bind_dynamic(0, T_DYN)
    state_slots.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    sin.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    cache_slots.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    cache.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    normalized.bind_dynamic(0, T_DYN)
    gate = pl.system.task_dummy(deps=[])
    pooled_tid, _ = compressor_ratio4_pool_projected(state, tables, ape, positions, pooled, values, scores, gate)
    compressor_ratio4_cache_write(normalized, pooled, norm, cos, sin, offsets, cache, cache_slots,
        state, tables, ape, values, scores, positions, state_slots, pooled_tid, pooled_tid)
    return state, cache, pooled, normalized


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--candidate', action='store_true')
    args = p.parse_args()
    report = {'status': 'FAIL', 'scope': 'captured main compressor only; no full-trajectory acceptance'}
    try:
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_full_compare import compare_tensor
        torch.set_num_threads(2)
        torch.npu.set_device(args.device)
        device = torch.device(f'npu:{args.device}')
        pypto.torch.init(device=args.device, platform='a2a3', runtime='tensormap_and_ringbuffer')
        d = torch.load(args.capture, map_location='cpu', weights_only=False)
        pos = d['position_ids']; rows = pos.numel(); tokens = torch.nonzero((pos + 1) % 4 == 0).flatten()
        ape = d['cmp_ape']; slot = d['state_slot_mapping'].long()
        current = d['native.state_after'][slot[:,0], :].gather(1, slot[:,1,None] * 2048 + torch.arange(2048))
        native_values = torch.zeros((BS_PAD,OUT_DIM),dtype=torch.float32)
        native_scores_ape = torch.zeros_like(native_values)
        native_values[:rows] = current[:,:OUT_DIM]; native_scores_ape[:rows] = current[:,OUT_DIM:]
        offsets = []; cursor = 0
        for begin, end, length in zip(d['cmp_query_start_loc'][:-1], d['cmp_query_start_loc'][1:], d['cmp_seq_lens']):
            start = int(length - (end - begin)); offsets.append(cursor - start//4 - 1)
            cursor += int(length)//4 - start//4
        offsets = torch.tensor(offsets,dtype=torch.int32)
        slots = d['cmp_slot_mapping'].clone(); valid = d['valid_compact_rows']; n = valid.numel()
        slots[valid,0] = torch.arange(n,dtype=slots.dtype)//32; slots[valid,1] = torch.arange(n,dtype=slots.dtype)%32
        cache = torch.zeros(((n+31)//32,32,1,512),dtype=torch.bfloat16,device=device)
        values = torch.zeros((BS_PAD,OUT_DIM),dtype=torch.float32,device=device);scores = torch.zeros_like(values)
        project(d['x_normed_t'].to(device),d['cmp_wkv'].to(device),d['cmp_wgate'].to(device),values,scores)
        torch.npu.synchronize()
        report['projection'] = {'values':compare_tensor(values[:rows].cpu(),native_values[:rows],0,0),
            'scores_with_ape':compare_tensor(scores[:rows].cpu()+ape[pos%4],native_scores_ape[:rows],0,0)}
        dump = {'pto.values':values.cpu(),'pto.scores':scores.cpu(),'native.values':native_values,'native.scores_with_ape':native_scores_ape,'tokens':tokens}
        report['paths'] = {}
        for label, projected_v, projected_s, pool_ape, before in (
            ('production',values,scores,ape,d['pto.state_before']),
            ('given_native_state',values,scores,ape,d['native.state_before']),
            ('given_native_projection_and_state',native_values.to(device),native_scores_ape.to(device),torch.zeros_like(ape),d['native.state_before']),
        ):
            state = before.to(device); pooled = torch.empty((BS_PAD,512),dtype=torch.float32,device=device)
            normalized = torch.zeros((rows,512),dtype=torch.float32,device=device)
            cache.zero_()
            postproject(state,d['state_block_table'].to(device),projected_v,projected_s,pool_ape.to(device),d['cmp_norm_w'].to(device),
                d['cmp_freqs_cos'].to(device),d['cmp_freqs_sin'].to(device),offsets.to(device),pos.to(device),
                d['state_slot_mapping'].to(device),slots.to(device),cache,pooled,normalized)
            torch.npu.synchronize()
            actual = cache.view(-1,512)[:n].cpu()
            checks = {'vs_saved_pto':compare_tensor(actual,d['pto.cache_written'],0,0),
                      'vs_native':compare_tensor(actual,d['native.cache_written'],0,0)}
            if label == 'production' and not args.candidate: assert checks['vs_saved_pto']['status']=='PASS', checks
            report['paths'][label] = checks
            dump[label+'.pooled'] = pooled.cpu()[tokens];dump[label+'.normalized'] = normalized.cpu()[tokens]
            dump[label+'.cache'] = actual
        args.output_dir.mkdir(parents=True,exist_ok=True)
        torch.save(dump,args.output_dir/'replay.pt')
        report['status'] = 'PASS'
        report['sources'] = {str(q.relative_to(REPO)):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            (Path(__file__).resolve(),REPO/'vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/decode_compressor_ratio4.py')}
        print(json.dumps(report),flush=True)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally: write_json(args.output_dir/'replay.json',report)

if __name__ == '__main__': main()
