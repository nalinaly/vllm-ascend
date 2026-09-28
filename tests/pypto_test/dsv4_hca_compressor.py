# SPDX-License-Identifier: Apache-2.0
"""正式层 3 权重的 C128 压缩链对照；仅此子链，不代表 HCA 整层通过。"""

import argparse
import json
import os
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_validation import compare_tensor

activate()

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_hca import decode_compressor_ratio128 as c
from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.decode_compressor_ratio128 import compressor_ratio128


@pl.jit
def hca_compressor_test(
    x: pl.Tensor[[c.T_DYN, c.D], pl.BF16],
    wkv: pl.Tensor[[c.HEAD_DIM, c.D], pl.BF16],
    wgate: pl.Tensor[[c.HEAD_DIM, c.D], pl.BF16],
    ape: pl.Tensor[[c.RATIO, c.HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[c.HEAD_DIM], pl.BF16],
    state: pl.InOut[pl.Tensor[[c.STATE_PAGES_DYN, c.STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
    state_table: pl.Tensor[[c.B_DYN, c.STATE_COLUMNS_DYN], pl.INT32],
    state_slots: pl.Tensor[[c.T_DYN, 2], pl.INT32],
    positions: pl.Tensor[[c.T_DYN], pl.INT64],
    query_bounds: pl.Tensor[[c.BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[c.B_DYN], pl.INT32],
    cos: pl.Tensor[[c.COMPACT_ROWS_DYN, c.ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[c.COMPACT_ROWS_DYN, c.ROPE_DIM], pl.FP32],
    cmp_slots: pl.Tensor[[c.COMPACT_ROWS_DYN, 2], pl.INT32],
    cmp_cache: pl.InOut[pl.Tensor[[c.CMP_PAGES_DYN, c.CMP_BLOCK, 1, c.HEAD_DIM], pl.BF16]],
):
    x.bind_dynamic(0, c.T_DYN)
    state.bind_dynamic(0, c.STATE_PAGES_DYN)
    state.bind_dynamic(1, c.STATE_PAGE_ELEMENTS_DYN)
    state_table.bind_dynamic(0, c.B_DYN)
    state_table.bind_dynamic(1, c.STATE_COLUMNS_DYN)
    state_slots.bind_dynamic(0, c.T_DYN)
    positions.bind_dynamic(0, c.T_DYN)
    query_bounds.bind_dynamic(0, c.BOUNDS_DYN)
    seq_lens.bind_dynamic(0, c.B_DYN)
    cos.bind_dynamic(0, c.COMPACT_ROWS_DYN)
    sin.bind_dynamic(0, c.COMPACT_ROWS_DYN)
    cmp_slots.bind_dynamic(0, c.COMPACT_ROWS_DYN)
    cmp_cache.bind_dynamic(0, c.CMP_PAGES_DYN)
    ready = pl.system.task_dummy(deps=[])
    compressor_ratio128(
        x, wkv, wgate, ape, norm_w, state, state_table, state_slots, positions,
        query_bounds, seq_lens, cos, sin, cmp_slots, cmp_cache, ready,
    )
    return state, cmp_cache


def formal_weights(checkpoint, device):
    import torch
    from safetensors import safe_open

    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    tensors, records = [], []
    for suffix in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
        key = f"layers.3.attn.compressor.{suffix}"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
            value = reader.get_tensor(key)
        # Native BF16 模型的参数加载会转换投影和 norm；APE 的参数显式声明为 FP32。
        target_dtype = torch.float32 if suffix == "ape" else torch.bfloat16
        tensors.append(value.to(device=device, dtype=target_dtype))
        records.append({"name": key, "shape": list(value.shape), "checkpoint_dtype": str(value.dtype),
                        "loaded_dtype": str(target_dtype), "shard": index[key]})
    return tensors, records


def run_case(kernel, weights, histories, active, device):
    import torch

    batch, tokens = len(histories), len(histories) * c.DECODE_SEQ
    # 每个请求 18 个滚动 state 页，逻辑页反序映射；附加首尾保护区。
    pages_per_request = 18
    pages = batch * pages_per_request + 1
    columns = (max(histories[:active]) + c.DECODE_SEQ + c.STATE_BLOCK - 1) // c.STATE_BLOCK + 1
    table = torch.empty((batch, columns), dtype=torch.int32)
    for request in range(batch):
        table[request] = 1 + request * pages_per_request + (pages_per_request - 1 - torch.arange(columns)) % pages_per_request
    positions = torch.tensor([start + step for start in histories for step in range(c.DECODE_SEQ)], dtype=torch.int64)
    slots = torch.full((tokens, 2), -1, dtype=torch.int32)
    for request in range(active):
        for step in range(c.DECODE_SEQ):
            token = request * c.DECODE_SEQ + step
            pos = int(positions[token])
            slots[token] = torch.tensor([table[request, pos // c.STATE_BLOCK], pos % c.STATE_BLOCK])
    bounds = torch.arange(batch + 1, dtype=torch.int32).clamp_max(active) * c.DECODE_SEQ
    lengths = torch.tensor([history + c.DECODE_SEQ if request < active else 0
                            for request, history in enumerate(histories)], dtype=torch.int32)
    starts = torch.tensor(histories, dtype=torch.int32)
    start_device, bounds_device = starts.to(device), bounds.to(device)
    table_device = table.to(device)
    cmp_columns = max(histories[:active]) // (c.RATIO * c.CMP_BLOCK) + 2
    cmp_table = (1 + torch.arange(batch * cmp_columns, dtype=torch.int32)).reshape(batch, cmp_columns).to(device)
    full_rows = max(histories[:active]) + c.RATIO + c.DECODE_SEQ
    angles = torch.arange(full_rows, device=device, dtype=torch.float32).unsqueeze(1) * torch.linspace(0.0001, 0.01, 32, device=device)
    full_cos, full_sin = angles.cos().repeat_interleave(2, dim=1), angles.sin().repeat_interleave(2, dim=1)
    compressed_count = sum((history + c.DECODE_SEQ) // c.RATIO - history // c.RATIO for history in histories[:active])
    # 沿用 Native builder 的容量公式；未使用行的 slot 为 -1，不伪造恰好等长的表。
    compact_rows = min(tokens, tokens // c.RATIO + batch)
    cos, sin, cmp_slots = torch.ops._C_ascend.compressor_metadata(
        full_cos, full_sin, bounds_device, start_device, cmp_table,
        c.CMP_BLOCK, 2, c.RATIO, compact_rows, active,
    )
    # 精确沿用 Native A3 state 的 [8, 1024] 页；校验完整分配，不能只看压缩输出。
    page_elements = c.STATE_BLOCK * c.STATE_WIDTH
    state_initial = torch.randn((pages * page_elements + 64,), dtype=torch.float32, device=device) * 0.1
    cache_pages = batch * cmp_columns + 1
    cache_elements = cache_pages * c.CMP_BLOCK * c.HEAD_DIM
    cache_initial = torch.randn((cache_elements + 64,), dtype=torch.bfloat16, device=device) * 0.1
    native_state, pto_state = state_initial.clone(), state_initial.clone()
    native_cache, pto_cache = cache_initial.clone(), cache_initial.clone()
    ns = native_state[32:-32].view(pages, c.STATE_BLOCK, c.STATE_WIDTH)
    ps = pto_state[32:-32].view(pages, page_elements)
    nc = native_cache[32:-32].view(cache_pages, c.CMP_BLOCK, 1, c.HEAD_DIM)
    pc = pto_cache[32:-32].view_as(nc)
    x = torch.randn((tokens, c.D), dtype=torch.bfloat16, device=device)
    wkv, wgate, ape, norm_w = weights
    native_output = torch.ops._C_ascend.compressor(
        x, wkv, wgate, ns, ape, norm_w, sin.view(-1, c.ROPE_DIM), cos.view(-1, c.ROPE_DIM),
        state_block_table=table_device, cu_seqlens=bounds_device, seqused=None,
        start_pos=start_device, rope_head_dim=c.ROPE_DIM, cmp_ratio=c.RATIO, coff=1,
        norm_eps=c.M.rms_norm_eps, rotary_mode=2, cache_mode=1,
    )
    if isinstance(native_output, tuple):
        native_output = native_output[0]
    if compact_rows:
        torch.ops._C_ascend.npu_scatter_nd_update_v2(nc, cmp_slots, native_output)
    args = (x, *weights, ps, table_device, slots.to(device), positions.to(device), bounds_device,
            lengths.to(device), cos.view(-1, c.ROPE_DIM), sin.view(-1, c.ROPE_DIM), cmp_slots, pc)
    kernel(*args)
    torch.npu.synchronize()
    comparisons = {
        "state": compare_tensor(pto_state, native_state, 0, 0),
        "compressed_cache": compare_tensor(pto_cache, native_cache, 0, 0),
    }
    guards = {}
    for name, initial, native, actual, rows, width, block in (
        ("state", state_initial, native_state, pto_state, slots, c.STATE_WIDTH, c.STATE_BLOCK),
        ("compressed_cache", cache_initial, native_cache, pto_cache, cmp_slots.cpu(), c.HEAD_DIM, c.CMP_BLOCK),
    ):
        allowed = torch.zeros(initial.numel(), dtype=torch.bool)
        for page, row in rows.tolist():
            if page >= 0 and row >= 0:
                begin = 32 + (page * block + row) * width
                allowed[begin:begin + width] = True
        for source, allocation in (("native", native), ("pto", actual)):
            changed = allocation.cpu().view(torch.uint8).reshape(-1, allocation.element_size()).ne(
                initial.cpu().view(torch.uint8).reshape(-1, allocation.element_size())).any(dim=1)
            guards[f"{source}.{name}"] = int((changed & ~allowed).sum())
    functional_pass = not any(guards.values()) and all(r.get("nonfinite") == 0 for r in comparisons.values())
    return {"batch": batch, "active": active, "histories": histories, "compact_rows": compact_rows,
            "compressed_count": compressed_count,
            "comparisons": comparisons, "outside_slot_elements": guards,
            "status": "MEASURED" if functional_pass else "FAIL"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower-only", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="完成设备代码编译，但不初始化 NPU")
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "C128 压缩子链；正式权重、合成输入与历史，不代表 HCA 整层通过", "status": "RUNNING"}
    try:
        from pypto.runtime import RunConfig
        if args.lower_only or args.build_only:
            if args.build_only:
                artifact = hca_compressor_test.warmup(config=RunConfig(
                    platform="a2a3", save_kernels=True,
                    save_kernels_dir=str((args.output / "build").resolve()),
                ))
                lowered = artifact.program
            else:
                lowered = hca_compressor_test.lower(config=RunConfig(platform="a2a3"))
            (args.output / "lowered.py").write_text(str(lowered))
            report.update(status="PASS", validation="CPU 设备代码编译，无设备执行" if args.build_only else "CPU lowering，无设备执行")
        else:
            if not os.environ.get("TASK_DEVICE"):
                raise RuntimeError("NPU 验证必须通过 task-submit 提交")
            import pypto.torch
            import torch
            import torch_npu  # noqa: F401
            root = Path(os.environ["PTO_EAGER_ROOT"])
            libraries = list((root / "vllm-ascend-dsv4-pto-0251rc1/vllm_ascend").glob("vllm_ascend_C*.so"))
            if len(libraries) != 1:
                raise RuntimeError(f"无法确定已验证的 Native 扩展：{libraries}")
            torch.ops.load_library(str(libraries[0]))
            torch.npu.set_device(0)
            torch.manual_seed(20260928)
            pypto.torch.init(device=0, platform="a2a3", runtime="tensormap_and_ringbuffer")
            kernel = pypto.torch.register(hca_compressor_test, "dsv4_hca_test::compressor")
            weights, report["weights"] = formal_weights(args.checkpoint, "npu:0")
            report["native_extension"] = str(libraries[0])
            report["cases"] = []
            for histories, active in (([124], 1), ([128], 1), ([0, 122, 127, 255], 4), ([124, 255, 128, 999999], 3)):
                with torch.inference_mode():
                    result = run_case(kernel, weights, histories, active, "npu:0")
                report["cases"].append(result)
                write_json(args.output / "report.json", report)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            report["status"] = "MEASURED" if all(case["status"] == "MEASURED" for case in report["cases"]) else "FAIL"
            if report["status"] == "FAIL":
                raise RuntimeError("C128 非有限值或写保护检查失败")
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
