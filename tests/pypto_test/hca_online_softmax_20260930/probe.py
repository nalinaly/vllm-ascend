"""独立FP64密集参考检查Attention；合成分页/尾块/补位，不代替模型token验收。"""

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from dsv4_csa_env import activate  # noqa: E402

activate()


def prepare(root, baseline=None, candidate=None):
    manifest = json.loads((root / "source.json").read_text())
    if baseline is not None and candidate is not None:
        manifest["sources"] = {"base": str(baseline.resolve()), "online": str(candidate.resolve())}
    paths = {}
    for side, source in manifest["sources"].items():
        dest = Path(source).parent / (side + "_" + root.name + "_probe")
        assert not dest.exists()
        shutil.copytree(source, dest)
        original = (dest / "deepseek_v4_flash_hca/decode_sparse_attn_hca.py").read_text()
        start = original.index("def sparse_attn_hca_tp1(")
        header = original[start:original.index("    raw_cache_ready_dep:", start)]
        header = header.replace("def sparse_attn_hca_tp1(", "def attention_probe(")
        header = header.replace(
            "o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],",
            "o_packed_heads: pl.Out[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16]],",
        )
        body = '''
    q.bind_dynamic(0, T_DYN)
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    ori_block_table.bind_dynamic(0, B_DYN)
    ori_block_table.bind_dynamic(1, ORI_TABLE_COLUMNS_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    q_ready = pl.array.create(4, pl.TASK_ID)
    for group in pl.unroll(4):
        q_ready[group] = pl.system.task_invalid()
    ready = pl.system.task_invalid()
    packed, done = sparse_attn_hca_tp1(
        q, ori_kv, ori_block_table, cmp_kv, cmp_block_table, position_ids,
        kv_seq_lens, attn_sink, freqs_cos, freqs_sin, o_packed_heads,
        ready, ready, q_ready,
    )
    return packed
'''
        # 使用原模块常量和实际Attention入口；参考算法独立在下面实现。
        target = dest / "deepseek_v4_flash_hca/attention_probe.py"
        imports = "from .decode_sparse_attn_hca import *\n\n@pl.jit(auto_scope=False)\n"
        target.write_text(imports + header + "):\n" + body)
        target.chmod(0o444)
        paths[side] = str(dest)
    (root / "probe_sources.json").write_text(json.dumps(paths, indent=2) + "\n")


def load(side, path):
    from hca_pair_screen_20260930.pair import load_variant

    load_variant(Path(path), "probe_" + side)
    module_name = "vllm_ascend.ops.pypto.hca_pair_probe_" + side + ".deepseek_v4_flash_hca.attention_probe"
    return importlib.import_module(module_name)


def fixture(module, history, edge):
    import torch

    generator = torch.Generator().manual_seed(20260930 + history)
    batch, steps, heads, width = 5, 6, 64, 512
    tokens = batch * steps
    positions = torch.arange(history, history + steps).repeat(batch)
    lengths = torch.full((batch,), history + steps, dtype=torch.int32)
    if edge:
        lengths[-1] = 0
    raw_columns = (history + steps + 31) // 32
    cmp_columns = (int(lengths.max()) // 128 + 31) // 32
    raw_table = torch.full((batch, raw_columns), -1, dtype=torch.int32)
    cmp_table = torch.arange(batch * cmp_columns, dtype=torch.int32).reshape(batch, cmp_columns).flip(0).contiguous()
    raw = torch.full((batch * 6, 32, 1, width), float("nan"), dtype=torch.bfloat16)
    compressed = torch.full((batch * cmp_columns, 32, 1, width), float("nan"), dtype=torch.bfloat16)
    for request in range(batch):
        start = history + 1 - 128
        first_page = start // 32
        last_page = (history + steps - 1) // 32
        for index, page in enumerate(range(first_page, last_page + 1)):
            physical = (batch - request - 1) * 6 + index
            raw_table[request, page] = physical
            lo, hi = max(start, page * 32), min(history + steps, (page + 1) * 32)
            raw[physical, lo % 32:lo % 32 + hi - lo, 0] = torch.randn(hi - lo, width, generator=generator).bfloat16()
        for row in range(int(lengths[request]) // 128):
            physical = int(cmp_table[request, row // 32])
            compressed[physical, row % 32, 0] = torch.randn(width, generator=generator).bfloat16()
    if edge:
        cmp_table[1, 1] = -1  # 不连续的无效页，不能把它作为零值有效KV计入分母。
    query = torch.randn(tokens, heads * width, generator=generator).bfloat16()
    sink = torch.linspace(-1, 3, heads)
    angles = torch.randn(tokens, 32, generator=generator).repeat_interleave(2, dim=1)
    cosine, sine = angles.cos(), angles.sin()
    args = [query, raw, raw_table, compressed, cmp_table, positions, lengths, sink, cosine, sine]
    reference = torch.empty(tokens, heads, width, dtype=torch.float64)
    for token in range(tokens):
        request = token // steps
        rows = []
        for row in range(min(int(positions[token]) + 1, int(lengths[request])) // 128):
            page = int(cmp_table[request, row // 32])
            if page >= 0:
                rows.append(compressed[page, row % 32, 0])
        visible = max(min(int(positions[token]) + 1, int(lengths[request])), 0)
        for row in range(max(visible - 128, 0), visible):
            page = int(raw_table[request, row // 32])
            if page >= 0:
                rows.append(raw[page, row % 32, 0])
        if not rows:
            reference[token].zero_()
            continue
        kv = torch.stack(rows).double()
        assert torch.isfinite(kv).all()
        scores = query[token].reshape(heads, width).double() @ kv.T * module.SOFTMAX_SCALE
        probabilities = torch.cat((scores, sink.double()[:, None]), dim=1).softmax(dim=1)[:, :-1]
        value = probabilities @ kv
        rope = value[:, -64:].clone()
        swapped = rope.reshape(heads, 32, 2).flip(-1).reshape(heads, 64)
        sign = torch.tensor([1, -1]).repeat(32)
        value[:, -64:] = rope * cosine[token] + swapped * sine[token] * sign
        reference[token] = value
    return args, reference


def metrics(actual, reference):
    delta = actual.double() - reference
    return {"max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
            "nonfinite": int((~actual.isfinite()).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--experiment-root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.baseline is None) != (args.candidate is None):
        parser.error("baseline和candidate必须同时指定")
    root = args.experiment_root.resolve()
    if args.output is None:
        args.output = root / "probe_result.json"
    if args.prepare:
        prepare(root, args.baseline, args.candidate)
        return
    os.environ.update(VLLM_ASCEND_ENABLE_NZ="2", VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0")
    paths = json.loads((root / "probe_sources.json").read_text())
    modules = {side: load(side, path) for side, path in paths.items()}
    if args.compile_only:
        from pypto.runtime import RunConfig

        for side, module in modules.items():
            module.attention_probe.compile(config=RunConfig(platform="a2a3")).load()
            print("CPU_COMPILE_PASS", side, flush=True)
        return
    assert args.device >= 0 and "TASK_DEVICE" in os.environ
    import pypto.torch
    import torch
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch_npu.npu.set_deterministic_level(0)
    pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    operators = {side: pypto.torch.register(module.attention_probe, f"hca_softmax_probe_{side}::attention")
                 for side, module in modules.items()}
    report = {"scope": "synthetic standalone Attention versus dense FP64; no model token acceptance",
              "device": args.device, "sources": paths, "cases": {}}
    module = modules["online"]
    for history, edge in ((131072, False), (16507, True)):
        inputs, reference = fixture(module, history, edge)
        device_inputs = [x.to(f"npu:{args.device}") for x in inputs]
        outputs = {}
        checks = {}
        for side, operator in operators.items():
            packed = torch.full((module.O_GROUPS * module.T_PAD, module.O_GROUP_IN), 17.0,
                                dtype=torch.bfloat16, device=f"npu:{args.device}")
            operator(*device_inputs, packed)
            torch.npu.synchronize()
            first = packed.cpu()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                operator(*device_inputs, packed)
            graph.replay()
            torch.npu.synchronize()
            actual = packed.cpu()
            assert torch.equal(first, actual)
            shaped = actual.reshape(module.O_GROUPS, module.T_PAD, module.O_GROUP_IN)
            assert torch.all(shaped[:, 30:] == 17)
            outputs[side] = shaped[:, :30].permute(1, 0, 2).reshape(30, 64, 512)
            checks[side] = metrics(outputs[side], reference)
            assert checks[side]["nonfinite"] == 0
            if edge:
                assert torch.count_nonzero(outputs[side][-6:]) == 0
            checks[side].update(graph_exact=True, output_padding_guard=True)
        report["cases"][str(history)] = {"edge": edge, "vs_dense_fp64": checks,
                                        "online_vs_base": metrics(outputs["online"], outputs["base"])}
    report["status"] = "DIAGNOSTIC_COMPLETE"
    report["acceptance"] = "FP64 errors reported without converting a measured error into an acceptance tolerance"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
