"""冻结两份完整源码，测试FP32残差流；不改生产或Native模型。"""

import ast
import difflib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-rope-early-revisit-46cec3a0-v1-baseline"
PREFIX = WORKSPACE / ".cache/csa-fp32-residual-46cec3a0-v1"
OLD_PACKAGE = "dsv4_csa_rope_early_revisit_46cec3a0_v1"
PACKAGE = "dsv4_csa_fp32_residual_46cec3a0_v1"
SHARED = Path("vllm_ascend/ops/pypto/deepseek_v4_flash_dspark")
PKG = Path("vllm_ascend/ops/pypto") / PACKAGE


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def change_root(text):
    text = replace_once(text, "x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]",
                        "x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]")
    text = replace_once(text, "x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]]",
                        "x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]]")
    start = text.index("    # 加宽后顺手按原512列次序计算RMS")
    end = text.index("\n    with pl.scope():", start)
    rms = '''    # FP32残差直接供Cube/mix读取，RMS保持现有512列归约次序。
    x_hc32 = x_hc
    x_hc_flat = pl.reshape(x_hc, [t_dim, HC_MULT * D])
    widen_rows = (t_dim + HC_WIDEN_T_TILE - 1) // HC_WIDEN_T_TILE
    hc_padded_rows = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    inv_rms = pl.create_tensor([hc_padded_rows, 1], dtype=pl.FP32)
    with pl.spmd(pl.min(widen_rows, HC_WIDEN_WORKERS), name_hint="hc_rms") as _rms_tid:
        for widen_blk in pl.range(pl.tile.get_block_idx(), widen_rows,
                                  pl.min(widen_rows, HC_WIDEN_WORKERS)):
            w_t0 = widen_blk * HC_WIDEN_T_TILE
            w_rows = pl.min(HC_WIDEN_T_TILE, t_dim - w_t0)
            w_sq_sum = pl.full([1, HC_WIDEN_T_TILE], dtype=pl.FP32, value=0.0)
            for w_db in pl.pipeline(HC_MULT * D // HC_WIDEN_D_TILE, stage=4):
                w_d0 = w_db * HC_WIDEN_D_TILE
                w_val = pl.slice(x_hc_flat, [HC_WIDEN_T_TILE, HC_WIDEN_D_TILE], [w_t0, w_d0],
                                 valid_shape=[w_rows, HC_WIDEN_D_TILE])
                if w_rows == HC_WIDEN_T_TILE:
                    w_sq = pl.mul(w_val, w_val)
                    w_sq_row = pl.reshape(pl.row_sum(w_sq), [1, HC_WIDEN_T_TILE])
                    w_sq_sum = pl.add(w_sq_sum, w_sq_row)
                else:
                    w_valid = pl.set_validshape(w_val, w_rows, HC_WIDEN_D_TILE)
                    w_clean = pl.fillpad(w_valid, pad_value=pl.PadValue.zero)
                    w_sq_tail = pl.mul(w_clean, w_clean)
                    w_sq_row_tail = pl.reshape(pl.row_sum(w_sq_tail), [1, HC_WIDEN_T_TILE])
                    w_sq_sum = pl.add(w_sq_sum, w_sq_row_tail)
            w_mean = pl.add(pl.mul(w_sq_sum, HC_DIM_INV), NORM_EPS)
            w_inv = pl.reshape(pl.rsqrt(w_mean, high_precision=True), [HC_WIDEN_T_TILE, 1])
            inv_rms[w_t0:w_t0 + HC_WIDEN_T_TILE, 0:1] = w_inv
'''
    text = text[:start] + rms + text[end:]
    text = replace_once(text,
                        "    hc 残差流按 Native 的 BF16 收发：43 层里只有 21 层走 PTO，`x_hc` / `x_out`\n"
                        "    要与 Native 的 `npu_hc_pre_v2` / `npu_hc_post` 在层间互传。",
                        "    私有试验：残差FP32收发，移除入口副本及末端BF16舍入；不是现有模型接入入口。")
    return text


def main():
    assert not (ROOT / "task.txt").exists()
    patches = []
    for side in ("baseline", "candidate"):
        dest = Path(str(PREFIX) + "-" + side)
        assert not dest.exists(), dest
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)

        def save(relative, text, dest=dest, side=side):
            path = dest / relative
            before = path.read_text()
            ast.parse(text)
            path.chmod(0o644)
            path.write_text(text)
            patches.extend(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                           fromfile=f"{side}/a/{relative}", tofile=f"{side}/b/{relative}"))

        harness = Path("tests/pypto_test/coefficients_seven_experiment/compiled_case.py")
        text = (dest / harness).read_text()
        if side == "candidate":
            text = replace_once(text, '            output = torch.empty_like(fixture["hidden"])',
                                '            fixture["hidden"] = fixture["hidden"].float()\n'
                                '            output = torch.empty_like(fixture["hidden"])')
        text = replace_once(text, '            topk = {}',
                            '            report["residual_dtype"] = str(fixture["hidden"].dtype)\n'
                            '            report["output_dtype"] = str(output.dtype)\n'
                            '            topk = {}')
        anchor = '            report["mean_us"] = statistics.mean(report["timing"]["samples_us"])'
        follow = '''            # 性能采样完成后才检查跨调用FP32残差；不将copy/验证算入CSA计时。
            from dsv4_csa_validation import compare_tensor

            fixture["hidden"].copy_(output)
            torch.npu.synchronize()
            chain_input = fixture["hidden"].cpu()
            report["residual_chain"] = {
                "scope": "上次attention-half输出作为下次输入，同层权重且cache恢复；非完整Transformer/model",
                "dtype": str(chain_input.dtype),
                "non_bf16_values": int((chain_input.float() != chain_input.bfloat16().float()).sum()),
            }
            restore(fixture)
            eager_call()
            torch.npu.synchronize()
            chain_ref = collect_state(fixture, output, topk_value())
            restore(fixture)
            compiled_call()
            torch.npu.synchronize()
            chain_graph = collect_state(fixture, output, topk_value())
            checks = {name: compare_tensor(chain_graph[name], value, 0, 0) for name, value in chain_ref.items()}
            chain_guards = guard_checks(fixture)
            report["residual_chain"].update(eager_graph_checks=checks, guards=chain_guards)
            if any(v["status"] != "PASS" for v in (*checks.values(), *chain_guards.values())):
                raise ValueError("Chained residual graph or storage guard failed")
            report["residual_chain"]["status"] = "PASS"
            if args.save_state:
                torch.save(chain_graph, args.output / "chain_states.pt")
'''
        text = replace_once(text, anchor, follow + anchor)
        save(harness, text)
        if side == "baseline":
            continue
        save(PKG / "decode_csa.py", change_root((dest / PKG / "decode_csa.py").read_text()))
        text = (dest / PKG / "decode_o_proj.py").read_text()
        assert text.count("pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]") == 4
        text = text.replace("pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]", "pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]")
        save(PKG / "decode_o_proj.py", text)
        text = (dest / SHARED / "hc_post.py").read_text()
        text = text.replace("pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]", "pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]")
        text = replace_once(text, 'single_result = pl.cast(single_y, pl.BF16, mode="rint")', 'single_result = single_y')
        text, count = re.subn(r"pl.cast\((pl.slice\(residual_flat,.*?clamp=True\)), pl.FP32\)", r"\1", text, flags=re.S)
        assert count == 4
        text = text.replace("这里改成与\nvllm-ascend Native 一致的 BF16 收发，在行一级 cast 成 FP32 参与计算。",
                            "本私有试验同样保留FP32残差收发；attention的BF16舍入不变。")
        save(PKG / "hc_post.py", text)
        text = (dest / PKG / "service.py").read_text()
        save(PKG / "service.py", replace_once(text, "hidden.dtype != torch.bfloat16", "hidden.dtype != torch.float32"))
        text = (dest / SHARED / "native_adapter.py").read_text()
        text = replace_once(text, "hidden.dtype != torch.bfloat16", "hidden.dtype != torch.float32")
        text = replace_once(text, 'x_out=empty("x_out", (tokens, 4, 4096), torch.bfloat16)',
                            'x_out=empty("x_out", (tokens, 4, 4096), torch.float32)')
        text = text.replace("CSA expects a contiguous BF16 hc residual stream",
                            "CSA expects a contiguous FP32 hc residual stream")
        save(SHARED / "native_adapter.py", text)
        relative = Path("tests/pypto_test/dsv4_csa_single_layer.py")
        text = (dest / relative).read_text()
        text = replace_once(text, '        call = adapter.NativeCSACall(\n',
                            '        # Native参考已先按BF16执行；只加宽同值输入给私有FP32根。\n'
                            '        fixture["hidden"] = fixture["hidden"].float()\n'
                            '        report["residual_dtype"] = str(fixture["hidden"].dtype)\n'
                            '        call = adapter.NativeCSACall(\n')
        text = replace_once(text,
                            'report["pto_native"] = {name: compare_tensor(pto[0][name], value, 0, 0) '
                            'for name, value in native[0].items()}',
                            'report["pto_native"] = {\n'
                            '            name: compare_tensor(pto[0][name], '
                            'value.float() if name == "x_out" else value, 0, 0)\n'
                            '            for name, value in native[0].items()}')
        save(relative, text)
    (ROOT / "candidate.patch").write_text("".join(patches))
    template = ROOT.parent / "csa_rope_early_revisit_20260929"
    for name in ("compile.py", "run_side.sh"):
        text = (template / name).read_text().replace(template.name, ROOT.name)
        text = text.replace("csa-rope-early-revisit-46cec3a0-v1", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        if name == "run_side.sh":
            text = text.replace("--swimlane-windows 4", "--swimlane-windows 2")
        (ROOT / name).write_text(text)
    (ROOT / "run.sh").write_text('''#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/compile_candidate.json"
for case_spec in 131072:16 8192:24; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    sides=(baseline candidate)
    if [[ "$history" == 8192 ]]; then sides=(candidate baseline); fi
    for side in "${sides[@]}"; do
        bash "$root/run_side.sh" "$side" timing "$history" "$batch"
    done
done
for side in baseline candidate; do
    bash "$root/run_side.sh" "$side" swimlane 131072 16
done
''')
    (ROOT / "source.json").write_text(json.dumps({
        "baseline": "46cec3a0", "source_prefix": str(PREFIX), "variant": "pkg:" + PACKAGE,
        "cases": [[131072, 16], [8192, 24]], "swimlane_cases": [[131072, 16]], "windows": 2,
        "change": "FP32残差入出，RMS沿用512列归约；直接借用输入取消FP32副本；HC_post取消残差转换和末端BF16舍入",
        "unchanged": "attention BF16舍入、matmul/量化/KV/cache/state/metadata/early/sync_start及权重布局",
        "scope": "单卡真实第4层权重合成历史CSA；输入一次加宽在计时外，非整模型FP32收益",
        "accuracy": "性能后比较八类状态；首调用除输出舍入外应相同，再检查FP32输出接回输入的eager/graph",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
