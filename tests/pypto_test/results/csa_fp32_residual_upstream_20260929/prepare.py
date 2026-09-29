"""参考pypto-lib的FP32 HC_pre：RMS与矩阵乘同域独立生产，RMS允许early。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
PREVIOUS = ROOT.parent / "csa_fp32_residual_20260929"
OLD_PREFIX = "csa-fp32-residual-46cec3a0-v1"
PREFIX = "csa-fp32-residual-upstream-46cec3a0-v1"
OLD_PACKAGE = "dsv4_csa_fp32_residual_46cec3a0_v1"
PACKAGE = "dsv4_csa_fp32_residual_upstream_46cec3a0_v1"


def main():
    changes = []
    for side in ("baseline", "candidate"):
        source = WORKSPACE / f".cache/{OLD_PREFIX}-{side}"
        dest = WORKSPACE / f".cache/{PREFIX}-{side}"
        assert not dest.exists(), dest
        shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
        if side == "baseline":
            continue
        shared = (packages / "deepseek_v4_flash_dspark/hc_pre.py").read_text()
        node = next(n for n in ast.parse(shared).body
                    if isinstance(n, ast.FunctionDef) and n.name == "hc_pre_gates_from_rms")
        helper = "\n".join(shared.splitlines()[node.decorator_list[0].lineno - 1:node.end_lineno]) + "\n"
        helper = helper.replace("def hc_pre_gates_from_rms(", "def hc_pre_gates_fp32(")
        helper = helper.replace("    inv_rms: pl.Tensor[[HC_PAD_ROWS_DYN, 1], pl.FP32],\n", "")
        helper = helper.replace("共享门控与Sinkhorn算术，复用调用方按原规则生成的RMS统计。",
                                "参考上游在同一inline域生成RMS和linear，二者只读同一FP32残差。")
        upstream = (WORKSPACE / "pypto-lib/models/deepseek_v4_flash_dspark/hc_pre.py").read_text()
        start = upstream.index("    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)")
        end = upstream.index("    # linear: split-K matmul", start)
        rms = upstream[start:end]
        assert rms.count("allow_early_resolve=True") == 1
        # 保留接入侧尾行清零规则；当前两代表档都是完整8行tile。
        rms = rms.replace("                x_sq_tail = pl.mul(x_chunk_tail, x_chunk_tail)",
                          "                x_clean_tail = pl.fillpad(x_chunk_tail, pad_value=pl.PadValue.zero)\n"
                          "                x_sq_tail = pl.mul(x_clean_tail, x_clean_tail)")
        helper = helper.replace("    # linear: split-K matmul", rms + "    # linear: split-K matmul", 1)
        file = packages / PACKAGE / "hc_pre.py"
        before = file.read_text()
        text = before + "\n\n" + helper
        ast.parse(text)
        file.chmod(0o644)
        file.write_text(text)
        changes.extend(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                       fromfile="v1/hc_pre.py", tofile="upstream/hc_pre.py"))
        file = packages / PACKAGE / "decode_csa.py"
        before = file.read_text()
        text = before.replace("    hc_pre_gates_from_rms,", "    hc_pre_gates_fp32,")
        start = text.index("    # FP32残差直接供Cube/mix读取")
        end = text.index("    wb_blocks =", start)
        text = text[:start] + '''    # Upstream FP32 residual flow: independent RMS/linear producers in one gate helper.
    hc_padded_rows = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    with pl.scope():
        pre_val_store = pl.create_tensor([hc_padded_rows, HC_PAD], dtype=pl.FP32)
        hc_pre_gates_fp32(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
            pre_val_store, post_t, comb_t, False,
        )
        hc_mix_norm(x_hc, pre_val_store, attn_norm_w, x_normed_t)
''' + text[end:]
        text = text.replace("    HC_DIM_INV,\n", "").replace("    NORM_EPS,\n", "")
        ast.parse(text)
        file.chmod(0o644)
        file.write_text(text)
        changes.extend(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                       fromfile="v1/decode_csa.py", tofile="upstream/decode_csa.py"))
    (ROOT / "candidate.patch").write_text("".join(changes))
    for name in ("compile.py", "run_side.sh", "collect.py"):
        text = (PREVIOUS / name).read_text().replace(PREVIOUS.name, ROOT.name)
        text = text.replace(OLD_PREFIX, PREFIX).replace(OLD_PACKAGE, PACKAGE)
        if name == "collect.py":
            text = text.replace('"hc_widen_rms", "hc_rms",', '"hc_widen_rms", "hc_rms", "hc_pre_rms",')
            text = text.replace('dfx = read(ROOT / f"h{history}_b{batch}/swimlane/{side}/report.json")',
                                'dfx_root = Path(source["reused_baseline_dfx"]) if side == "baseline" else ROOT\n'
                                '                dfx = read(dfx_root / '
                                'f"h{history}_b{batch}/swimlane/{side}/report.json")')
        (ROOT / name).write_text(text)
    run = (PREVIOUS / "run.sh").read_text().replace("for side in baseline candidate; do", "for side in candidate; do")
    (ROOT / "run.sh").write_text(run)
    source = json.loads((PREVIOUS / "source.json").read_text())
    source.update(source_prefix=str(WORKSPACE / ".cache" / PREFIX), variant="pkg:" + PACKAGE,
                  reused_baseline_dfx=str(PREVIOUS), upstream="pypto-lib2164563: hc_pre.py/hc_pre_gates",
                  previous_fp32_source=str(PREVIOUS),
                  change="FP32残差收发；按上游RMS/linear组织并开启RMS early；保持接入侧尾行、缓存策略与融合HC_post",
                  scope="两档新鲜同卡正式A/B；长档BF16泳道复用前轮，FP32采新两窗，不把跨轮DFX当同轮正式事件")
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
