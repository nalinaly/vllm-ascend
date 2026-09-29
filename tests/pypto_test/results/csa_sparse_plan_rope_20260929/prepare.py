"""复用已有rope_cs任务承接提前的SWA计划，避免首版新增16份AIV任务。"""

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def window_with_rope(before, split_plan):
    after = split_plan(before)
    start = after.index("    # Native cosine rows")
    end = after.index("    # QK/PV scratch tensors.", start)
    rope_section = after[start:end]
    declaration_end = rope_section.index("    # 不再依赖 rope_swap")
    declaration = rope_section[:declaration_end]
    body_start = rope_section.index("        for cs_rb in pl.range")
    rope_body = rope_section[body_start:]
    after = after[:start] + after[end:]
    marker = "    # SWA positions/page tables are independent"
    after = after.replace(marker, declaration + marker, 1)
    marker = "    # Both phases use scalar stores"
    after = after.replace(marker, rope_body + marker, 1)
    after = after.replace('with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_slots_window_plan",',
                          'with pl.spmd(pl.min(rope_cs_blocks, ROPE_CS_WORKERS),\n'
                          '                 name_hint="csa_slots_window_rope_plan",')
    after = after.replace("window_plan_tid", "rope_tid")
    assert 'name_hint="rope_cs"' not in after
    assert 'name_hint="csa_slots_window_plan"' not in after
    ast.parse(after)
    return after


def main():
    previous = ROOT.parent / "csa_sparse_plan_split_20260929"
    spec = importlib.util.spec_from_file_location("split_prepare", previous / "prepare.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    original = builder.split_plan
    builder.ROOT = ROOT
    builder.PREFIX = ROOT.parents[4] / ".cache/csa-sparse-plan-rope-7b296153"
    builder.PACKAGE = "dsv4_csa_sparse_plan_rope_7b296153"
    builder.split_plan = lambda before: window_with_rope(before, original)
    builder.main()
    for name in ("run.sh", "collect.py"):
        text = (previous / name).read_text()
        if name == "collect.py":
            text = text.replace("csa_slots_window_plan", "csa_slots_window_rope_plan")
            text = text.replace('["csa_slots_build_valid_qk_plan"])',
                                '["rope_cs", "csa_slots_build_valid_qk_plan"])')
            text = text.replace("Sparse计划拆分：滑窗提前、压缩索引等待Top-K", "复用RoPE符号任务承接SWA计划")
            text = text.replace("额外增加16份SWA计划worker；比较两任务总核时及完整CSA，不把关键链缩短等同算术减少。",
                                "不新增任务数；baseline核时合并原plan与rope_cs，candidate合并window+rope与压缩plan。")
            text = text.replace("额外计划核时和其他任务等待必须保留。", "合并计划/RoPE总核时和其他任务等待必须保留。")
            text = text.replace("计划总核时μs", "计划加RoPE总核时μs")
        (ROOT / name).write_text(text)
    source_path = ROOT / "source.json"
    source = json.loads(source_path.read_text())
    source["change"] = "SWA计划复用已有rope_cs任务提前执行；压缩plan依赖此任务及Top-K，不增加AIV任务数"
    source["comparison_scope"] = "两侧总核时均包含SWA、压缩plan及RoPE符号计算"
    source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
