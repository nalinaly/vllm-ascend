"""复用最新early双档，补五档；沿用Native七档与原报告口径，增加融合收尾明细。"""

import importlib.util
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAIR = ROOT.parent / "csa_indexer_early_chain_20260929"
PREVIOUS = ROOT.parent / "csa_single_root_seven_20260929"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


previous = load("early_seven_previous", PREVIOUS / "collect.py")
CASES = previous.CASES
finished = previous.finished


def add_fused_tail(result):
    lines = ["", "## O-B反量化与HC_post融合任务", "",
             "当前没有独立hc_post；下表补充proj_b_act_hc_post，避免遗漏新版收尾工作。",
             "核时含流水等待，Native HcPost与PTO融合任务工作范围不同，不能直接比较单worker均值。", "",
             "| 档位 | worker数 | 四窗核时均值μs | 各窗范围μs | 首start | 末end | 启动分散 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        tasks = [w["tasks"]["proj_b_act_hc_post"] for w in case["worker_windows"]]
        workers = (case["batch"] * 6 + 3) // 4
        if len(tasks) != 4 or any(t["blocks"] != workers for t in tasks):
            raise ValueError("融合收尾worker覆盖不符")
        kernels = [t["kernel_mean_us"] for t in tasks]
        times = [statistics.mean(t[k] for t in tasks) for k in
                 ("first_start_us", "last_end_us", "start_spread_us")]
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {workers} | {statistics.mean(kernels):.3f} "
                     f"| {min(kernels):.3f}–{max(kernels):.3f} | " + " | ".join(f"{x:.3f}" for x in times) + " |")
    path = ROOT / "TASKS.md"
    path.write_text(path.read_text() + "\n".join(lines) + "\n")


def main():
    previous.ROOT = ROOT
    previous.PAIR = PAIR
    original_load = previous.load

    def with_fused_tail(name, path):
        module = original_load(name, path)
        if path.parent.name == "upstream_725":
            canonical = module.canonical
            module.canonical = lambda task: ("proj_b_act_hc_post" if re.fullmatch(
                r"proj_b_act_hc_post(?:_\d+)?", canonical(task)) else canonical(task))
        if path.parent.name == "csa_native_inplace_seven_20260929":
            original_write = module.write_task_details

            def write_details(result):
                original_write(result)
                add_fused_tail(result)

            module.write_task_details = write_details
        return module

    previous.load = with_fused_tail
    previous.main()
    for name in ("RESULTS.md", "TASKS.md"):
        path = ROOT / name
        text = path.read_text().replace("单根Indexer采用后的PTO七档与现有Native",
                                       "七处early及O-B/HC收尾采用后的PTO七档与现有Native")
        text = text.replace("单根方案的严格同卡局部A/B见../csa_score_single_root_20260929/RESULTS.md。",
                            "七处early的独立同卡A/B见../csa_indexer_early_chain_20260929/RESULTS.md；"
                            "此前O-B及收尾融合的独立收益分别记录，不能把累计变化归因于early单项。")
        text = text.replace("特别是PTO HC_post每worker循环最多4个token，",
                            "特别是PTO融合收尾每worker循环最多4个token，")
        path.write_text(text)


if __name__ == "__main__":
    main()
