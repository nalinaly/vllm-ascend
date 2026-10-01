# SPDX-License-Identifier: Apache-2.0
"""筛选后的请求必须重新执行三方对照；保留重新组批后的参考漂移和功能检查。"""

import argparse
import gzip
import itertools
import json
from pathlib import Path

from low_acceptance_20261001.five_way import compare_rows, validate

CONFIGURATIONS = ("native", "csa_precision_native_hca", "csa_performance_native_hca")


def compact_comparison(left, right):
    result = compare_rows(left, right)
    result["differences"] = [row for row in result.pop("details")
                             if row["token_mismatches"] or row["dspark_fields"]
                             or not row["acceptance_event_sequence_equal"]]
    return result


def guard_reports(root):
    reports, errors = {}, []
    for variant in ("performance", "precision"):
        data = json.loads((root / variant / "report.json").read_text())
        for field in ("native_self", "pto_self"):
            if any(check["status"] != "PASS" for check in data[field].values()):
                errors.append(f"{variant}: {field}不一致")
        for field in ("graph", "padding_graph"):
            if data[field]["status"] != "PASS":
                errors.append(f"{variant}: {field}失败")
        for field in ("native_guards", "pto_guards"):
            if any(check["status"] != "PASS" for call in data[field] for check in call.values()):
                errors.append(f"{variant}: {field}失败")
        if data["status"] != "MEASURED":
            errors.append(f"{variant}: 单卡未完整结束")
        reports[variant] = {field: data[field] for field in (
            "status", "batch", "history", "layer_index", "scope", "native_self", "pto_self",
            "native_guards", "pto_guards", "graph", "padding_graph", "topk_selection",
        )}
    return reports, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--guard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = len(json.loads((args.bank / "plan.json").read_text())["cases"])
    report = {"batch": batch, "configurations": {}, "pairwise": {}, "vs_original_selected_reference": {},
              "errors": [], "scope": "历史三方一致请求重新组批的正向复现；不代替原低接受率失败基准"}
    guards, report["errors"] = guard_reports(args.guard_root)
    guard_detail = args.output.with_name("shape_guards.full.json.gz")
    with gzip.open(guard_detail, "wt", encoding="utf-8") as stream:
        json.dump(guards, stream, ensure_ascii=False)
    report["shape_guards_full_report"] = guard_detail.name
    report["shape_guards"] = {
        variant: {
            "source": str((args.guard_root / variant / "report.json").resolve()),
            "batch": value["batch"], "history": value["history"], "layer_index": value["layer_index"],
            "native_self_exact": all(check["status"] == "PASS" for check in value["native_self"].values()),
            "pto_self_exact": all(check["status"] == "PASS" for check in value["pto_self"].values()),
            "state_fields": list(value["pto_self"]),
            "guards_pass": all(check["status"] == "PASS" for field in ("native_guards", "pto_guards")
                               for call in value[field] for check in call.values()),
            "graph": value["graph"]["status"],
            "graph_inputs": [row["input"] for row in value["graph"]["replays"]],
            "padding_graph": value["padding_graph"]["status"],
            "padding_active_batches": [row["active_batch"] for row in value["padding_graph"]["replays"]],
        } for variant, value in guards.items()
    }
    with gzip.open(args.reference, "rt", encoding="utf-8") as stream:
        reference = json.load(stream)["rows"]
    outputs, configs = {}, {}
    for name in CONFIGURATIONS:
        check, outputs[name], configs[name] = validate(args.root / name, args.bank, batch, name, (1.0, 5.0))
        report["configurations"][name] = check
        report["errors"].extend(f"{name}: {error}" for error in check["validation_errors"])
        if check["acceptance"]["total_preemptions"] != 0:
            report["errors"].append(f"{name}: 有抢占或缺少抢占记录")
        if check["cross_dp_different_outputs_within_run"] or check["cross_dp_different_acceptance_events_within_run"]:
            report["errors"].append(f"{name}: 同次跨DP不一致")
        report["vs_original_selected_reference"][name] = compact_comparison(reference, outputs[name])
        report["configurations"][name]["acceptance_scope"] = (
            "qualified仍按0～5全覆盖判断；本用例是筛选的正向子集，不要求也不宣称低接受率基准合格")
    report["worker_runtime_config_rank0"] = {name: rows[0] for name, rows in configs.items()}
    for name, ranks in configs.items():
        for rank, (actual, native) in enumerate(zip(ranks, configs["native"])):
            def without_event(workers):
                return [{key: value for key, value in worker.items() if key != "cann_event_work_mode"}
                        for worker in workers]
            if without_event(actual) != without_event(native):
                report["errors"].append(f"{name}/rank{rank}: 除event外worker配置不同")
    for left, right in itertools.combinations(CONFIGURATIONS, 2):
        report["pairwise"][left + "__vs__" + right] = compact_comparison(outputs[left], outputs[right])
    report["three_way_status"] = "PASS" if all(r["status"] == "PASS" for r in report["pairwise"].values()) else "FAIL"
    report["original_reference_status"] = (
        "PASS" if all(r["status"] == "PASS" for r in report["vs_original_selected_reference"].values()) else "FAIL")
    report["status"] = ("PASS" if not report["errors"] and report["three_way_status"] == "PASS"
                        and report["original_reference_status"] == "PASS" else "FAIL")
    report["functional_scope"] = (
        "单卡：正式layer_index=2的CSA权重、合成输入/历史，新B9形状自身重复、A/B/A图更新、满档/补位/满档、"
        "cache/state写入保护及只读metadata。整模型：所选真实bank请求、实际算子选择、静态图覆盖、"
        "192token及逐轮事件、无抢占、同次跨DP；不据正向子集证明其他输入没有功能错误。")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "errors": report["errors"],
                      "three_way_status": report["three_way_status"],
                      "original_reference_status": report["original_reference_status"],
                      "pairwise": report["pairwise"],
                      "acceptance": {name: check["acceptance"] for name, check in report["configurations"].items()}},
                     ensure_ascii=False, indent=2))
    raise SystemExit(report["status"] != "PASS")


if __name__ == "__main__":
    main()
