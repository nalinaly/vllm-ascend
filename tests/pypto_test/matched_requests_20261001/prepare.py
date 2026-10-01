# SPDX-License-Identifier: Apache-2.0
"""从完整三方结果挑选一致请求，复用已审计bank；不重算prefill、不做hash扫描。"""

import argparse
import copy
import gzip
import itertools
import json
from pathlib import Path

from low_acceptance_20261001.reference import extract

CONFIGURATIONS = ("native", "csa_precision_native_hca", "csa_performance_native_hca")
FIELDS = ("tokens", "events", "drafts", "proposed", "accepted", "histogram")


def prepare(source, destination, evidence):
    if (evidence / "selected_reference.json.gz").exists():
        raise FileExistsError("固定参考已存在；重新筛选请使用新的证据目录")
    candidates, native_reference, selected = [], {}, []
    plans, audits = {}, {}
    for shape, batch in (("128k", 24), ("8k", 40)):
        bank = source / f"mixed{shape}_fixed"
        plans[shape] = json.loads((bank / "plan.json").read_text())
        audits[shape] = json.loads((bank / "audit.json").read_text())
        if audits[shape]["status"] != "PASS":
            raise ValueError(f"源bank未通过审计：{bank}")
        rows = {name: {(row["rank"], row["key"]): row
                       for row in extract(source / "five_way_sync" / shape / name)}
                for name in CONFIGURATIONS}
        group = []
        for case in plans[shape]["cases"][:batch]:
            key = case["key"]
            mismatches = {name: sum(any(rows[name][rank, key][field] != rows["native"][rank, key][field]
                                       for field in FIELDS) for rank in range(16))
                          for name in CONFIGURATIONS[1:]}
            accepted = not any(mismatches.values())
            candidates.append({"key": key, "source_shape": shape, "selected": accepted,
                               "mismatched_replicas_vs_native": mismatches})
            if not accepted:
                continue
            reference = rows["native"][0, key]
            if any(any(rows["native"][rank, key][field] != reference[field] for field in FIELDS)
                   for rank in range(1, 16)):
                raise ValueError(f"Native自身跨DP不一致：{key}")
            native_reference.update({(rank, key): rows["native"][rank, key] for rank in range(16)})
            group.append({"case": case, "bank": bank, "shape": shape, "reference": reference})
        selected.append(group)
    # 保留各长度内部顺序；交错长短请求，检验重新组批后的异长请求处理。
    chosen = [item for pair in itertools.zip_longest(*selected) for item in pair if item is not None]
    if not chosen:
        raise ValueError("没有三方全16rank一致的请求")
    if len(chosen) != 9:
        raise ValueError("本次固定复现为B9；候选集合变化时须创建新的复现口径")
    for shape in plans:
        for field in ("model", "layout", "prefill", "decode"):
            if plans[shape][field] != plans["128k"][field]:
                raise ValueError(f"两种bank的{field}不一致")
    destination.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=True)
    plan = copy.deepcopy(plans["128k"])
    plan["cases"] = [item["case"] for item in chosen]
    plan.pop("extended_from_bank", None)
    plan.update(inherited_prefix_bank=[str(item["bank"].resolve()) for item in chosen],
                selection_scope="三方全16rank逐token、计数、直方图、逐轮事件一致；重新组为单个异长B9批次")
    audit_rows, selections = [], []
    for item in chosen:
        case, bank, row = item["case"], item["bank"], item["reference"]
        tokens = bank / case["tokens"]
        manifest = json.loads((bank / case["key"] / "tp0/manifest.json").read_text())
        if len(json.loads(tokens.read_text())) != manifest["history"] + 1 or manifest["history"] != case["history"]:
            raise ValueError(f"源bank边界不符：{case['key']}")
        (destination / case["tokens"]).symlink_to(tokens.resolve())
        (destination / case["key"]).symlink_to((bank / case["key"]).resolve(), target_is_directory=True)
        audit_rows.append(next(entry for entry in audits[item["shape"]]["cases"] if entry["key"] == case["key"]))
        selections.append({"key": case["key"], "source_shape": item["shape"], "source_bank": str(bank.resolve()),
                           "kind": case["question_kind"], "history": case["history"],
                           "drafts": row["drafts"], "accepted": row["accepted"], "histogram": row["histogram"],
                           "mean_accepted": row["accepted"] / row["drafts"],
                           "rejection_rounds": [[index + 1, proposed, accepted]
                                                for index, (proposed, accepted) in enumerate(row["events"])
                                                if accepted < proposed],
                           "has_rejection": any(accepted < proposed for proposed, accepted in row["events"])})
    (destination / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (destination / "audit.json").write_text(json.dumps({
        "status": "PASS", "cases": audit_rows,
        "scope": "复用已审计bank的同一物理文件，仅检查所选请求边界；不重新扫描cache内容",
        "source_audits": [str((source / f"mixed{shape}_fixed/audit.json").resolve()) for shape in plans],
    }, ensure_ascii=False, indent=2) + "\n")
    report = {"source": str(source.resolve()), "bank": str(destination.resolve()), "batch": len(chosen),
              "ranks": 16, "tokens_per_request": 192, "configurations": list(CONFIGURATIONS),
              "fields": list(FIELDS), "candidates": candidates, "selected": selections,
              "scope": "按历史一致性筛出的正向复现，不能替代低接受率失败基准，不能据此排除其他输入的功能错误",
              "regrouping": "将3条128K和6条8K请求交错放入B9，图档位6/54；此变化必须重新实测"}
    (evidence / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with gzip.open(evidence / "selected_reference.json.gz", "xt", encoding="utf-8") as stream:
        json.dump({"schema": 2, "source": str(source.resolve()), "rows": [
            native_reference[rank, item["case"]["key"]] for rank in range(16) for item in chosen
        ]}, stream, ensure_ascii=False)
    print(json.dumps({"batch": len(chosen), "selected": selections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.bank, args.evidence)
