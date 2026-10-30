# SPDX-License-Identifier: Apache-2.0
"""用同一图优化前后的保留task ID核对21层CSA的SK合并范围，非性能计时。"""

import argparse
import bisect
import json
import re
from collections import defaultdict
from pathlib import Path


def short_name(name):
    return re.sub(r"_[a-f0-9]{64}_\d+_d0", "", name).replace("static_kernel_", "").replace("__kernel0", "")


def audit(record):
    info = json.loads(record.read_text())
    before = [e for e in json.loads(Path(info["before_sk"]).read_text()) if e.get("ph") == "X"]
    after = [e for e in json.loads(Path(info["after_sk"]).read_text()) if e.get("ph") == "X"]
    streams = defaultdict(list)
    for event in after:
        streams[event["args"]["Stream Id"]].append(event)
    for events in streams.values():
        events.sort(key=lambda e: e["args"]["Task Id"])
    roots = [e for e in before if "CsaSpmdTndDevicePto" in e["name"]]
    if len(roots) != 21 or any("CsaSpmdTndDevicePto" in e["name"] for e in after):
        raise ValueError(f"{record}: 不是21层CSA全部合并的图")
    groups = []
    for root in roots:
        stream, task = root["args"]["Stream Id"], root["args"]["Task Id"]
        events = streams[stream]
        ids = [e["args"]["Task Id"] for e in events]
        index = bisect.bisect_right(ids, task) - 1
        if index < 0 or index + 1 >= len(events):
            raise ValueError(f"{record}: CSA task{task}没有完整保留边界")
        merged = events[index]
        first, limit = ids[index], ids[index + 1]
        if not merged["name"].startswith("sk_entry_"):
            raise ValueError(f"{record}: CSA task{task}未映射到SK task")
        members = [e for e in before if e["args"]["Stream Id"] == stream
                   and first <= e["args"]["Task Id"] < limit]
        if not all(e["args"]["Task Type"].startswith("KERNEL") for e in members):
            raise ValueError(f"{record}: 合并范围内存在非计算task，需人工核对")
        if sum("CsaSpmdTndDevicePto" in e["name"] for e in members) != 1:
            raise ValueError(f"{record}: CSA合并成员不唯一")
        groups.append({"stream": stream, "original_csa_task": task, "retained_sk_task": first,
                       "next_retained_task": limit, "sk_entry": merged["name"],
                       "original_csa_is_static": root["name"].startswith("static_kernel_"),
                       "members": [{"task": e["args"]["Task Id"], "name": short_name(e["name"])} for e in members]})
    return {"record": str(record), "model_ids": sorted({e["args"]["Model Id"] for e in roots}),
            "before_csa_count": len(roots), "after_unfused_csa_count": 0,
            "all_csa_static": all(g["original_csa_is_static"] for g in groups), "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [audit(path) for path in sorted(args.directory.glob("*_record.json"))]
    if not reports:
        raise ValueError("没有SK前后图记录")
    result = {"method": "比较同一NPUGraph优化前后保留的Stream Id/Task Id；将被合并的连续原始计算task映射到SK保留锚点。"
                        "仅覆盖含21层CSA的整模型图。debug_dump的ts/dur是显示占位，绝不作为性能值。",
              "graphs": reports}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "groups"} for r in reports], ensure_ascii=False))


if __name__ == "__main__":
    main()
