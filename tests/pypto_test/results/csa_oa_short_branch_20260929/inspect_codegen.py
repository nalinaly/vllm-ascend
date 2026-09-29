"""核对分支使用已测K256/K512核体，记录实际长度扫描，无设备访问。"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def normalize(path):
    text = re.sub(r' loc\("[^"\n]*"\:\d+:\d+\)', "", path.read_text())
    text = re.sub(r"@proj_a_mm(?:_\d+)?", "@kernel", text)
    names = {}

    def rename(match):
        value = match.group()
        if value not in names:
            names[value] = "%v" + str(len(names))
        return names[value]

    return re.sub(r"%[A-Za-z0-9_]+", rename, text)


def main():
    baseline = ROOT.parent / "csa_qrope_flat_gather_20260929/compiled/candidate"
    previous = ROOT.parent / "csa_oa_l1k512_20260929/compiled/candidate"
    candidate = ROOT / "compiled/candidate"
    expected = {
        "proj_a_mm": baseline / "ptoas/proj_a_mm.pto",
        "proj_a_mm_0": previous / "ptoas/proj_a_mm_0.pto",
        "proj_a_mm_1": baseline / "ptoas/proj_a_mm_0.pto",
        "proj_a_mm_2": baseline / "ptoas/proj_a_mm_0.pto",
        "proj_a_mm_3": baseline / "ptoas/proj_a_mm_1.pto",
    }
    evidence = {"comparison": "完整PTO IR只规范化SSA名字、函数名字与源码位置；不使用hash", "kernels": {}}
    for name, reference in expected.items():
        path = candidate / "ptoas" / (name + ".pto")
        same = normalize(path) == normalize(reference)
        evidence["kernels"][name] = {"path": str(path), "reference": str(reference), "identical": same}
        if not same:
            raise ValueError(f"核体与已测参考不同，需要审查: {name}")
    read = "get_tensor_data<int32_t>(ext_kv_seq_lens,"
    evidence["orchestration_seq_len_read_sites"] = {
        side: (folder / "orchestration/_decode_csa_tp1_layer.cpp").read_text().count(read)
        for side, folder in (("baseline", baseline), ("candidate", candidate))
    }
    counts = evidence["orchestration_seq_len_read_sites"]
    assert counts["candidate"] == counts["baseline"] - 1
    evidence["scope"] = "原两次整批max扫描上提合为一次，Indexer临时张量作用域保持；编译/link/load另有证据"
    (ROOT / "static_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
