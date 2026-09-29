"""同卡KV NZ对照，沿用完整状态/四窗覆盖并验证Native权重原地址绑定。"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT.parent / "csa_sparse_first_pv_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("kv_nz_collection", path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    common.ROOT = ROOT
    common.TITLE = "KV投影直接复用Native NZ原地址"
    common.TARGETS = {"kv": "kv_proj_matmul", "score_aic": "indexer_score_topk_native_pair_aic",
                      "score_aiv": "indexer_score_topk_native_pair_aiv",
                      "sparse_aic": "qk_pv_aic", "sparse_aiv": "qk_pv_aiv"}
    common.DESCRIPTION = (
        "两侧保持相同M/N/K分块及工作编号，长B16为12份、短B24为8份KV工作。"
        "候选按原1/2/3 M组显式constexpr化以满足NZ偏移证明，完整CSA含新增编排选择成本。"
        "Native与PTO的分工不同，不能把其单核均值直接相减；此处比较PTO同工作量A/B。"
    )
    common.main()
    source = json.loads((ROOT / "source.json").read_text())
    bindings = []
    for history, batch in source["cases"]:
        path = ROOT / f"h{history}_b{batch}/swimlane/candidate/report.json"
        report = json.loads(path.read_text())
        binding = report["weight_storage_binding"]["wkv"]
        if not (binding["same_data_ptr"] and binding["native_format"] == 29
                and binding["pto_format"] == 29 and binding["root_layout"] == "NZ"
                and binding["shape"] == binding["native_shape"] == [512, 4096]):
            raise ValueError(f"候选没有复用Native NZ KV存储: {binding}")
        bindings.append({"history": history, "batch": batch, "source": str(path), "binding": binding})
    (ROOT / "bindings.json").write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
