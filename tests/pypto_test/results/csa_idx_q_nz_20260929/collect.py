"""同卡Indexer Q NZ对照，沿用完整状态/四窗覆盖并验证Native权重原地址绑定。"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT.parent / "csa_sparse_first_pv_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("idx_q_nz_collection", path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    common.ROOT = ROOT
    common.TITLE = "Indexer Q投影直接复用Native NZ原地址"
    common.TARGETS = {"idx_q": "idx_qr_proj_matmul"}
    common.DESCRIPTION = (
        "只改Indexer Q的B权重布局与加载期原地址绑定；24份工作、N分块/K顺序及依赖保持。"
        "采用以完整CSA绝对耗时/P95和性能后逐元素检查为准，核内收益单独列出。"
    )
    common.main()
    source = json.loads((ROOT / "source.json").read_text())
    bindings = []
    for history, batch in source["cases"]:
        path = ROOT / f"h{history}_b{batch}/swimlane/candidate/report.json"
        report = json.loads(path.read_text())
        binding = report["weight_storage_binding"]["idx_wq_b"]
        if not (binding["same_data_ptr"] and binding["native_format"] == 29
                and binding["pto_format"] == 29 and binding["root_layout"] == "NZ"
                and binding["shape"] == binding["native_shape"] == [1024, 8192]):
            raise ValueError(f"候选没有复用Native NZ Indexer Q存储: {binding}")
        bindings.append({"history": history, "batch": batch, "source": str(path), "binding": binding})
    (ROOT / "bindings.json").write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
