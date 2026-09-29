"""两档完整CSA/P95、性能后八类状态，以及Indexer反量化四窗。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        "rope_early_revisit_collect", ROOT.parent / "csa_sparse_first_pv_20260929/collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.TITLE = "仅T144启用RoPE符号生产者提前派发"
    module.TARGETS = {"aiv": "idx_qr_dequant_rope"}
    module.DESCRIPTION = (
        "同一inline函数按T144选择生产者early标志，其他行数False，保留真实依赖；本项是调度候选，"
        "没有改变算术，不把核时随机下降称为核内优化或用其替代完整CSA收益。"
    )
    module.main()
    # 原始profile保留在本地，版本记录只存本试验相关task，避免重复堆积整份泳道解析。
    evidence_path = ROOT / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    related = {"csa_rope_sign", "csa_rope_sign_0", "idx_qr_proj_matmul", "idx_qr_dequant_rope",
               "qr_hadamard_matmul", "qr_hadamard_quant", "qproj_matmul", "qproj_dequant_rms_nope_rope",
               "indexer_head_coefficients", "indexer_score_topk_native_pair_aic",
               "indexer_score_topk_native_pair_aiv", "indexer_topk_query_merge", "qk_pv_aic", "qk_pv_aiv",
               "proj_a_mm", "quant", "proj_b_mm", "proj_b_act_hc_post"}
    for case in evidence["cases"]:
        for side in case["sides"].values():
            for field in ("profile_types_us", "static_descriptor_types", "installed_static_binaries"):
                side.pop(field, None)
            for window in side["windows"]:
                window["tasks"] = {n: t for n, t in window["tasks"].items() if n in related}
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    results = ROOT / "RESULTS.md"
    results.write_text(results.read_text().replace(
        "采用还需依据四窗口分布判断真实核内收益，状态失败则禁止采用；不是Native或模型token/DSpark验收。",
        "这是纯调度试验，采用依据正式完整CSA/P95及实际前置时序；状态失败禁止采用。不是Native或模型验收。"))
    residual = {"scope": "复用本轮DFX诊断已有Native对照；不新增Native计时或设备测试，不冒称模型验收", "cases": []}
    for history, batch in ((131072, 16), (8192, 24)):
        item = {"history": history, "batch": batch, "sides": {}}
        for side in ("baseline", "candidate"):
            path = ROOT / f"h{history}_b{batch}/swimlane/{side}/report.json"
            report = json.loads(path.read_text())
            if set(report["pto_self"]) != module.STATES:
                raise ValueError("DFX自身状态覆盖不足")
            if any(v["status"] != "PASS" for v in report["pto_self"].values()):
                raise ValueError("DFX自身图重放状态差异")
            if any(v["status"] != "PASS" for checks in report["pto_guards"] for v in checks.values()):
                raise ValueError("DFX保护区差异")
            item["sides"][side] = {"source": str(path), "pto_native": report["pto_native"],
                                   "native_self": report["native_self"],
                                   "topk_selection": report["topk_selection"],
                                   "pto_self": "PASS", "pto_guards": "PASS"}
        residual["cases"].append(item)
    (ROOT / "native_reference_residual.json").write_text(json.dumps(residual, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
