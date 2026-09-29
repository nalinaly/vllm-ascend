"""复用完整状态和四窗口收集器，目标为Indexer query反量化/RoPE。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        "indexer_rope_collect", ROOT.parent / "csa_sparse_first_pv_20260929/collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.TITLE = "Indexer query整块Gather"
    module.TARGETS = {"aiv": "idx_qr_dequant_rope"}
    module.DESCRIPTION = (
        "满行从连续8x128反量化输入按绝对元素索引Gather成1x512，再恢复8x64；"
        "避免直接展平带128列行距的64列切片。保留逐head、乘法/舍入及尾行。"
    )
    module.main()


if __name__ == "__main__":
    main()
