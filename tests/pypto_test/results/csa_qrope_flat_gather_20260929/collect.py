"""沿用Q反量化目标核与完整状态口径，仅替换候选来源与说明。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        "qrope_collect", ROOT.parent / "csa_qdequant_pair_20260929/collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.TITLE = "Q RoPE满行按展平整块Gather置换"
    module.DESCRIPTION = "逐head和48-worker保持，8行局部索引加行偏移后按1×512 gather；尾行与算术不变。"
    module.main()


if __name__ == "__main__":
    main()
