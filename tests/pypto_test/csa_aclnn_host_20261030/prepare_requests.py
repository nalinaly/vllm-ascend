# SPDX-License-Identifier: Apache-2.0
"""复用已有40问题清单，为Native真实后缀prefill建立固定输入。"""

import runpy
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
tests = here.parent
workspace = here.parents[3]
bank = here / "results/kv"
sys.argv = [
    "prepare_mixed.py", "--source-bank", str(bank / "prefix128k"), "--bank", str(bank / "mixed128k"),
    "--encoding", str(workspace / "csa-repro-20261024-current/vllm/vllm/tokenizers/deepseek_v4_encoding.py"),
    "--record", str(bank / "mixed_questions.json"),
]
runpy.run_path(str(tests / "low_acceptance_20261001/prepare_mixed.py"), run_name="__main__")
sys.argv = ["fixed_bank.py", "prepare", "--source", str(bank / "mixed128k"), "--bank", str(bank / "mixed128k_fixed")]
runpy.run_path(str(tests / "low_acceptance_20261001/fixed_bank.py"), run_name="__main__")
