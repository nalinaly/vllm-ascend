# SPDX-License-Identifier: Apache-2.0
"""复用188路对象编译器，为ACLNN入口构建独立调用约定的策略对象。"""

import argparse
import json
from pathlib import Path

from build_cpp import build

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--jobs", type=int, default=188)
args = parser.parse_args()
raise SystemExit(build(args.output, json.loads((args.output / "cpp_manifest.json").read_text()), args.jobs))
