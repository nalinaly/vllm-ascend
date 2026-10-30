# SPDX-License-Identifier: Apache-2.0
"""CPU离线处理真实ACLNN参数dump；不启动模型，不安装公共静态核。"""

import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--superkernel", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
command = [
    "op_compiler", "-p", str(args.input.resolve()), "-v", "Ascend910_9391",
    "-j", "64", "-o", str(args.output.resolve()), "-f", "true", "-l", "info",
]
if args.superkernel:
    command.append("--enable_super_kernel")
started = time.monotonic()
with (args.output / "compile.log").open("w") as stream:
    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
record = dict(command=command, returncode=completed.returncode, wall_s=time.monotonic() - started,
              packages=[str(p) for p in args.output.rglob("*.run")])
(args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(completed.returncode)
