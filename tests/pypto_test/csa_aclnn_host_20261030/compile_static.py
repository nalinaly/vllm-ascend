# SPDX-License-Identifier: Apache-2.0
"""CPU离线处理真实ACLNN参数dump；不启动模型，不安装公共静态核。"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--superkernel", action="store_true")
parser.add_argument("--policy-link", action="store_true", help="私有编译扩展合入188个策略对象")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
command = [
    "op_compiler",
    "-p",
    str(args.input.resolve()),
    "-v",
    "Ascend910_9391",
    "-j",
    "64",
    "-o",
    str(args.output.resolve()),
    "-f",
    "true",
    "-l",
    "info",
]
if args.superkernel:
    command.append("--enable_super_kernel")
started = time.monotonic()
environment = os.environ.copy()
here = Path(__file__).resolve().parent
environment["ASCEND_OPP_PATH"] = subprocess.check_output(
    [sys.executable, str(here.parent / "dsv4_hca_prepare_opp.py"), "--destination", str(args.output / "private_opp")],
    text=True,
).strip()
if args.policy_link:
    repo = here.parents[2]
    environment["CSA_DEVICE_POLICY_BUILD"] = str(here.parent / "csa_device_policy_20261029/probe/aclnn_v1")
    environment["PYTHONPATH"] = str(here / "static_link_hook") + ":" + environment.get("PYTHONPATH", "")
    environment["ASCEND_CUSTOM_OPP_PATH"] = environment["ASCEND_CUSTOM_OPP_PATH"].replace(
        str(repo / ".cache/csa/csa-native-ops-install/vendors/custom_transformer"),
        str(here / "results/static_vendor/custom_transformer"),
    )
with (args.output / "compile.log").open("w") as stream:
    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False, env=environment)
record = dict(
    command=command,
    returncode=completed.returncode,
    wall_s=time.monotonic() - started,
    packages=[str(p) for p in args.output.rglob("*.run")],
)
(args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(completed.returncode)
