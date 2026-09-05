# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fresh-process entry point for TRB/HBG A3 CSA validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.pypto_dsv4_decode_csa.a3_smoke import run_a3_zero_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        required=True,
        choices=("tensormap_and_ringbuffer", "host_build_graph"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_a3_zero_smoke(
        runtime=args.runtime,
        device=args.device,
        batch=args.batch,
        artifact_dir=args.artifact_dir,
    ).to_dict()
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
