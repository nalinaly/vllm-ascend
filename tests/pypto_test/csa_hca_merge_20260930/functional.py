# SPDX-License-Identifier: Apache-2.0
"""联合整网功能检查：固定 rank/请求覆盖、离线 KV 恢复及 41 层图重放。"""

import argparse
import json
from pathlib import Path

from offline_pd.compare import _load_rank


def validate(root, plan, batch, tokens, ranks=16):
    targets = {f"model.layers.{layer}.self_attn.attn" for layer in range(2, 43)}
    query = plan["decode"]["speculative_tokens"] + 1
    report = {
        "status": "FAIL", "scope": "仅验证联合整网功能，不代表 Native 精度或性能通过",
        "expected": {"ranks": ranks, "batch_per_rank": batch, "decode_tokens": tokens,
                     "csa_layers": 21, "hca_layers": 20},
        "errors": [], "ranks": [],
    }
    for rank in range(ranks):
        try:
            keys = [case["key"] for case in plan["cases"]
                    if case["p_dp_rank"] == rank % plan["prefill"]["dp"]]
            cases = _load_rank(root, "pto", rank, keys, batch, tokens, query - 1)
            data = json.loads((root / f"rank{rank}.json").read_text())
            if data.get("pto_attention") != "both" or data.get("decode_dp") != ranks:
                raise ValueError("联合入口或 DP 数量不符")
            log = (root / f"rank{rank}.log").read_text(errors="replace")
            coverage = []
            for case in cases:
                loaded = log.count(f"OFFLINE_CACHE_LOADED dp={rank} key={case['key']}")
                if loaded < batch:
                    raise ValueError(f"{case['key']}: 离线 KV 恢复记录不足 {batch} 条")
                observations = case.get("csa_observation")
                if not isinstance(observations, list) or len(observations) != 1:
                    raise ValueError("缺少唯一的 TP1 执行观测")
                observed = observations[0]
                selection = observed.get("capture_time_selection", {})
                if (set(observed.get("target_layer_names", [])) != targets
                        or set(selection) != targets):
                    raise ValueError("捕获未覆盖完整 21 个 CSA 与 20 个 HCA 层")
                bucket = f"pto_tokens{batch * query}"
                if not all(selection[layer].get(bucket, 0) > 0 for layer in targets):
                    raise ValueError("实际 B×S 档位存在未选中 PTO 的层")
                replayed = observed.get("model_forward_counts", {}).get(f"FULL_tokens{batch * query}", 0)
                if replayed <= 0:
                    raise ValueError("生成阶段未重放本档位联合 PTO 图")
                coverage.append({"key": case["key"], "restored_requests": loaded,
                                 "target_layers": len(targets), "full_graph_forwards": replayed,
                                 "output_tokens": batch * tokens})
            report["ranks"].append({"rank": rank, "cases": coverage})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report["errors"].append(f"rank{rank}: {exc}")
    if len(report["ranks"]) == ranks and not report["errors"]:
        report["status"] = "PASS"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.root, json.loads((args.bank / "plan.json").read_text()),
                      args.batch, args.decode_tokens)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "ranks"}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
