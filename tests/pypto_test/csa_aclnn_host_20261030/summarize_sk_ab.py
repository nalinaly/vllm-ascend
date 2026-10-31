# SPDX-License-Identifier: Apache-2.0
"""离线汇总Native/PTO×SK0/1；校对实际配置、固定输入和全部稳态样本。"""

import argparse
import json
from pathlib import Path

from summarize_model import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path(__file__).with_name("sk_ab_validation.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.matrix.read_text())
    cases = {}
    inputs = None
    raw = {}
    for case in protocol["cases"]:
        label = case["label"]
        directory = args.matrix.parent / "results" / label
        result = summarize(directory)  # Requires 16 ranks × 10 full T96 samples; keeps outliers.
        records = [json.loads((directory / f"rank{rank}.performance.json").read_text()) for rank in range(16)]
        for rank, record in enumerate(records):
            if record["backend"] != case["backend"] or record["weight_nz_mode"] != 2:
                raise ValueError(f"{label}/rank{rank}: backend或NZ不符")
            if not record["worker_runtime_config"]:
                raise ValueError(f"{label}/rank{rank}:缺少实际worker配置证据")
            for worker in record["worker_runtime_config"]:
                options = worker["decode_optimizations"]["ascend_compilation_config"]
                expected = {"enable_npugraph_ex": True, "enable_static_kernel": True,
                            "inplace_pass": True, "enable_super_kernel": case["sk"]}
                if any(options.get(key) is not value for key, value in expected.items()):
                    raise ValueError(f"{label}/rank{rank}:实际编译配置不符，{options}")
                if worker["cann_event_work_mode"] != 0 or worker["task_queue_enable"] != 1:
                    raise ValueError(f"{label}/rank{rank}:event或Host队列模式不符")
                if case["backend"] == "pto" and not worker["no_pypto_execution"]["passed"]:
                    raise ValueError(f"{label}/rank{rank}:出现Python PTO运行")
        current_inputs = [(r["request_keys"], r["request_histories"]) for r in records]
        if inputs is None:
            inputs = current_inputs
        elif current_inputs != inputs:
            raise ValueError(f"{label}:四组输入不一致，不能作SK开关对照")
        key = f"{case['backend']}_sk{int(case['sk'])}"
        cases[key] = {"directory": str(directory), "task": case["task"], **result}
        raw[key] = records

    pairs = {}
    for backend in ("native", "pto"):
        before, after = (cases[f"{backend}_sk{sk}"] for sk in (0, 1))
        checks = []
        for rank, (left, right) in enumerate(zip(raw[f"{backend}_sk0"], raw[f"{backend}_sk1"])):
            checks.append({
                "rank": rank, "requests": len(left["output_token_ids"]),
                "compared_requests": len(right["output_token_ids"]),
                "all_output_tokens_equal": left["output_token_ids"] == right["output_token_ids"],
                "matching_requests": sum(a == b for a, b in zip(left["output_token_ids"], right["output_token_ids"])),
                "steady_tokens_equal": left["steady_output_token_ids"] == right["steady_output_token_ids"],
                "dspark_counters_equal": left["spec_decode"] == right["spec_decode"],
            })
        pairs[backend] = {
            "sk_saved_ms": before["slowest_rank_mean_ms"] - after["slowest_rank_mean_ms"],
            "sk_saved_percent": 100 * (1 - after["slowest_rank_mean_ms"] / before["slowest_rank_mean_ms"]),
            "accuracy": checks,
        }
    backends = {}
    for sk in (0, 1):
        native, pto = cases[f"native_sk{sk}"], cases[f"pto_sk{sk}"]
        backends[f"sk{sk}"] = {
            "pto_saved_ms": native["slowest_rank_mean_ms"] - pto["slowest_rank_mean_ms"],
            "pto_saved_percent": 100 * (1 - pto["slowest_rank_mean_ms"] / native["slowest_rank_mean_ms"]),
            "pto_timing_threshold_met": pto["slowest_rank_mean_ms"] <= 48,
        }
    report = {"protocol": protocol, "cases": cases, "within_backend_sk_comparison": pairs,
              "same_sk_backend_comparison": backends,
              "scope": "均值为最慢rank的10步均值；P95/max取全部160样本。这里只验证实际配置和计时，"
                       "静态核重选及SK执行覆盖仍须结合独立三步profile，不能由配置推断。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("case\tslowest_rank\tmean_ms\tp95_ms\tmax_ms")
    for key, result in cases.items():
        print(f"{key}\t{result['slowest_rank']}\t{result['slowest_rank_mean_ms']:.6f}\t"
              f"{result['pooled_p95_ms']:.6f}\t{result['pooled_max_ms']:.6f}")
    print(json.dumps({k: {field: value for field, value in v.items() if field != "accuracy"}
                      for k, v in pairs.items()}, ensure_ascii=False))
    print(json.dumps(backends, ensure_ascii=False))


if __name__ == "__main__":
    main()
