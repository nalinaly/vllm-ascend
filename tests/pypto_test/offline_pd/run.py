# SPDX-License-Identifier: Apache-2.0
"""Create a reproducible cache bank, run P4x4, then Native/PTO D1x16.

The plan and audit commands are CPU only. Both execution commands require a
16-device task-submit allocation. Logs/artifacts belong to --bank/--output.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


FORMAL_MODEL = "/data/model/DeepSeek-V4-Flash-0731-w8a8"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_plan(bank):
    plan = json.loads((bank / "plan.json").read_text())
    if plan["model"] != FORMAL_MODEL:
        raise ValueError("Only the formal ModelSlim checkpoint is authorized")
    for name, key in (("config.json", "model_config_sha256"),
                      ("quant_model_weights.safetensors.index.json", "weight_index_sha256")):
        digest = hashlib.sha256((Path(FORMAL_MODEL) / name).read_bytes()).hexdigest()
        if digest != plan[key]:
            raise ValueError(f"Checkpoint manifest changed since P plan creation: {name}")
    return plan


def make_plan(args):
    from tokenizers import Tokenizer

    bank = args.bank.resolve()
    bank.mkdir(parents=True, exist_ok=True)
    if (bank / "plan.json").exists():
        raise FileExistsError(bank / "plan.json")
    model = Path(FORMAL_MODEL)
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 75 or any(not (model / shard).is_file() for shard in shards):
        raise ValueError("The complete formal 75-shard ModelSlim checkpoint is required")
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    cases = []
    for history in map(int, args.histories.split(",")):
        if history < 128:
            raise ValueError("Use at least 128 historical tokens for the SWA fixture")
        for variant in range(4):
            text = (f"Document {variant}: This is a deterministic systems engineering workload. "
                    "Explain paged attention, distributed inference and cache consistency. "
                    "缓存需要保留上下文次序、压缩状态和索引。请基于材料继续分析。\n")
            base = tokenizer.encode(text, add_special_tokens=False).ids
            tokens = (base * ((history + 1 + len(base) - 1) // len(base)))[:history + 1]
            key = f"h{history}_v{variant}"
            write_json(bank / f"{key}.tokens.json", tokens)
            cases.append({"key": key, "history": history, "p_dp_rank": variant,
                          "tokens": f"{key}.tokens.json"})
    write_json(bank / "plan.json", {
        "schema": 1, "model": FORMAL_MODEL,
        "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "weight_index_sha256": hashlib.sha256((model / "quant_model_weights.safetensors.index.json").read_bytes()).hexdigest(),
        "weight_shards": len(shards), "weight_bytes": sum((model / s).stat().st_size for s in shards),
        "prefill": {"tp": 4, "dp": 4, "ep": 16},
        "decode": {"tp": 1, "dp": 16, "ep": 16, "speculative_tokens": 5},
        "layout": {"weight_nz_mode": 0, "enable_kv_nz": False, "block_size": 32,
                   "indexer_kv_dtype": "int8", "dtype": "bfloat16"},
        "decode_batches_per_rank": [1, 4, 8, 16, 24, 32, 40],
        "boundary": "P computes H tokens; D loads h(H) and recomputes token H from H+1 prompt tokens",
        "cases": cases,
    })
    print(f"Prepared {len(cases)} P fixtures in {bank}")


def audit(args):
    plan = read_plan(args.bank)
    results = []
    for case in plan["cases"]:
        replicas = [json.loads((args.bank / case["key"] / f"tp{tp}" / "manifest.json").read_text())
                    for tp in range(4)]
        base = replicas[0]
        errors = []
        for tp, other in enumerate(replicas):
            if not (args.bank / case["key"] / f"tp{tp}" / "cache.safetensors").is_file():
                errors.append(f"tp{tp}: missing payload")
            if other["history"] != case["history"] or other["token_sha256"] != base["token_sha256"]:
                errors.append(f"tp{tp}: prefix mismatch")
            if other["entries"] != base["entries"]:
                errors.append(f"tp{tp}: KV/state replica mismatch")
        # Draft layers must be present along with every target layer.
        names = list(base["entries"])
        target_layers = [i for i in range(43) if any(f"layers.{i}." in n for n in names)]
        if len(target_layers) != 43 or not any("draft" in n or "layers.43." in n for n in names):
            errors.append("incomplete target/draft cache layer coverage")
        results.append({"key": case["key"], "tensors": len(names), "errors": errors})
    passed = all(not r["errors"] for r in results)
    write_json(args.bank / "audit.json", {"status": "PASS" if passed else "FAIL", "cases": results})
    if not passed:
        raise RuntimeError("Offline cache audit failed; see audit.json")
    print(f"PASS: {len(results)} cases, four TP replicas, all target/draft groups")


def worker(args):
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.platforms import current_platform

    # Match `vllm serve`: Ascend widens the int8 indexer config and registers
    # its Native model before LLM's dict-to-dataclass validation.
    current_platform.pre_register_and_update()

    plan = read_plan(args.bank)
    prefill = args.command == "prefill"
    cases = [c for c in plan["cases"] if c["p_dp_rank"] == args.rank % 4]
    connector = KVTransferConfig(
        kv_connector="OfflineDSV4Connector", kv_connector_module_path="offline_pd.connector",
        kv_role="kv_producer" if prefill else "kv_consumer",
        kv_connector_extra_config={"bank": str(args.bank.resolve())},
    )
    overrides = {"sliding_window": 128}
    if not prefill and args.backend == "pto":
        overrides["architectures"] = ["PyptoCSADeepseekV4ForCausalLM"]
    llm = LLM(
        model=plan["model"], tokenizer_mode="deepseek_v4", trust_remote_code=True,
        tensor_parallel_size=4 if prefill else 1, enable_expert_parallel=True,
        dtype="bfloat16", quantization="ascend", hf_overrides=overrides,
        max_model_len=max(c["history"] for c in cases) + args.decode_tokens + 32,
        max_num_seqs=1 if prefill else args.batch,
        max_num_batched_tokens=2048 if prefill else max(256, args.batch * 6),
        enable_prefix_caching=False, enforce_eager=True, seed=1024,
        gpu_memory_utilization=0.9, block_size=32,
        attention_config={"indexer_kv_dtype": "int8"},
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        additional_config={"weight_nz_mode": 0, "enable_kv_nz": False, "enable_dsa_cp": False},
        model_loader_extra_config={"enable_multithread_load": True, "num_threads": 16},
        kv_transfer_config=connector, disable_log_stats=False,
        **({} if prefill else {"worker_extension_cls": "offline_pd.observer.OfflineCSAObserver"}),
    )
    print(f"OFFLINE_MODEL_READY role={args.command} dp={args.rank}", flush=True)
    outputs = []
    for case in cases:
        tokens = json.loads((args.bank / case["tokens"]).read_text())
        params = SamplingParams(temperature=0, max_tokens=1 if prefill else args.decode_tokens,
                                ignore_eos=True,
                                extra_args={"kv_transfer_params": {"offline_key": case["key"]}})
        prompt = {"prompt_token_ids": tokens[:-1] if prefill else tokens}
        if not prefill:
            llm.collective_rpc("offline_begin_observation")
        start = time.perf_counter()
        result = llm.generate([prompt] * (1 if prefill else args.batch), params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        observation = None if prefill else llm.collective_rpc("offline_end_observation")
        outputs.append({"key": case["key"], "elapsed_including_io_seconds": elapsed,
                        "output_token_ids": [list(r.outputs[0].token_ids) for r in result],
                        "csa_observation": observation})
        write_json(args.output / f"rank{args.rank}.json", {"role": args.command, "backend": args.backend,
                   "rank": args.rank, "batch": args.batch, "cases": outputs})
        if observation is not None and args.backend == "pto":
            for rank_stats in observation:
                if any(not any(k.startswith("pto_") and v for k, v in counts.items())
                       for counts in rank_stats.values()):
                    raise RuntimeError("At least one target CSA layer never used PTO; see csa_observation")
    # Engine shutdown tears down its owned workers; launcher checks all ranks.
    llm.llm_engine.engine_core.shutdown()


def launch(args):
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    devices = os.environ.get("TASK_DEVICE", "").split(",")
    if len(devices) != 16 or any(not d.isdigit() for d in devices) or len(set(devices)) != 16:
        raise RuntimeError("Run through task-submit --device auto --device-num 16")
    if args.command == "decode":
        report = json.loads((args.bank / "audit.json").read_text())
        if report["status"] != "PASS":
            raise ValueError("P cache bank must pass audit before D loads it")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("rank*.log")):
        raise FileExistsError("Use a fresh --output directory to preserve prior run evidence")
    prefill = args.command == "prefill"
    tp, dp = (4, 4) if prefill else (1, 16)
    children, files = [], []
    try:
        for rank in range(dp):
            env = os.environ.copy()
            env.update({
                "VLLM_DP_SIZE": str(dp), "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": "0",
                "VLLM_DP_MASTER_IP": args.host, "VLLM_DP_MASTER_PORT": str(args.port),
                "ASCEND_RT_VISIBLE_DEVICES": ",".join(devices[rank * tp:(rank + 1) * tp]),
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_V2_MODEL_RUNNER": "0",
                "VLLM_ASCEND_ENABLE_NZ": "0", "OMP_NUM_THREADS": "4", "OMP_PROC_BIND": "false",
                "HCCL_IF_IP": args.host, "HCCL_SOCKET_IFNAME": args.nic,
                "GLOO_SOCKET_IFNAME": args.nic, "TP_SOCKET_IFNAME": args.nic,
                "HCCL_CONNECT_TIMEOUT": "120", "HCCL_EXEC_TIMEOUT": "204",
                "HCCL_BUFFSIZE": "1024", "HCCL_OP_EXPANSION_MODE": "AIV",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", ""),
            })
            env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
            cmd = [sys.executable, str(Path(__file__).resolve()), args.command,
                   "--bank", str(args.bank.resolve()), "--output", str(args.output.resolve()),
                   "--rank", str(rank), "--batch", str(args.batch), "--backend", args.backend,
                   "--decode-tokens", str(args.decode_tokens)]
            file = (args.output / f"rank{rank}.log").open("w")
            files.append(file)
            children.append(subprocess.Popen(cmd, env=env, stdout=file, stderr=subprocess.STDOUT, start_new_session=True))
        while any(child.poll() is None for child in children):
            failed = [(rank, p.returncode) for rank, p in enumerate(children) if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Offline {args.command} rank failed: {failed}; see {args.output}")
            time.sleep(2)
        if any(p.returncode for p in children):
            raise RuntimeError("An offline rank failed")
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for file in files:
            file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "audit", "prefill", "decode"])
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--histories", default="255,4095,32767,131071,131072,131073")
    parser.add_argument("--host", default="192.168.0.106")
    parser.add_argument("--nic", default="enp23s0f3")
    parser.add_argument("--port", type=int, default=29683)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--backend", choices=["native", "pto"], default="native")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args)
    elif args.command == "audit":
        audit(args)
    elif args.rank >= 0:
        worker(args)
    else:
        if args.output is None:
            parser.error("--output is required for prefill/decode")
        launch(args)


if __name__ == "__main__":
    main()
