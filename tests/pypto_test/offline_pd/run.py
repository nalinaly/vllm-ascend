# SPDX-License-Identifier: Apache-2.0
"""Create a reproducible cache bank, run P4x4, then Native/PTO D1x16.

The plan and audit commands are CPU only. Both execution commands require a
16-device task-submit allocation. Logs/artifacts belong to --bank/--output.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


FORMAL_MODEL = "/data/model/DeepSeek-V4-Flash-0731-w8a8"
# release 正式配置里第一个 compress_ratio==4 的 target 层，泳道采集默认取它。
FIRST_TARGET_CSA_LAYER = 2


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_plan(bank):
    plan = json.loads((bank / "plan.json").read_text())
    if plan["model"] != FORMAL_MODEL:
        raise ValueError("Only the formal ModelSlim checkpoint is authorized")
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
    import torch
    from safetensors.torch import load_file
    from prefix import cache_contract, prefix_tensor

    plan = read_plan(args.bank)
    config = json.loads((Path(FORMAL_MODEL) / "config.json").read_text())
    expected = cache_contract(config)
    results, payloads = [], []
    for case in plan["cases"]:
        replicas = [json.loads((args.bank / case["key"] / f"tp{tp}" / "manifest.json").read_text())
                    for tp in range(4)]
        base = replicas[0]
        errors, raw_differences = [], {}
        reference_raw, reference_prefix = None, None
        for tp, other in enumerate(replicas):
            payload = args.bank / case["key"] / f"tp{tp}" / "cache.safetensors"
            if not payload.is_file():
                errors.append(f"tp{tp}: missing payload")
                continue
            payloads.append({"path": str(payload.relative_to(args.bank)), "bytes": payload.stat().st_size})
            if other["history"] != case["history"]:
                errors.append(f"tp{tp}: prefix mismatch")
            tensors = load_file(str(payload))
            if set(tensors) != set(expected) or set(other["entries"]) != set(expected):
                errors.append(f"tp{tp}: incomplete target/draft cache coverage")
                continue
            prefixes = {}
            raw_differences[f"tp{tp}"] = []
            for name, ratio in expected.items():
                entry, value = other["entries"][name], tensors[name]
                if (list(value.shape[1:]) != entry["shape"] or str(value.dtype) != entry["dtype"]
                        or entry.get("compress_ratio", ratio) != ratio):
                    errors.append(f"tp{tp}: payload geometry mismatch: {name}")
                fields = ("logical_blocks", "group", "block_size", "shape", "dtype", "stride")
                if any(entry[k] != base["entries"].get(name, {}).get(k) for k in fields):
                    errors.append(f"tp{tp}: replica layout mismatch: {name}")
                prefix = prefix_tensor(value, entry["logical_blocks"], entry["block_size"], case["history"], ratio)
                prefixes[name] = prefix
                if reference_raw is not None:
                    if not torch.equal(value.view(torch.uint8), reference_raw[name].view(torch.uint8)):
                        raw_differences[f"tp{tp}"].append(name)
                    if not torch.equal(prefix.view(torch.uint8), reference_prefix[name].view(torch.uint8)):
                        errors.append(f"tp{tp}: computed-prefix data mismatch: {name}")
            if reference_raw is None:
                reference_raw, reference_prefix = tensors, prefixes
        results.append({"key": case["key"], "tensors": len(base["entries"]), "errors": errors,
                        "raw_payload_differences": raw_differences})
    passed = all(not r["errors"] for r in results)
    write_json(args.bank / "audit.json", {"status": "PASS" if passed else "FAIL", "cases": results,
               "comparison": "bitwise computed prefix; rows beyond H or floor(H/ratio) cleared on restore",
               "payloads": payloads})
    if not passed:
        raise RuntimeError("Offline cache audit failed; see audit.json")
    print(f"PASS: {len(results)} cases, four TP replicas, all target/draft groups")


def generate_round(llm, args, case, limit):
    """每轮都用同一个 offline_key 重新提交，缓存由 connector 从同一 bank 初态恢复。"""
    from vllm import SamplingParams

    tokens = json.loads((args.bank / case["tokens"]).read_text())
    params = SamplingParams(temperature=0, max_tokens=limit, ignore_eos=True,
                            extra_args={"kv_transfer_params": {"offline_key": case["key"]}})
    start = time.perf_counter()
    result = llm.generate([{"prompt_token_ids": tokens}] * args.batch, params, use_tqdm=False)
    return {"elapsed_seconds": time.perf_counter() - start,
            "output_token_ids": [list(r.outputs[0].token_ids) for r in result]}


def diagnose(args, llm, cases):
    """先预热掉首次编译和缓存冷读，再单独开一次诊断窗口。

    profile 采 PyTorch/NPU 数据用于 Native/PTO 结构对照，swimlane 采一次真实 CSA
    调用的 PTO DFX 记录。两者都带同步和采集开销，各自单独运行，不互相混用，
    也不作为稳态延迟或吞吐结论。
    """
    # 稳态构成直接取生产入口的 S，不在测试里另写一份常量。
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service_config import QUERY_TOKENS

    case = cases[0]
    expected_tokens = args.batch * QUERY_TOKENS
    warmup = [generate_round(llm, args, case, args.warmup_tokens)["elapsed_seconds"]
              for _ in range(args.warmup_rounds)]
    common = {"command": args.command, "backend": args.backend, "rank": args.rank,
              "batch": args.batch, "key": case["key"], "history": case["history"],
              "warmup_rounds": args.warmup_rounds, "warmup_tokens": args.warmup_tokens,
              "warmup_elapsed_seconds": warmup, "expected_tokens": expected_tokens}
    if args.command == "profile":
        started = llm.collective_rpc("offline_begin_profile", args=(
            str((args.output / "trace").resolve()), args.profile_start_step,
            args.profile_steps, expected_tokens, args.batch))
        measured = generate_round(llm, args, case, args.decode_tokens)
        window = llm.collective_rpc("offline_end_profile")
        common.update({
            "decode_tokens": args.decode_tokens,
            "profiler": {"activities": ["CPU", "NPU"], "profiler_level": "Level1",
                         "with_stack": False, "with_modules": False,
                         "record_shapes": False, "profile_memory": False},
            "started": started, "window": window,
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "output_token_ids": measured["output_token_ids"],
            "scope": "窗口含采集与同步开销，仅用于Native/PTO结构对照，不是稳态性能结论",
        })
    elif args.command == "hostprofile":
        started = llm.collective_rpc("offline_begin_host_profile", args=(
            str((args.output / "host").resolve()), args.profile_start_step,
            args.profile_steps, expected_tokens, args.batch))
        measured = generate_round(llm, args, case, args.decode_tokens)
        window = llm.collective_rpc("offline_end_host_profile")
        common.update({
            "decode_tokens": args.decode_tokens, "started": started, "window": window,
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "output_token_ids": measured["output_token_ids"],
            "scope": "cProfile放大Python调用开销，只用于主机侧相对归因，不与设备耗时相加",
        })
    else:
        started = llm.collective_rpc("offline_begin_swimlane",
                                     args=(args.swimlane_layer, expected_tokens))
        measured = generate_round(llm, args, case, args.warmup_tokens)
        window = llm.collective_rpc("offline_end_swimlane")
        common.update({
            "decode_tokens": args.warmup_tokens, "swimlane_layer": args.swimlane_layer,
            "started": started, "window": window,
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "output_token_ids": measured["output_token_ids"],
            "scope": "DFX诊断窗口带边界同步开销，只用于查看任务依赖，不参与耗时对比",
        })
    write_json(args.output / f"rank{args.rank}.{args.command}.json", common)


def profile_export(args):
    """离线解析采集到的 PROF 原始数据，生成带 device kernel 的 trace_view.json。

    解析是纯 CPU 工作，放在采集任务之外执行：既不占用设备队列，也便于按需重跑。
    原始数据保持不变，同一批数据重复解析得到同样的导出文件。
    """
    from torch_npu.profiler.profiler import analyse

    root = (args.output / "trace").resolve()
    wanted = None if args.profile_ranks == "all" else {f"rank{r}" for r in args.profile_ranks.split(",")}
    exported = []
    for directory in sorted(root.glob("rank*"), key=lambda p: int(p.name.removeprefix("rank"))):
        if wanted is not None and directory.name not in wanted:
            continue
        captures = sorted(directory.glob("*_ascend_pt"))
        if len(captures) != 1:
            raise ValueError(f"Expected one capture under {directory}, found {len(captures)}")
        capture = captures[0]
        output = capture / "ASCEND_PROFILER_OUTPUT"
        if not output.is_dir():
            analyse(profiler_path=str(capture), max_process_number=args.analyse_processes)
        trace = output / "trace_view.json"
        if not trace.is_file():
            raise RuntimeError(f"Offline analysis produced no trace_view.json under {capture}")
        exported.append({"rank": directory.name, "capture": str(capture), "trace_view": str(trace),
                         "trace_view_bytes": trace.stat().st_size,
                         "exported_files": sorted(p.name for p in output.iterdir())})
    if not exported:
        raise ValueError(f"No capture matched --profile-ranks {args.profile_ranks} under {root}")
    report = {"command": "profile-export", "root": str(root), "exported": exported,
              "scope": "仅离线解析已采集数据，不改变原始PROF记录，也不产生新的执行结果"}
    write_json(args.output / "profile_export.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def profile_compare(args):
    """对照两次采集：同一 bank/case/batch 与同一 step 窗口，逐个 device kernel 列差异。

    只汇总解析出来的实际 kernel 记录，不另立分类口径。窗口内含 EP 全互联的等待时间，
    也含采集与同步开销，因此这里给的是结构对照，不是稳态延迟或吞吐结论。
    """
    import csv

    rank = f"rank{args.profile_ranks}"

    def load(backend):
        root = args.output / backend
        meta = json.loads((root / f"{rank}.profile.json").read_text())
        if meta["backend"] != backend or meta["command"] != "profile":
            raise ValueError(f"{root} does not hold a {backend} profile capture")
        entry = next(item for item in json.loads((root / "profile_export.json").read_text())["exported"]
                     if item["rank"] == rank)
        totals, counts, cores = {}, {}, {}
        with (Path(entry["trace_view"]).parent / "kernel_details.csv").open() as handle:
            for row in csv.DictReader(handle):
                name, duration = row["Name"], float(row["Duration(us)"])
                totals[name] = totals.get(name, 0.0) + duration
                counts[name] = counts.get(name, 0) + 1
                cores[row["Accelerator Core"]] = cores.get(row["Accelerator Core"], 0.0) + duration
        return {"meta": meta, "entry": entry, "totals": totals, "counts": counts, "cores": cores}

    native, pto = load("native"), load("pto")
    for key in ("key", "history", "batch", "expected_tokens", "decode_tokens"):
        if native["meta"][key] != pto["meta"][key]:
            raise ValueError(f"Captures differ in {key}; they are not comparable")
    windows = {side["meta"]["backend"]: side["meta"]["window"][0] for side in (native, pto)}
    if {w["profiled_steps"] for w in windows.values()} != {args.profile_steps}:
        raise ValueError(f"Both captures must cover {args.profile_steps} steady steps: {windows}")
    kernels = []
    for name in sorted(native["totals"].keys() | pto["totals"].keys()):
        left, right = native["totals"].get(name, 0.0), pto["totals"].get(name, 0.0)
        kernels.append({"name": name, "native_us": round(left, 1), "pto_us": round(right, 1),
                        "delta_us": round(right - left, 1),
                        "native_calls": native["counts"].get(name, 0),
                        "pto_calls": pto["counts"].get(name, 0)})
    kernels.sort(key=lambda item: abs(item["delta_us"]), reverse=True)
    report = {
        "command": "profile-compare", "rank": rank,
        "fixture": {key: native["meta"][key] for key in ("key", "history", "batch", "expected_tokens")},
        "window": {side: {"steady_step_indices": [s["steady_step_index"] for s in window["window"]],
                          "scheduled_tokens": [s["scheduled_tokens"] for s in window["window"]],
                          "requests": [s["requests"] for s in window["window"]]}
                   for side, window in windows.items()},
        "device_totals_us": {side["meta"]["backend"]: round(sum(side["totals"].values()), 1)
                             for side in (native, pto)},
        "device_kernel_records": {side["meta"]["backend"]: sum(side["counts"].values())
                                  for side in (native, pto)},
        "accelerator_core_us": {side["meta"]["backend"]: {k: round(v, 1) for k, v in sorted(side["cores"].items())}
                                for side in (native, pto)},
        "kernels_by_absolute_delta": kernels[:args.compare_top],
        "output_token_ids_equal": native["meta"]["output_token_ids"] == pto["meta"]["output_token_ids"],
        "trace_view": {side["meta"]["backend"]: {"path": side["entry"]["trace_view"],
                                                 "bytes": side["entry"]["trace_view_bytes"]}
                       for side in (native, pto)},
        "scope": "3个稳态decode step的结构对照；窗口含EP等待、采集与同步开销，不是稳态性能结论",
    }
    write_json(args.output / "profile_comparison.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "kernels_by_absolute_delta"},
                     indent=2, ensure_ascii=False))
    print(f"\n{'kernel':<54}{'native us':>12}{'pto us':>12}{'delta us':>12}")
    for item in kernels[:args.compare_top]:
        print(f"{item['name'][:52]:<54}{item['native_us']:>12.1f}{item['pto_us']:>12.1f}{item['delta_us']:>12.1f}")


def swimlane_export(args):
    """把 DFX 记录转成可直接打开的泳道图，任务名取自本次实际生成的 kernel_config。"""
    import ast
    import shutil

    directory = args.output / "swimlane"
    records, deps = directory / "chip_swimlane_records.json", directory / "deps.json"
    raw = json.loads(records.read_text())
    metadata = raw.get("metadata", {})
    boundaries = metadata.get("run_boundaries", [])
    if len(boundaries) != 1 or metadata.get("dropped_run_boundaries", 0):
        raise ValueError(f"Expected exactly one complete DFX window: {metadata}")
    tasks = raw.get("aicore_tasks", [])
    if not tasks or not deps.is_file():
        raise ValueError("DFX capture produced no AICore task or dependency record")
    table = args.kernel_config
    if table is None:
        # 开了 DFX 的 rank 会单独编译一份 kernel，直接从它自己的日志里取实际路径，
        # build_output 下还有其他 rank 和历史运行的同名产物，不能按时间或数量猜。
        log = (args.output / f"rank{args.swimlane_rank}.log").read_text()
        used = sorted(set(re.findall(r"build_output/_jit__decode_csa_tp1_attention_\w+", log)))
        if len(used) != 1:
            raise ValueError(f"Pass --kernel-config; rank{args.swimlane_rank} log names {len(used)} JIT builds")
        table = Path(used[0]) / "kernel_config.py"
    tree = ast.parse(table.read_text())
    tables = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "KERNELS" for target in node.targets)]
    if len(tables) != 1 or not isinstance(tables[0], ast.List):
        raise ValueError(f"Unexpected KERNELS table in {table}")
    names = {}
    for node in tables[0].elts:
        values = {ast.literal_eval(key): value for key, value in zip(node.keys, node.values)}
        names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
    active = {str(k) for task in json.loads(deps.read_text())["tasks"] for k in task["kernel_ids"] if k >= 0}
    if active - names.keys():
        raise ValueError(f"Capture used kernel ids absent from {table}: {sorted(active - names.keys())}")
    shutil.copy2(table, directory / "kernel_config_source.py")
    name_map = directory / "name_map.json"
    write_json(name_map, {"callable_id_to_name": names})
    merged = directory / "merged_swimlane.json"
    # 转换器会打印任务数、执行与调度占比，存档下来，避免只靠终端回看。
    converted = subprocess.run([sys.executable, "-m", "simpler_setup.tools.swimlane_converter", str(records),
                                "--func-names", str(name_map), "-o", str(merged)],
                               check=True, capture_output=True, text=True)
    (directory / "converter_output.txt").write_text(converted.stdout + converted.stderr)
    events = json.loads(merged.read_text())["traceEvents"]
    slices = sum(event.get("ph") == "X" and event.get("cat") == "event" for event in events)
    if slices < len(tasks):
        raise ValueError(f"Converted {slices} device slices for {len(tasks)} AICore tasks")
    report = {"captured_pypto_launches": len(boundaries), "aicore_task_count": len(tasks),
              "converted_device_slices": slices, "named_callables": len(names),
              "kernel_config_source": str(table),
              "chip_swimlane_records": str(records), "deps_json": str(deps),
              "merged_swimlane": str(merged),
              "scope": "DFX诊断窗口带边界同步开销，只用于查看任务与依赖，不参与耗时对比"}
    write_json(directory / "swimlane_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def worker(args):
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.platforms import current_platform

    # Match `vllm serve` model/config registration before constructing LLM.
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
        enable_prefix_caching=False, enforce_eager=prefill or args.graph_mode == "eager", seed=1024,
        gpu_memory_utilization=0.9, block_size=32,
        # D 侧上线口径是 FULL_DECODE_ONLY，见 dsv4_perf_accuracy_20260827/runtime；
        # eager 只用于定位问题，其每步重入 Python 派发路径，不代表上线表现。
        # draft 与该参考配置一致保持 eager。NZ 当前一定不能开，两个后端都不开。
        **({} if prefill or args.graph_mode == "eager"
           else {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}),
        # Release A3 Native chooses INT8 Indexer storage in its constructor;
        # the upstream release AttentionConfig does not accept an int8 Literal.
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        additional_config={"weight_nz_mode": 0, "enable_kv_nz": False, "enable_dsa_cp": False,
                           **({} if not args.recompute_scheduler
                              else {"recompute_scheduler_enable": True})},
        model_loader_extra_config={"enable_multithread_load": True, "num_threads": 16},
        kv_transfer_config=connector, disable_log_stats=False,
        **({} if prefill else {"worker_extension_cls": "offline_pd.observer.OfflineCSAObserver"}),
    )
    print(f"OFFLINE_MODEL_READY role={args.command} dp={args.rank}", flush=True)
    if args.layout_only:
        write_json(args.output / f"rank{args.rank}.cache_layout.json",
                   llm.collective_rpc("offline_cache_layout"))
        llm.llm_engine.engine_core.shutdown()
        return
    if args.command in ("profile", "hostprofile", "swimlane"):
        diagnose(args, llm, cases)
        llm.llm_engine.engine_core.shutdown()
        return
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
    if args.command in ("decode", "profile", "hostprofile", "swimlane"):
        report = json.loads((args.bank / "audit.json").read_text())
        if report["status"] != "PASS":
            raise ValueError("P cache bank must pass audit before D loads it")
    if args.command == "swimlane" and args.backend != "pto":
        raise ValueError("Swimlane capture requires --backend pto")
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
                # 采集窗口会在各 rank 上做同步和落盘，给集合通信留出等待余量。
                "HCCL_CONNECT_TIMEOUT": "120",
                "HCCL_EXEC_TIMEOUT": "1800" if args.command != "decode" else "204",
                "HCCL_BUFFSIZE": "1024", "HCCL_OP_EXPANSION_MODE": "AIV",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", ""),
            })
            env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
            # DFX 配置必须在模型加载前绑定，只有被选中的 rank 采集泳道。
            if args.command == "swimlane" and rank == args.swimlane_rank:
                env["OFFLINE_PTO_SWIMLANE_DIR"] = str((args.output / "swimlane").resolve())
            cmd = [sys.executable, str(Path(__file__).resolve()), args.command,
                   "--bank", str(args.bank.resolve()), "--output", str(args.output.resolve()),
                   "--rank", str(rank), "--batch", str(args.batch), "--backend", args.backend,
                   "--decode-tokens", str(args.decode_tokens),
                   "--warmup-rounds", str(args.warmup_rounds),
                   "--warmup-tokens", str(args.warmup_tokens),
                   "--profile-start-step", str(args.profile_start_step),
                   "--profile-steps", str(args.profile_steps),
                   "--swimlane-layer", str(args.swimlane_layer),
                   "--graph-mode", args.graph_mode]
            if args.recompute_scheduler:
                cmd.append("--recompute-scheduler")
            if args.layout_only:
                cmd.append("--layout-only")
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
    parser.add_argument("command", choices=["plan", "audit", "prefill", "decode", "profile", "hostprofile",
                                            "profile-export", "profile-compare",
                                            "swimlane", "swimlane-export"])
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
    parser.add_argument("--layout-only", action="store_true", help="加载D模型后仅采集缓存描述符")
    parser.add_argument("--warmup-rounds", type=int, default=1, help="诊断前的预热轮数，排除首次编译与缓存冷读")
    parser.add_argument("--warmup-tokens", type=int, default=96, help="每个预热轮的生成token数")
    parser.add_argument("--profile-start-step", type=int, default=8, help="从第几个稳态decode step开始采集")
    parser.add_argument("--profile-steps", type=int, default=3, help="采集的完整decode step数")
    parser.add_argument("--swimlane-rank", type=int, default=0, help="采集PTO DFX泳道的DP rank")
    parser.add_argument("--swimlane-layer", type=int, default=FIRST_TARGET_CSA_LAYER,
                        help="采集泳道的target C4层序号")
    parser.add_argument("--kernel-config", type=Path, help="swimlane-export用于命名任务的JIT kernel_config.py")
    parser.add_argument("--graph-mode", choices=["full_decode_only", "eager"], default="full_decode_only",
                        help="D侧执行模式；上线口径为FULL_DECODE_ONLY，eager仅用于定位问题")
    parser.add_argument("--recompute-scheduler", action="store_true",
                        help="开启recompute_scheduler_enable；DP>1下PTO图模式需要它才能跳过DP padding")
    parser.add_argument("--profile-ranks", default="0", help="profile-export要解析的DP rank，all表示全部")
    parser.add_argument("--analyse-processes", type=int, default=16, help="离线解析使用的进程数上限")
    parser.add_argument("--compare-top", type=int, default=25, help="profile-compare列出的kernel差异条数")
    args = parser.parse_args()
    if args.layout_only and args.command != "decode":
        parser.error("--layout-only 仅适用于 decode")
    if args.command == "plan":
        make_plan(args)
    elif args.command == "audit":
        audit(args)
    elif args.command in ("profile-export", "profile-compare", "swimlane-export"):
        if args.output is None:
            parser.error(f"--output is required for {args.command}")
        {"profile-export": profile_export, "profile-compare": profile_compare,
         "swimlane-export": swimlane_export}[args.command](args)
    elif args.rank >= 0:
        worker(args)
    else:
        if args.output is None:
            parser.error("--output is required for prefill/decode/profile/swimlane")
        launch(args)


if __name__ == "__main__":
    main()
