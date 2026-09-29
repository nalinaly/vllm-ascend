"""PTO候选在同进程内按ABBA交错筛选；正式Native/服务入口验收仍用既有脚本。"""

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
from dsv4_csa_env import activate  # noqa: E402

activate()


def load_variant(source, label):
    import vllm_ascend.ops.pypto  # noqa: F401

    name = "vllm_ascend.ops.pypto.hca_pair_" + label
    spec = importlib.util.spec_from_file_location(
        name,
        source / "__init__.py",
        submodule_search_locations=[str(source)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    root = importlib.import_module(name + ".deepseek_v4_flash_hca.decode_hca")
    adapter = importlib.import_module(name + ".deepseek_v4_flash_hca.native_adapter")
    return root, adapter


def stats(samples):
    return {
        "count": len(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "mean_us": statistics.mean(samples),
        "p50_us": statistics.median(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--padding-graph", action="store_true", help="候选同址图检查满档→补位→满档，不计入性能")
    parser.add_argument("--padding-variant", choices=("base", "candidate"), default="candidate")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("cycles必须为正数")
    if args.padding_graph and args.batch < 2:
        parser.error("padding-graph需要batch至少为2")
    args.output.mkdir(parents=True, exist_ok=True)
    import pypto.torch
    import torch

    sources = {"base": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    modules = {label: load_variant(source, label) for label, source in sources.items()}
    operators = {
        label: pypto.torch.register(root.decode_hca_tp1_layer_test, f"hca_pair_{label}::attention")
        for label, (root, adapter) in modules.items()
    }
    assert str(operators["base"]) != str(operators["candidate"])
    report = {
        "status": "REGISTERED",
        "sources": {k: str(v) for k, v in sources.items()},
        "operators": {k: str(v) for k, v in operators.items()},
        "case": [args.history, args.batch],
        "device": args.device,
        "scope": "same-process PTO-root screening, not Native or production-service acceptance",
    }
    if args.check_only:
        (args.output / "registration.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return
    if args.device < 0:
        raise ValueError("设备执行必须使用task-submit分配的TASK_DEVICE")

    import torch_npu
    from dsv4_csa_native_case import native_session
    from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer, restore
    from dsv4_hca_compiled_case import CHECKPOINT, LAYER_INDEX, SEED, decode_additional_config
    from dsv4_hca_compiled_measure import collect_state, device_interval
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    from vllm_ascend.utils import enable_custom_op

    current_platform.pre_register_and_update()
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_deterministic_level(0)
    if not enable_custom_op():
        raise RuntimeError("Native metadata自定义算子不可用")
    tokens = args.batch * 6
    additional = decode_additional_config([tokens])
    additional.update(weight_nz_mode=2, enable_kv_nz=False, enable_dsa_cp=False)
    config = EngineArgs(
        model=CHECKPOINT,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        dtype="bfloat16",
        quantization="ascend",
        tensor_parallel_size=1,
        hf_overrides={"sliding_window": 128},
        max_model_len=max(16384, args.history + 128),
        max_num_seqs=40,
        max_num_batched_tokens=256,
        block_size=32,
        enable_prefix_caching=False,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [tokens],
            "cache_dir": str(args.output / "compile_cache"),
        },
        additional_config=additional,
    ).create_engine_config()
    with native_session(config, args.device), torch.inference_mode():
        device = torch.device(f"npu:{args.device}")
        layer, weight_report = make_layer(config, Path(CHECKPOINT), device, LAYER_INDEX)
        fixture = make_fixture(config, layer.self_attn, args.batch, args.history, SEED, device)
        groups = {
            name: (fixture["metadata"][group["prefix"]], tuple(group["views"]))
            for name, group in fixture["groups"].items()
        }
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        report.update(
            cann=os.environ.get("ASCEND_HOME_PATH"),
            weights=weight_report,
            options={
                "dynamic": False,
                "inplace_pass": True,
                "static_kernel_compile": True,
                "super_kernel_optimize": False,
            },
        )
        calls, outputs, references = {}, {}, {}

        class RootModule(torch.nn.Module):
            def __init__(self, operator):
                super().__init__()
                self.operator = operator

            def forward(self, *inputs):
                return self.operator(*inputs)

        torch.npu.set_compile_mode(jit_compile=False)
        for label, (root, adapter) in modules.items():
            output = torch.empty_like(fixture["hidden"])
            call = adapter.NativeHCACall(
                adapter.HCAOperators(operators[label]),
                adapter.prepare_weights(layer.self_attn, layer),
                fixture["hidden"],
                fixture["positions"],
                groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=fixture["compact"]["compressed"],
                output=output,
            )
            report.setdefault("runtime_shapes", {})[label] = {
                "tokens": tokens,
                "cmp_table": list(call.args["cmp_table"].shape),
                "attention_path": "long" if call.args["cmp_table"].shape[1] > 4 else "short",
            }
            restore(fixture)
            call()
            torch.npu.synchronize()
            references[label] = collect_state(fixture, output)
            compiled = torch.compile(
                RootModule(operators[label]),
                backend="npugraph_ex",
                fullgraph=True,
                dynamic=False,
                options={
                    "force_eager": False,
                    "inplace_pass": True,
                    "clone_input": False,
                    "clone_output": False,
                    "static_kernel_compile": True,
                    "super_kernel_optimize": False,
                },
            )
            calls[label] = (compiled, call.core_args)
            outputs[label] = output
            for _ in range(5):
                restore(fixture)
                compiled(*call.core_args)
            torch.npu.synchronize()
            actual = collect_state(fixture, output)
            assert all(torch.equal(actual[k], references[label][k]) for k in actual), label

        exact = {k: bool(torch.equal(v, references["candidate"][k])) for k, v in references["base"].items()}
        report["cross_variant_exact"] = exact
        if not all(exact.values()):
            from dsv4_csa_validation import compare_tensor

            report["status"] = "CORRECTNESS_FAILED"
            report["differences"] = {
                k: compare_tensor(references["candidate"][k], references["base"][k], 0, 0)
                for k, equal in exact.items()
                if not equal
            }
            (args.output / "failure.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            torch.save(
                {label: {k: v for k, v in values.items() if not exact[k]} for label, values in references.items()},
                args.output / "differences.pt",
            )
            raise RuntimeError("数值中性候选逐bit检查失败，详见failure.json；未开始性能计时")
        order = ("base", "candidate", "candidate", "base") * args.cycles
        # 与既有profiler路径一致：计时窗之前恢复初态，窗内连续重放同一输入。
        restore(fixture)
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / "profile")),
        ) as trace:
            for index, label in enumerate(order):
                compiled, core_args = calls[label]
                with torch.profiler.record_function(f"HCA_PAIR_{index}_{label}"):
                    compiled(*core_args)
                    torch.npu.synchronize()
            trace.step()
        profile = device_interval(args.output / "profile", replays=len(order))
        assert profile.get("available") and len(profile["span_us_all"]) == len(order)
        assert len(set(profile["kernels_per_replay"])) == 1, profile["kernels_per_replay"]
        report["profile"] = profile
        report["order"] = order
        report["summary"] = {
            label: stats([value for side, value in zip(order, profile["span_us_all"]) if side == label])
            for label in sources
        }
        report["pairs"] = [
            {
                "base_us": statistics.mean(profile["span_us_all"][i : i + 4 : 3]),
                "candidate_us": statistics.mean(profile["span_us_all"][i + 1 : i + 3]),
            }
            for i in range(0, len(order), 4)
        ]
        report["state_exact"] = {}
        for label, output in outputs.items():
            state = collect_state(fixture, output)
            report["state_exact"][label] = {
                key: bool(torch.equal(value, references[label][key])) for key, value in state.items()
            }
        report["guards"] = {k: v["status"] for k, v in guard_checks(fixture).items()}
        assert all(all(values.values()) for values in report["state_exact"].values())
        assert all(value == "PASS" for value in report["guards"].values())
        if args.padding_graph:
            from dsv4_hca_padding import check_padding_graph

            padding_side = args.padding_variant
            _, adapter = modules[padding_side]
            weights = adapter.prepare_weights(layer.self_attn, layer)
            output = outputs[padding_side]

            def make_padding_call(compact):
                return adapter.NativeHCACall(
                    adapter.HCAOperators(operators[padding_side]),
                    weights,
                    fixture["hidden"],
                    fixture["positions"],
                    groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=compact,
                    output=output,
                )

            restore(fixture)
            make_padding_call(fixture["compact"]["compressed"])()
            torch.npu.synchronize()
            eager = {"output": output.cpu()}
            eager.update({name: group["allocation"].cpu() for name, group in fixture["groups"].items()})
            report["padding_variant"] = padding_side
            try:
                check_padding_graph(
                    fixture, output, eager, make_padding_call, layer.self_attn.dsa_attn.dsa_attn.impl, report
                )
            except Exception:
                report["status"] = "PADDING_FAILED"
                (args.output / "failure.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                raise
        report["status"] = "MEASURED"
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
