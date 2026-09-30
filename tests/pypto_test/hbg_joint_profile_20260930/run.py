# SPDX-License-Identifier: Apache-2.0
"""单卡真实服务入口 CSA→HCA 联合图：三步 PyTorch/DFX 同轮采集。"""

import argparse
import os
import shutil
import statistics
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=int, choices=(0,), default=0)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--swimlane-level", type=int, choices=(0, 1, 4), default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dsv4_csa_env import activate, write_json

    activate()
    import vllm_ascend.ops.pypto as package

    package.__path__ = [str(args.source.resolve())]
    import pypto.torch
    import torch
    import torch_npu
    from dsv4_csa_native_case import native_session
    from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
    from dsv4_csa_validation import compare_tensor
    from pypto._kernel_abi import SIMPLER_KERNEL_REVISION
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import get_forward_context
    from vllm.platforms import current_platform

    import vllm_ascend.ops.dsv4_csa  # noqa: F401
    import vllm_ascend.ops.dsv4_hca  # noqa: F401
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.native_adapter import HBGCSAOperators
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.service import CSAServiceRuntime
    from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.native_adapter import HCAOperators
    from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service import HCAServiceRuntime
    from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs
    from vllm_ascend.utils import enable_custom_op

    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    report = {
        "status": "RUNNING",
        "batch": args.batch,
        "history": args.history,
        "profile_steps": 3,
        "runtime": "host_build_graph",
        "source": str(args.source),
        "checkpoint": str(checkpoint),
        "scope": "正式第2/3层 attention 半层串接，不含 MoE；固定 metadata 合成历史重放",
        "task_device": os.environ["TASK_DEVICE"],
        "weight_nz_mode": 2,
        "atomic_add": 0,
        "deterministic_level": 1,
        "graph": "manual NPUGraph, same graph for all three steps",
    }
    report["versions"] = {
        "simpler_revision": SIMPLER_KERNEL_REVISION,
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "cann_home": os.environ.get("ASCEND_HOME_PATH"),
    }
    try:
        current_platform.pre_register_and_update()
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(1)
        if not enable_custom_op():
            raise RuntimeError("Native custom ops unavailable")
        config = EngineArgs(
            model=str(checkpoint),
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="bfloat16",
            quantization="ascend",
            hf_overrides={"sliding_window": 128},
            max_model_len=max(16384, args.history + 128),
            max_num_seqs=40,
            max_num_batched_tokens=256,
            enable_prefix_caching=False,
            enforce_eager=True,
            block_size=32,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        with native_session(config, 0), torch.inference_mode():
            device = torch.device("npu:0")
            csa_layer, _ = make_layer(config, checkpoint, device, 2)
            hca_layer, _ = make_layer(config, checkpoint, device, 3)
            csa = make_fixture(config, csa_layer.self_attn, args.batch, args.history, 1024, device)
            hca = make_fixture(config, hca_layer.self_attn, args.batch, args.history, 20260928, device)
            intermediate, output = torch.empty_like(csa["hidden"]), torch.empty_like(csa["hidden"])
            hca["hidden"] = intermediate
            metadata = {**csa["metadata"], **hca["metadata"]}
            pypto.torch.init(
                device=0,
                platform="a2a3",
                runtime="host_build_graph",
                **ring_sizing_kwargs(),
                enable_chip_swimlane=args.swimlane_level,
                enable_dep_gen=bool(args.swimlane_level),
                output_dir=str((args.output / "dfx").resolve()),
            )
            csa_runtime = CSAServiceRuntime(csa_layer.self_attn, HBGCSAOperators.register(), 40, csa_layer)
            hca_runtime = HCAServiceRuntime(hca_layer.self_attn, HCAOperators.register(), 40, hca_layer)
            csa_wrapper, hca_wrapper = csa_layer.self_attn.dsa_attn, hca_layer.self_attn.dsa_attn
            csa_wrapper._pto_csa_runtime = csa_runtime
            hca_wrapper._pto_hca_runtime = hca_runtime
            hca_wrapper._pto_hca_layer = (hca_layer,)
            assert csa_runtime.is_hbg and hca_runtime.operators.is_hbg

            def reset():
                restore(csa)
                restore(hca)

            def call_one(name):
                fixture, runtime, wrapper, target = (
                    (csa, csa_runtime, csa_wrapper, intermediate)
                    if name == "CSA"
                    else (hca, hca_runtime, hca_wrapper, output)
                )
                with set_ascend_forward_context(
                    metadata, config, num_tokens=csa["tokens"], num_actual_tokens=csa["tokens"]
                ):
                    context = get_forward_context()
                    if not runtime.eligible(context, fixture["hidden"], fixture["positions"]):
                        raise RuntimeError(f"{name} would fall back to Native")
                    op = torch.ops.vllm.dsv4_csa_forward if name == "CSA" else torch.ops.vllm.dsv4_hca_forward
                    op(fixture["hidden"], fixture["positions"], target, wrapper.prefix)

            def collect():
                values = {
                    f"csa.{k}": v
                    for k, v in collect_state(csa, intermediate, csa_runtime.topk[: csa["tokens"]]).items()
                }
                values.update(
                    {
                        "hca.output": output.cpu(),
                        **{f"hca.{name}": group["allocation"].cpu() for name, group in hca["groups"].items()},
                    }
                )
                return values

            reset()
            call_one("CSA")
            torch.npu.synchronize()
            reset()
            used_dep_files = set()
            for window, name in enumerate(("CSA", "HCA")):
                if not args.swimlane_level:
                    call_one(name)
                    continue
                pypto.torch.begin_dfx()
                try:
                    call_one(name)
                finally:
                    pypto.torch.end_dfx()
                candidates = set((args.output / "dfx").glob("deps_callable_*.json")) - used_dep_files
                if len(candidates) != 1:
                    raise RuntimeError(f"Expected one new {name} dependency graph: {candidates}")
                source = candidates.pop()
                used_dep_files.add(source)
                shutil.copyfile(source, args.output / f"{name.lower()}_deps.json")
            eager = collect()
            reset()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                call_one("CSA")
                call_one("HCA")
            for _ in range(3):
                reset()
                graph.replay()
            torch.npu.synchronize()
            samples = []
            for _ in range(10):
                reset()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            report["unprofiled_joint_us"] = {
                "samples": samples,
                "min": min(samples),
                "mean": statistics.mean(samples),
                "max": max(samples),
            }
            if args.swimlane_level:
                pypto.torch.begin_dfx()
            try:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    with_modules=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / "pytorch")),
                ) as trace:
                    for step in range(3):
                        with torch.autograd.profiler.record_function("ResetSyntheticState_OutsideForward"):
                            reset()
                            torch.npu.synchronize()
                        with torch.autograd.profiler.record_function(f"Step_{step + 1:02d}_CSA_to_HCA_HBG"):
                            graph.replay()
                            torch.npu.synchronize()
                        trace.step()
            finally:
                if args.swimlane_level:
                    pypto.torch.end_dfx()
            report["joint_dfx_directory"] = str(args.output / "dfx/window_2") if args.swimlane_level else None
            report["swimlane_level"] = args.swimlane_level
            actual = collect()
            report["graph_eager"] = {k: compare_tensor(v, eager[k], 0, 0) for k, v in actual.items()}
            report["guards"] = {"csa": guard_checks(csa), "hca": guard_checks(hca)}
            checks = [
                *report["graph_eager"].values(),
                *report["guards"]["csa"].values(),
                *report["guards"]["hca"].values(),
            ]
            if args.reference:
                expected = torch.load(args.reference, map_location="cpu", weights_only=False)
                report["reference"] = {k: compare_tensor(v, expected[k], 0, 0) for k, v in actual.items()}
                checks.extend(report["reference"].values())
            torch.save(actual, args.output / "states.pt")
            if any(value["status"] != "PASS" for value in checks):
                raise AssertionError("联合图输出/状态/保护区检查失败")
            report["status"] = "PASS"
            print("JOINT_PROFILE_PASS", report["unprofiled_joint_us"], flush=True)
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
