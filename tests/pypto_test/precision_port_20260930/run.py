# SPDX-License-Identifier: Apache-2.0
"""固定源码、单卡TMR：性能版与迁移前后精度版的CSA对照。"""

import argparse
import importlib
import json
import os
import statistics
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--atomic", type=int, choices=(0, 1), required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--variant", choices=("precision", "performance"), required=True)
    parser.add_argument("--reverse-pages", action="store_true")
    parser.add_argument("--device")  # task-submit supplies physical index; visible logical index is zero
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

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.reduction import ATOMIC_ADD
    module = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark" + ("_perf" if args.variant == "performance" else "")
    qkv = importlib.import_module(module + ".qkv_proj_rope")
    decode = importlib.import_module(module + ".decode_csa")
    csa_adapter = importlib.import_module(module + ".native_adapter")
    assert args.atomic == ATOMIC_ADD
    roots = (decode.decode_csa_tp1_layer_test,)
    for root in roots:
        root._get_dep_graph()
    if args.compile_only:
        from pypto.runtime import RunConfig

        for name, root in zip(("CSA",), roots):
            artifact = root._resolve_kernel_artifact(
                (), {"config": RunConfig(platform="a2a3")}, allow_signature_mode=True,
            )
            artifact.load()
            print("COMPILE_PASS", name, args.atomic, artifact.output_dir, flush=True)
        return

    from dsv4_csa_native_case import native_session
    from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer
    from dsv4_csa_validation import compare_tensor
    from pypto._kernel_abi import SIMPLER_KERNEL_REVISION
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs
    from vllm_ascend.utils import enable_custom_op

    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    report = {
        "status": "RUNNING", "batch": args.batch, "history": args.history,
        "variant": args.variant, "reverse_pages": args.reverse_pages, "atomic_add": ATOMIC_ADD,
        "qr_split_k": qkv.QR_OK, "kv_split_k": qkv.KV_OK,
        "kv_k_tile": qkv.KV_K_TILE, "runtime": "tensormap_and_ringbuffer",
        "source": str(args.source), "checkpoint": str(checkpoint),
        "task_device": os.environ["TASK_DEVICE"], "weight_nz_mode": 2,
        "deterministic_level": 0, "warmup": 5, "iterations": 20,
        "scope": "正式第2层CSA的HC_pre+norm+attention+HC_post；合成固定历史；无MoE",
        "metadata": "初始化时准备并复用 compact metadata；计时仅包含根算子图重放",
        "reset": "每次图重放前D2D恢复相同初态并同步，恢复不计时",
        "versions": {"simpler_revision": SIMPLER_KERNEL_REVISION, "torch": str(torch.__version__),
                     "torch_npu": str(torch_npu.__version__), "cann_home": os.environ.get("ASCEND_HOME_PATH")},
    }
    try:
        current_platform.pre_register_and_update()
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        if not enable_custom_op():
            raise RuntimeError("Native custom ops unavailable")
        config = EngineArgs(
            model=str(checkpoint), tokenizer_mode="deepseek_v4", trust_remote_code=True,
            tensor_parallel_size=1, dtype="bfloat16", quantization="ascend",
            hf_overrides={"sliding_window": 128}, max_model_len=max(16384, args.history + 128),
            max_num_seqs=40, max_num_batched_tokens=256, enable_prefix_caching=False,
            enforce_eager=True, block_size=32,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        with native_session(config, 0), torch.inference_mode():
            device = torch.device("npu:0")
            csa_layer, _ = make_layer(config, checkpoint, device, 2)
            csa = make_fixture(
                config, csa_layer.self_attn, args.batch, args.history, 1024, device,
                reverse_pages=args.reverse_pages,
            )
            csa_out = torch.empty_like(csa["hidden"])
            pypto.torch.init(device=0, platform="a2a3", runtime="tensormap_and_ringbuffer",
                             **ring_sizing_kwargs())

            def groups(fixture):
                return {name: (fixture["metadata"][group["prefix"]], tuple(group["views"]))
                        for name, group in fixture["groups"].items()}

            hadamard = csa["metadata"][csa["groups"]["indexer"]["prefix"]].hadamard
            csa_call = csa_adapter.NativeCSACall(
                csa_adapter.CSAOperators.register(),
                csa_adapter.prepare_weights(csa_layer.self_attn, hadamard, csa_layer),
                csa["hidden"], csa["positions"], groups(csa),
                layer_name=csa_layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=csa["compact"], buffers={
                    "x_out": csa_out,
                    "idx_topk_scores": torch.empty((csa["tokens"], 512), dtype=torch.float32, device=device),
                    "idx_topk": torch.empty((csa["tokens"], 512), dtype=torch.int32, device=device),
                },
            )
            initial = [(group["allocation"], group["allocation"].clone())
                       for fixture in (csa,) for group in fixture["groups"].values()]

            def reset():
                for allocation, value in initial:
                    allocation.copy_(value)

            functions = {"CSA": (csa_call, csa_out)}
            saved = {}
            report["timing_us"], report["replay_vs_eager"] = {}, {}
            for name, (call, output) in functions.items():
                print("BEGIN", name, args.batch, args.history, "atomic", args.atomic, flush=True)
                reset()
                call()
                torch.npu.synchronize()
                eager = output.cpu()
                reset()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    call()
                for _ in range(5):
                    reset()
                    graph.replay()
                torch.npu.synchronize()
                samples = []
                for _ in range(20):
                    reset()
                    torch.npu.synchronize()
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
                saved[name] = output.cpu()
                report["replay_vs_eager"][name] = compare_tensor(saved[name], eager, 0, 0)
                report["timing_us"][name] = {
                    "min": min(samples), "mean": statistics.mean(samples), "max": max(samples),
                    "median": statistics.median(samples),
                    "p95": float(torch.tensor(samples, dtype=torch.float64).quantile(0.95)),
                    "samples": samples,
                }
                print("TIMING", name, json.dumps(report["timing_us"][name]), flush=True)
                del graph
            saved["idx_topk"] = csa_call.args["idx_topk"].cpu()
            report["nonfinite"] = {name: int((~torch.isfinite(value)).sum())
                                   for name, value in saved.items() if value.is_floating_point()}
            report["guards"] = {"CSA": guard_checks(csa), }
            saved["idx_topk_scores"] = csa_call.args["idx_topk_scores"].cpu()
            for name, group in csa["groups"].items():
                # 保存允许写入区的原始字节，结合guard检查覆盖全部cache/state，无需全量快照。
                saved["state." + name] = group["allocation"].cpu()[group["allowed"]]
            torch.save(saved, args.output / "outputs.pt")
            if any(report["nonfinite"].values()):
                raise AssertionError("Nonfinite output")
            if any(check["status"] != "PASS" for checks in report["guards"].values() for check in checks.values()):
                raise AssertionError("Guard or metadata mutation")
            if not ATOMIC_ADD and any(v["status"] != "PASS" for v in report["replay_vs_eager"].values()):
                raise AssertionError("Fixed-K graph/eager differs")
            report["status"] = "MEASURED"
            report["accuracy_scope"] = "输出、TopK/score及cache/state写入区；保护区检查；不代表Native/token/DSpark验收"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
