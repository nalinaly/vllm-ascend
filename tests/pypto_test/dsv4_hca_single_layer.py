# SPDX-License-Identifier: Apache-2.0
"""正式 HCA 第 3 层的 Native/PTO 整层对照，复用 CSA 的 Native fixture。"""

import argparse
import os
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_validation import compare_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--history", type=int, default=124)
    parser.add_argument("--device", type=int, choices=[0], default=0, help="可见设备中的逻辑卡号")
    parser.add_argument("--service-graph", action="store_true", help="额外验证服务 custom op 与一次图重放")
    parser.add_argument("--poison-unused-compressed", action="store_true",
                        help="将本步不可见的压缩 KV 行填为 NaN，验证 attention 不消费未初始化数据")
    parser.add_argument("--timing-iters", type=int, default=0, help="每侧图重放计时次数；0 不计时")
    parser.add_argument("--timing-warmup", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="计时结束后分别采集 Native/PTO 图重放 profiler")
    parser.add_argument("--swimlane", action="store_true", help="独立采集两次 PTO 图重放泳道，不与正式计时混用")
    parser.add_argument("--operator-source", type=Path, help="只供对照测试：指定已冻结的 ops/pypto 源码目录")
    parser.add_argument("--reference-state", type=Path, help="纯调度或搬运优化：要求 PTO 输出与旧快照逐 bit 相同")
    parser.add_argument("--weight-nz-mode", type=int, choices=[1, 2], default=1)
    parser.add_argument("--deterministic-level", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"))
    args = parser.parse_args()
    if not 1 <= args.batch <= 40 or args.history < 0:
        parser.error("batch 必须为 1～40，history 不得为负")
    if args.timing_iters < 0 or args.timing_warmup < 1:
        parser.error("timing-iters 不得为负，timing-warmup 至少为 1")
    if args.profile and not args.timing_iters:
        parser.error("profile 需要 timing-iters")
    if args.swimlane and args.timing_iters:
        parser.error("泳道采集必须与无 profiler 计时分开进程")
    if not os.environ.get("TASK_DEVICE"):
        raise RuntimeError("NPU 验证必须通过 task-submit 提交")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = str(args.weight_nz_mode)
    os.environ["HCCL_DETERMINISTIC"] = "true" if args.deterministic_level else "false"
    activate()
    if args.operator_source:
        import vllm_ascend.ops.pypto as operator_package

        source = args.operator_source.resolve()
        if not (source / "deepseek_v4_flash_hca/decode_hca.py").is_file():
            raise ValueError(f"无效的 HCA 源码目录：{source}")
        operator_package.__path__ = [str(source)]
    report = {
        "scope": "正式第 3 层权重，合成输入和历史；不代表整模型验收", "status": "RUNNING",
        "checkpoint": str(args.checkpoint), "seed": 20260928, "task_device": os.environ["TASK_DEVICE"],
        "weight_nz_mode": args.weight_nz_mode, "deterministic_level": args.deterministic_level,
        "speculative_tokens": 5, "query_tokens_per_request": 6,
        "operator_source": str(args.operator_source.resolve()) if args.operator_source else "当前 worktree",
    }
    try:
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer, restore
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        from vllm_ascend.utils import enable_custom_op

        current_platform.pre_register_and_update()
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        if not enable_custom_op():
            raise RuntimeError("Native 自定义算子注册失败")
        torch_npu.npu.set_deterministic_level(args.deterministic_level)
        config = EngineArgs(
            model=str(args.checkpoint), tokenizer_mode="deepseek_v4", trust_remote_code=True,
            tensor_parallel_size=1, dtype="bfloat16", quantization="ascend", hf_overrides={"sliding_window": 128},
            max_model_len=max(16384, args.history + 128), max_num_seqs=40, max_num_batched_tokens=256,
            enable_prefix_caching=False, enforce_eager=True, block_size=32,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": args.weight_nz_mode, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        with native_session(config, 0), torch.inference_mode():
            device = torch.device("npu:0")
            layer, details = make_layer(config, args.checkpoint, device, layer_index=3)
            report.update(details, batch=args.batch, history=args.history)
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 20260928, device)
            if args.poison_unused_compressed:
                group = fixture["groups"]["compressed"]
                cache = group["views"][0]
                table = fixture["metadata"][group["prefix"]].decode.block_table.cpu()
                valid_rows = (args.history + 6) // 128
                poisoned_rows = 0
                for row in table:
                    for column, page in enumerate(row.tolist()):
                        first_unused = max(0, valid_rows - column * 32)
                        if 0 <= page < cache.shape[0] and first_unused < 32:
                            cache[page, first_unused:].fill_(float("nan"))
                            poisoned_rows += 32 - first_unused
                group["initial"] = group["allocation"].cpu()
                report["poisoned_compressed_rows"] = poisoned_rows
                if not poisoned_rows:
                    raise RuntimeError("测试未覆盖任何不可见压缩行")
            report["layouts"] = {name: group["layout"] for name, group in fixture["groups"].items()}
            output = torch.empty_like(fixture["hidden"])

            def collect():
                return {"output": output.cpu(), **{
                    name: group["allocation"].cpu() for name, group in fixture["groups"].items()
                }}

            def native_call():
                with set_ascend_forward_context(fixture["metadata"], config, num_tokens=fixture["tokens"],
                                               num_actual_tokens=fixture["tokens"]):
                    _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)

            restore(fixture)
            native_call()
            torch.npu.synchronize()
            native = collect()
            report["native_output_nonfinite"] = int((~torch.isfinite(native["output"])).sum())
            report["native_guards"] = guard_checks(fixture)
            from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.native_adapter import HCAOperators, NativeHCACall, prepare_weights
            from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.reduction import ATOMIC_ADD

            pypto.torch.init(
                device=0, platform="a2a3", runtime="tensormap_and_ringbuffer",
                **({"enable_chip_swimlane": 4, "enable_dep_gen": True,
                    "output_dir": str((args.output / "dfx").resolve())} if args.swimlane else {}),
            )
            report["atomic_add"] = ATOMIC_ADD
            operators = HCAOperators.register()
            groups = {name: (fixture["metadata"][group["prefix"]], tuple(group["views"]))
                      for name, group in fixture["groups"].items()}
            call = NativeHCACall(
                operators, prepare_weights(layer.self_attn, layer), fixture["hidden"], fixture["positions"], groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=fixture["compact"]["compressed"], output=output,
            )
            restore(fixture)
            call()
            torch.npu.synchronize()
            pto = collect()
            report["pto_output_nonfinite"] = int((~torch.isfinite(pto["output"])).sum())
            report["pto_guards"] = guard_checks(fixture)
            report["pto_native"] = {name: compare_tensor(value, native[name], 0, 0) for name, value in pto.items()}
            if args.reference_state:
                import json

                reference_report = json.loads(args.reference_state.with_name("report.json").read_text())
                for key in ("batch", "history", "seed", "checkpoint", "weight_nz_mode", "atomic_add"):
                    if report[key] != reference_report[key]:
                        raise ValueError(f"旧快照配置不一致：{key}")
                reference = torch.load(args.reference_state, map_location="cpu", weights_only=False)["pto"]
                report["reference_state"] = str(args.reference_state)
                report["pto_reference"] = {name: compare_tensor(value, reference[name], 0, 0)
                                           for name, value in pto.items()}
                if any(value["status"] != "PASS" for value in report["pto_reference"].values()):
                    # 失败后仍保留已同步到 CPU 的结果，供定位使用；不进入计时。
                    torch.save({"native": native, "pto": pto}, args.output / "failed_states.pt")
                    report["failed_snapshot"] = "failed_states.pt（仅诊断，本地保留）"
                    raise RuntimeError("PTO 搬运或调度改动改变了输出或完整 allocation")
            if args.service_graph or args.timing_iters:
                import vllm_ascend.ops.dsv4_hca  # noqa: F401
                from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service import HCAServiceRuntime

                wrapper = layer.self_attn.dsa_attn
                wrapper._pto_hca_layer = (layer,)
                runtime = HCAServiceRuntime(layer.self_attn, operators, 40, layer)
                wrapper._pto_hca_runtime = runtime

                def service_call():
                    with set_ascend_forward_context(fixture["metadata"], config, num_tokens=fixture["tokens"],
                                                   num_actual_tokens=fixture["tokens"]):
                        from vllm.forward_context import get_forward_context

                        if not runtime.eligible(get_forward_context(), fixture["hidden"], fixture["positions"]):
                            raise RuntimeError("服务入口未选中 HCA，禁止用 Native fallback 冒充通过")
                        torch.ops.vllm.dsv4_hca_forward(fixture["hidden"], fixture["positions"], output, wrapper.prefix)

                restore(fixture)
                service_call()
                torch.npu.synchronize()
                service = collect()
                report["service_direct"] = {name: compare_tensor(value, pto[name], 0, 0)
                                            for name, value in service.items()}
                if any(value["status"] != "PASS" for value in report["service_direct"].values()):
                    raise RuntimeError("服务调用与直接调用不一致")
                if args.service_graph:
                    restore(fixture)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        service_call()
                    restore(fixture)
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.npu.synchronize()
                    replay = collect()
                    report["graph_eager"] = {name: compare_tensor(value, service[name], 0, 0)
                                             for name, value in replay.items()}
                    report["graph_guards"] = guard_checks(fixture)
                    if any(value["status"] != "PASS" for key in ("graph_eager", "graph_guards")
                           for value in report[key].values()):
                        raise RuntimeError("图重放与服务 eager 不一致")
                if args.timing_iters:
                    from dsv4_hca_performance import measure_graph_pair

                    report["timing"] = measure_graph_pair(
                        fixture, output, {"native": native_call, "pto": service_call},
                        {"native": native, "pto": service}, collect,
                        iters=args.timing_iters, warmup=args.timing_warmup,
                        profile_dir=args.output / "profile" if args.profile else None,
                    )
            if args.swimlane:
                from dsv4_hca_performance import capture_swimlane

                report["swimlane_windows"] = capture_swimlane(fixture, call, args.output / "dfx")
            # 把四份结果留在本地，后续定位沿用这一轮数据，不反复重跑 Native。
            torch.save({"native": native, "pto": pto}, args.output / "states.pt")
            report["local_snapshot"] = "states.pt（本地保留，不入 Git）"
            report["status"] = "MEASURED"
            if any(value["status"] != "PASS" for key in ("native_guards", "pto_guards") for value in report[key].values()):
                raise RuntimeError("写保护或只读 metadata 检查失败")
            if any(value.get("nonfinite") != 0 for value in report["pto_native"].values()):
                raise RuntimeError("完整输出或缓存出现非有限值")
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
