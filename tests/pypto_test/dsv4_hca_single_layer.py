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
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"],
                        default="tensormap_and_ringbuffer")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--history", type=int, default=124)
    parser.add_argument("--device", type=int, choices=[0], default=0, help="可见设备中的逻辑卡号")
    parser.add_argument("--service-graph", action="store_true", help="额外验证服务 custom op 与一次图重放")
    parser.add_argument("--padding-graph", action="store_true",
                        help="同一张图跑 满档→满档−1→1→满档，验证动态 BS 切换与 padding/dummy")
    parser.add_argument("--trajectory-steps", type=int, default=0,
                        help="连续 decode 轨迹步数；两侧同输入各跑这么多步，比较状态偏差走势")
    parser.add_argument("--lifecycle", action="store_true",
                        help="请求生命周期（页复用，NaN 污染作 oracle）与 prefix 共享（共享页输出须逐 bit 相同）")
    parser.add_argument("--poison-unused-compressed", action="store_true",
                        help="将本步不可见的压缩 KV 行填为 NaN，验证 attention 不消费未初始化数据")
    parser.add_argument("--timing-iters", type=int, default=0, help="每侧图重放计时次数；0 不计时")
    parser.add_argument("--timing-warmup", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="计时结束后分别采集 Native/PTO 图重放 profiler")
    parser.add_argument("--timing-direct", action="store_true",
                        help="额外给直接算子调用计时（compact metadata 预先算好），"
                             "用于量出服务入口里整步 metadata 生成的设备耗时")
    parser.add_argument("--swimlane", action="store_true", help="独立采集两次 PTO 图重放泳道，不与正式计时混用")
    parser.add_argument("--swimlane-cold-l2", action="store_true",
                        help="泳道每个窗口前冲刷 L2，使权重读取与正式交替计时一样为冷数据")
    parser.add_argument("--swimlane-after-native", action="store_true",
                        help="泳道每个窗口前先跑一次 Native，复现正式交替计时中 PTO 所见的 L2 状态")
    parser.add_argument("--operator-source", type=Path, help="只供对照测试：指定已冻结的 ops/pypto 源码目录")
    parser.add_argument("--reference-state", type=Path, help="纯调度或搬运优化：要求 PTO 输出与旧快照逐 bit 相同")
    parser.add_argument("--weight-nz-mode", type=int, choices=[1, 2], default=1)
    parser.add_argument("--deterministic-level", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"))
    args = parser.parse_args()
    weight_options = {"host_scalars": True} if args.runtime == "host_build_graph" else {}
    if not 1 <= args.batch <= 40 or args.history < 0:
        parser.error("batch 必须为 1～40，history 不得为负")
    if args.timing_iters < 0 or args.timing_warmup < 1:
        parser.error("timing-iters 不得为负，timing-warmup 至少为 1")
    if args.profile and not args.timing_iters:
        parser.error("profile 需要 timing-iters")
    if args.swimlane and args.timing_iters:
        parser.error("泳道采集必须与无 profiler 计时分开进程")
    if args.padding_graph and (args.timing_iters or args.swimlane):
        parser.error("补位图检查必须与计时/泳道分开进程")
    if args.padding_graph and args.batch < 2:
        parser.error("补位图检查需要 batch >= 2，否则没有可补位的请求")
    if args.trajectory_steps and (args.timing_iters or args.swimlane or args.padding_graph):
        parser.error("轨迹检查必须单独一个进程")
    if args.trajectory_steps < 0:
        parser.error("trajectory-steps 不得为负")
    if args.lifecycle and (args.timing_iters or args.swimlane or args.padding_graph or args.trajectory_steps):
        parser.error("生命周期检查必须单独一个进程")
    if args.lifecycle and args.batch < 2:
        parser.error("生命周期与 prefix 共享检查需要 batch >= 2")
    if not os.environ.get("TASK_DEVICE"):
        raise RuntimeError("NPU 验证必须通过 task-submit 提交")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = str(args.weight_nz_mode)
    os.environ["HCCL_DETERMINISTIC"] = "true" if args.deterministic_level else "false"
    os.environ["PTO_CSA_RUNTIME"] = args.runtime
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
        "runtime": args.runtime, "weight_nz_mode": args.weight_nz_mode, "deterministic_level": args.deterministic_level,
        "speculative_tokens": 5, "query_tokens_per_request": 6,
        "operator_source": str(args.operator_source.resolve()) if args.operator_source else "当前 worktree",
        "simpler_root": os.environ.get("HCA_SIMPLER_ROOT_ACTIVE", "共用 pto-eager/simpler"),
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
            max_model_len=max(16384, args.history + 128), max_num_seqs=max(40, args.batch),
            max_num_batched_tokens=400,
            enable_prefix_caching=True, block_size=32,
            # 不能用 enforce_eager：它让 cudagraph_mode 变成 NONE，而 platform.py 在该分支下
            # 会显式把 enable_npugraph_ex 与 enable_static_kernel 置 False（并改写
            # additional_config 里的值），导致上线口径的这两个开关被静默关掉。
            # 按上线模板给 FULL_DECODE_ONLY；本 bench 仍自己捕获 NPUGraph 做单层计时。
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            # 与上线 decode 口径对齐：
            # vllm-ascend-main/tests/dsv4_perf_accuracy_20260827/runtime/decode/run_dp_template.sh
            # 的 --additional-config。Native 与 PTO 共用同一个 config 对象，两侧必然同配。
            # multistream_overlap_shared_expert 在单层里没有 MoE 可重叠，仍按模板给上，
            # 避免两套配置在别处产生隐性差异。
            additional_config={
                "weight_nz_mode": args.weight_nz_mode, "enable_kv_nz": False, "enable_dsa_cp": False,
                "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": True},
                "enable_cpu_binding": True,
                "multistream_overlap_shared_expert": True,
                "recompute_scheduler_enable": False,
            },
        ).create_engine_config()
        # 把平台最终生效的编译开关回显到报告里：platform.py 会在 cudagraph_mode 为 NONE 时
        # 把它们强制置 False，只看自己传进去的值会误判。
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            compile_config = get_ascend_config().ascend_compilation_config
            report["effective_compilation"] = {
                "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
                "enable_npugraph_ex": bool(compile_config.enable_npugraph_ex),
                "enable_static_kernel": bool(compile_config.enable_static_kernel),
                "fuse_norm_quant": bool(getattr(compile_config, "fuse_norm_quant", False)),
            }
        except Exception as exc:  # noqa: BLE001 - 回显失败不该影响计时
            report["effective_compilation"] = {"error": repr(exc)}
        with native_session(config, 0), torch.inference_mode():
            device = torch.device("npu:0")
            layer, details = make_layer(config, args.checkpoint, device, layer_index=3)
            report.update(details, batch=args.batch, history=args.history)
            # 轨迹测试会把 positions 推进 6*steps，页表宽度必须按轨迹终点定，
            # 否则 pos//block_size 会越过 make_fixture 按 history 算出的列数，
            # 读到错误物理页而表现为"发散"（实测 48 步时输出 max_abs 涨到 1.41）。
            fixture = make_fixture(
                config, layer.self_attn, args.batch, args.history, 20260928, device,
                table_history=(args.history + 6 * args.trajectory_steps) if args.trajectory_steps else None,
            )
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
            from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.reduction import ATOMIC_ADD
            from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.native_adapter import (
                HCAOperators,
                NativeHCACall,
                prepare_weights,
            )
            from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

            pypto.torch.init(
                device=0, platform="a2a3", runtime=args.runtime, **ring_sizing_kwargs(),
                **({"enable_chip_swimlane": 4, "enable_dep_gen": True,
                    "output_dir": str((args.output / "dfx").resolve())} if args.swimlane else {}),
            )
            report["atomic_add"] = ATOMIC_ADD
            operators = HCAOperators.register()
            groups = {name: (fixture["metadata"][group["prefix"]], tuple(group["views"]))
                      for name, group in fixture["groups"].items()}
            call = NativeHCACall(
                operators, prepare_weights(layer.self_attn, layer, **weight_options), fixture["hidden"], fixture["positions"], groups,
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
            if args.padding_graph:
                from dsv4_hca_padding import check_padding_graph

                # 权重准备必须在图捕获之前做完：prepare_weights 内部的 scale() 会做
                # bool(count_nonzero(offset).cpu())，那是同步 D2H 拷贝，捕获期间会被
                # rtStreamSynchronize 拒绝（error 107030 / EE1016）。
                padding_weights = prepare_weights(layer.self_attn, layer, **weight_options)

                def make_call(compact):
                    return NativeHCACall(
                        operators, padding_weights, fixture["hidden"],
                        fixture["positions"], groups,
                        layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                        compact_metadata=compact, output=output,
                    )

                check_padding_graph(
                    fixture, output, pto, make_call,
                    layer.self_attn.dsa_attn.dsa_attn.impl, report,
                )
            if args.lifecycle:
                from dsv4_hca_lifecycle import check_lifecycle

                lifecycle_weights = prepare_weights(layer.self_attn, layer, **weight_options)

                def lifecycle_call():
                    compact = layer.self_attn.dsa_attn.dsa_attn.impl._compute_compressor_metadata(
                        fixture["metadata"][fixture["groups"]["compressed"]["prefix"]].decode)
                    NativeHCACall(
                        operators, lifecycle_weights, fixture["hidden"], fixture["positions"], groups,
                        layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                        compact_metadata=compact, output=output,
                    )()

                def lifecycle_native():
                    with set_ascend_forward_context(fixture["metadata"], config, num_tokens=fixture["tokens"],
                                                    num_actual_tokens=fixture["tokens"]):
                        _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"],
                                               fixture["positions"], output)

                check_lifecycle(fixture, output, lifecycle_call, lifecycle_native,
                                layer.self_attn.dsa_attn.dsa_attn.impl, report)
            if args.trajectory_steps:
                from dsv4_hca_trajectory import check_trajectory

                def native_step():
                    with set_ascend_forward_context(fixture["metadata"], config, num_tokens=fixture["tokens"],
                                                    num_actual_tokens=fixture["tokens"]):
                        _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"],
                                               fixture["positions"], output)

                def pto_step():
                    # compact metadata 每步都要按推进后的 metadata 重算，否则消费的是首步的行。
                    compact = layer.self_attn.dsa_attn.dsa_attn.impl._compute_compressor_metadata(
                        fixture["metadata"][fixture["groups"]["compressed"]["prefix"]].decode)
                    NativeHCACall(
                        operators, trajectory_weights, fixture["hidden"], fixture["positions"], groups,
                        layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                        compact_metadata=compact, output=output,
                    )()

                trajectory_weights = prepare_weights(layer.self_attn, layer, **weight_options)
                check_trajectory(
                    fixture, output, native_step, pto_step,
                    layer.self_attn.dsa_attn.dsa_attn.impl, args.trajectory_steps, 20260929, report,
                )
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

                    timed = {"native": native_call, "pto": service_call}
                    refs = {"native": native, "pto": service}
                    if args.timing_direct:
                        # 直接算子调用：compact metadata 预先算好，不含服务入口那段。
                        # 生产里那段每步只算一次、61 层共享（见 dspark/service.py 的
                        # _compact_metadata docstring），单层 bench 会把它整份记在这一层上。
                        timed["pto_direct"] = call
                        refs["pto_direct"] = pto
                    report["timing"] = measure_graph_pair(
                        fixture, output, timed, refs, collect,
                        iters=args.timing_iters, warmup=args.timing_warmup,
                        profile_dir=args.output / "profile" if args.profile else None,
                    )
            if args.swimlane:
                from dsv4_hca_performance import capture_swimlane

                report["swimlane_windows"] = capture_swimlane(fixture, call, args.output / "dfx",
                                                          cold_l2=args.swimlane_cold_l2,
                                                          before=native_call if args.swimlane_after_native else None)
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
