# SPDX-License-Identifier: Apache-2.0
"""HCA 单算子的上线编译口径对照：一次只测一侧（native 或 pto）。

为什么另起一个入口，而不是改 `dsv4_hca_single_layer.py`：

- 那个 bench 通过 `EngineArgs` 把 `enable_npugraph_ex`／`enable_static_kernel` 写进配置，
  但它随后是**直接构造 layer 并自己捕获 NPUGraph**，根本不经过 vLLM 的编译路径，
  所以那两个开关对它不生效——写进去只是写进去了。static_kernel 与 SuperKernel
  必须由 `torch.compile(backend="npugraph_ex")` 这条入口才会真正启用。
- 编译期装静态包会写 `$ASCEND_OPP_PATH/static_kernel`，两侧必须各自从干净目录开始，
  因此不能像原 bench 那样在同一个进程里交替 A/B，只能一侧一个进程。

做法照 CSA 会话的 `coefficients_seven_experiment/compiled_case.py`，按 HCA 调整：
layer 取第 3 层、PTO 走 `torch.ops.vllm.dsv4_hca_forward`、运行时是 `HCAServiceRuntime`、
没有 indexer 因此不校验 Top-K。

⚠ 口径：`inplace_pass` 按用户 2026-09-29 的要求设为 True（CSA 原本是 False）。
`dynamic=False`，每个档位按固定 shape 单独编译，预热后重放计时。
"""

import argparse
import contextlib
import json
import os
import statistics
from pathlib import Path

from dsv4_csa_env import activate

SOURCE = activate()
import torch  # noqa: E402
import torch_npu  # noqa: E402
from dsv4_csa_native_case import native_session  # noqa: E402
from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer, restore  # noqa: E402
from dsv4_hca_compiled_measure import collect_state, measure_graph_interval  # noqa: E402
from vllm.compilation.decorators import support_torch_compile  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

CHECKPOINT = "/data/model/DeepSeek-V4-Flash-0731-w8a8"
LAYER_INDEX = 3
SEED = 20260928


def decode_additional_config(capture_sizes, *, graph=True, recompute_scheduler=False):
    """两侧共用上线 runtime/decode/run_dp_template.sh 的计算配置。

    与 CSA 会话 `offline_pd/run.py` 的 `decode_additional_config` 保持一致：
    上线模板里 `ascend_compilation_config` 只有 `enable_npugraph_ex` 与
    `enable_static_kernel`，`fuse_norm_quant` 取默认 True；给了固定 capture_sizes 时
    要关掉 `align_decode_capture_sizes`，否则档位会被对齐到别的桶。
    """
    return {
        "ascend_compilation_config": {
            "enable_npugraph_ex": graph,
            "enable_static_kernel": graph,
            "fuse_norm_quant": True,
            **({} if not capture_sizes else {"align_decode_capture_sizes": False}),
        },
        "enable_cpu_binding": True,
        "multistream_overlap_shared_expert": True,
        "recompute_scheduler_enable": bool(recompute_scheduler),
    }


@support_torch_compile(dynamic_arg_dims={"hidden_states": 0, "positions": 0, "output": 0})
class CompiledNativeHalf(torch.nn.Module):
    """Native 的 attention 半边：mHC pre + input_layernorm + attention + mHC post。

    与 `vllm_ascend/ops/dsv4_hca.py` 的 `native_attention_half` 逐行同形，
    只是包成 Module 以便 `torch.compile` 接管，调用顺序与参数都不改。
    """

    def __init__(self, *, vllm_config, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor):
        layer = self.layer
        residual = hidden_states.clone()
        mixed, post, comb = layer.hc_pre(
            hidden_states, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        normed = layer.input_layernorm(mixed)
        attended = layer.self_attn(positions=positions, hidden_states=normed, llama_4_scaling=None)
        output.copy_(layer.hc_post(attended, residual, post, comb))
        return output


@support_torch_compile(dynamic_arg_dims={"hidden_states": 0, "positions": 0, "output": 0})
class CompiledPTOHalf(torch.nn.Module):
    """PTO 走生产服务入口，与整机同一条路径。"""

    def __init__(self, *, vllm_config, layer):
        super().__init__()
        self.prefix = layer.self_attn.dsa_attn.prefix

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor):
        torch.ops.vllm.dsv4_hca_forward(hidden_states, positions, output, self.prefix)
        return output


class RequirePTORuntime:
    """任何 Native 回退都直接报错；计数只在捕获期发生，不在重放期。"""

    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0

    def eligible(self, context, hidden, positions):
        if not self.delegate.eligible(context, hidden, positions):
            raise RuntimeError("PTO 用例意外回退到 Native")
        return True

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.delegate(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("native", "pto"), required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--super-kernel", type=int, choices=(0, 1), default=1,
                        help="SuperKernel 开关；分析 incore 时才关掉，其余一律开")
    parser.add_argument("--inplace-pass", type=int, choices=(0, 1), default=1,
                        help="npugraph_ex 的 inplace_pass；按用户要求默认开")
    # CSA 用的是 20/5。这里提到 100/10：单次重放约 0.8 ms，100 次也只有 80 ms，
    # 而实测 20 次的 us_mean 与 us_p50 能差 48 μs（长尾），轮内样本多一些更稳。
    # 真正的大头是进程间差异（两侧各一个进程、无法互相扣漂移，实测 p50 stdev 6～13 μs），
    # 那只能靠重复整轮来压。
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--weight-nz-mode", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--swimlane", action="store_true",
                        help="PTO 侧采 level-4 泳道用于拆解设备侧任务图跨度；"
                             "只作诊断，会把采样数压到很小，计时结果不可用于对比")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if args.swimlane:
        if args.side != "pto":
            raise SystemExit("泳道只对 PTO 侧有意义：Native 侧没有 PyPTO 任务图")
        # 100 次重放会产出巨量记录；诊断只需要几次。
        args.iters, args.warmup = 3, 1
    args.output.mkdir(parents=True, exist_ok=True)
    # 静态包必须落在本侧独占的干净目录里，否则两侧装包互相污染、数字不可归因。
    preexisting = list((Path(os.environ["ASCEND_OPP_PATH"]) / "static_kernel").glob("**/binary_info_config.json"))
    if preexisting:
        raise RuntimeError(f"本侧需要干净的 OPP static_kernel 目录，实到 {len(preexisting)} 份已装包")
    tokens = args.batch * 6
    report = {
        "status": "RUNNING", "side": args.side, "batch": args.batch, "history": args.history,
        "tokens": tokens, "device": args.device, "layer_index": LAYER_INDEX, "source": str(SOURCE),
        "compile_entry": 'torch.compile(backend="npugraph_ex", dynamic=False)',
        "super_kernel": bool(args.super_kernel), "inplace_pass": bool(args.inplace_pass),
        "variant": os.environ.get("PTO_CSA_VARIANT", "（未设置）"),
        "cann": os.environ.get("ASCEND_HOME_PATH"), "opp": os.environ.get("ASCEND_OPP_PATH"),
        "custom_opp": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "weight_nz_mode": args.weight_nz_mode,
        "scope": "单层 attention 半边，正式第 3 层权重，合成历史；不代表整机验收",
    }
    try:
        current_platform.pre_register_and_update()
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Native 自定义算子不可用")
        additional = decode_additional_config([tokens])
        additional.update(weight_nz_mode=args.weight_nz_mode, enable_kv_nz=False, enable_dsa_cp=False)
        config = EngineArgs(
            model=CHECKPOINT, tokenizer_mode="deepseek_v4", trust_remote_code=True, dtype="bfloat16",
            quantization="ascend", tensor_parallel_size=1, hf_overrides={"sliding_window": 128},
            max_model_len=max(16384, args.history + 128), max_num_seqs=40, max_num_batched_tokens=256,
            block_size=32, enable_prefix_caching=False,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [tokens],
                                "cache_dir": str(args.output / "compile_cache")},
            additional_config=additional,
        ).create_engine_config()
        report["requested"] = additional
        report["compile_config"] = str(config.compilation_config)
        from vllm.forward_context import get_forward_context

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_hca import native_attention_half

        with native_session(config, args.device), torch.inference_mode():
            device = torch.device(f"npu:{args.device}")
            layer, weights = make_layer(config, Path(config.model_config.model), device, LAYER_INDEX)
            report["weights"] = weights
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, SEED, device)
            report["layouts"] = {name: group["layout"] for name, group in fixture["groups"].items()}
            output = torch.empty_like(fixture["hidden"])
            runtime = observed = service = compact_cache_key = None
            wrapper = layer.self_attn.dsa_attn
            wrapper._pto_hca_layer = (layer,)
            if args.side == "pto":
                import pypto.torch

                import vllm_ascend.ops.dsv4_hca  # noqa: F401  注册 dsv4_hca_forward
                from vllm_ascend.ops.pypto.deepseek_v4_flash_hca import service as hca_service
                from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.native_adapter import HCAOperators

                # 缓存键定义在 CSA 的 service 里；HCA 的 HCAServiceRuntime 只是继承
                # CSAServiceRuntime，本模块上没有这个名字。
                from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import _COMPACT_METADATA_CACHE

                compact_cache_key = _COMPACT_METADATA_CACHE
                service = hca_service
                pypto.torch.init(
                    device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
                    **({"enable_chip_swimlane": 4, "enable_dep_gen": True,
                        "output_dir": str((args.output / "dfx").resolve())} if args.swimlane else {}),
                )
                runtime = hca_service.HCAServiceRuntime(layer.self_attn, HCAOperators.register(), 40, layer)
                observed = RequirePTORuntime(runtime)
                wrapper._pto_hca_runtime = observed
                report["implementation_package"] = hca_service.__package__
                report["implementation_source"] = str(Path(hca_service.__file__).resolve())
                report["native_weight_formats"] = {
                    name: int(torch_npu.get_npu_format(getattr(layer.self_attn, name).weight))
                    for name in ("wq_a", "wq_b", "wo_a", "wo_b")}

            @contextlib.contextmanager
            def context():
                with set_ascend_forward_context(
                        fixture["metadata"], config, num_tokens=tokens, num_actual_tokens=tokens):
                    if runtime is not None:
                        # 与生产一致：compact metadata 每步只算一次、跨层复用。
                        # 这里预置进 forward context 的缓存，避免 PTO 侧多算两个设备算子。
                        current = get_forward_context()
                        cache = current.additional_kwargs.setdefault(compact_cache_key, {})
                        req = fixture["metadata"][runtime.prefixes["compressed"]].decode
                        cache[id(req)] = (req, fixture["compact"]["compressed"])
                    yield

            def eager_call():
                with context():
                    if runtime is None:
                        native_attention_half(wrapper, fixture["hidden"], fixture["positions"], output)
                    else:
                        torch.ops.vllm.dsv4_hca_forward(
                            fixture["hidden"], fixture["positions"], output, wrapper.prefix)

            # 先用未编译路径取同初态参照，编译后的重放必须与它一致。
            restore(fixture)
            eager_call()
            torch.npu.synchronize()
            reference = collect_state(fixture, output)
            report["initial_guards"] = guard_checks(fixture)
            report["eager_nonfinite"] = int((~torch.isfinite(output)).sum())

            from npugraph_ex._acl_concrete_graph import static_kernel

            static_results = []
            static_super_flags = []
            original_compile = static_kernel.static_compile

            def observe_static(*inputs, **kwargs):
                # 核对 SuperKernel 标记真的传到了静态编译器；只写进 options 不算生效。
                import inspect

                bound = inspect.signature(original_compile).bind_partial(*inputs, **kwargs)
                flag = bool(kwargs.get("super_kernel_optimize",
                                       bound.arguments.get("super_kernel_optimize", False)))
                static_super_flags.append(flag)
                if flag != bool(args.super_kernel):
                    raise RuntimeError("静态编译器收到的 superkernel 标记与请求不一致")
                success = original_compile(*inputs, **kwargs)
                static_results.append(success)
                return success

            static_kernel.static_compile = observe_static
            try:
                options = {
                    "force_eager": False,
                    "inplace_pass": bool(args.inplace_pass),
                    "clone_input": False, "clone_output": False,
                    "static_kernel_compile": True,
                    "super_kernel_optimize": bool(args.super_kernel),
                }
                report["backend_options"] = options
                cls = CompiledNativeHalf if runtime is None else CompiledPTOHalf
                module = cls(vllm_config=config, layer=layer)
                torch.npu.set_compile_mode(jit_compile=False)
                compiled = torch.compile(module, backend="npugraph_ex", fullgraph=True,
                                         dynamic=False, options=options)

                def compiled_call():
                    with context():
                        compiled(fixture["hidden"], fixture["positions"], output)

                restore(fixture)
                compiled_call()
                torch.npu.synchronize()
                report["compiler"] = {
                    "static_compile_results": static_results,
                    "static_super_flags": static_super_flags,
                    "installed_static_packages": len(static_kernel._installed_run_pkgs),
                    "pto_dispatch_calls": observed.calls if observed is not None else 0,
                    # 下面两个只作诊断：本 vLLM 版本里它们不是可靠指标——我们是自己对一个
                    # 已被 @support_torch_compile 装饰的 module 再调 torch.compile，
                    # 内层 wrapper 不会被标记。判据用上面的静态编译证据。
                    "module_compiled_attr": getattr(module, "compiled", None),
                    "module_aot_loaded_attr": getattr(module, "was_aot_compile_fn_loaded_from_disk", None),
                }
                # 判据按侧区分，与 CSA 的 compiled_case 一致：
                #
                # Native 侧的半边是一串 aclnn 算子，static_kernel 必须真的编译并装包，
                # 否则等于没走上线口径。
                # PTO 侧整个半边就是一个自定义算子（dsv4_hca_forward → PyPTO kernel），
                # 图里没有可做静态编译的 aclnn 算子，所以 static_compile 一次都不会触发、
                # 装包数为 0 是**正常**的；这一侧要验证的是服务入口确实被走到。
                # fullgraph=True 下 dynamo 不会静默回退 eager，会直接抛错，故不必另判。
                if any(not r for r in static_results):
                    raise RuntimeError(f"静态编译发生了但未全部成功：{report['compiler']}")
                if runtime is None and (not static_results or not static_kernel._installed_run_pkgs):
                    raise RuntimeError(f"Native 侧要求实际完成静态编译并装包：{report['compiler']}")
                if observed is not None and observed.calls == 0:
                    raise RuntimeError(f"未观察到任何 PTO 服务调用：{report['compiler']}")
                # observe_static 已逐次核对过，这里兜一道总检查（没触发静态编译时为空集）。
                if any(flag != bool(args.super_kernel) for flag in static_super_flags):
                    raise RuntimeError(f"SuperKernel 标记与请求不一致：{report['compiler']}")
                report["timing"] = measure_graph_interval(
                    fixture, compiled_call, output, reference,
                    iters=args.iters, warmup=args.warmup, require_exact=runtime is not None,
                    profile_dir=args.output / "profile" if args.profile else None)
                if args.swimlane:
                    # 光在 pypto.torch.init 里开 enable_chip_swimlane 不会落盘，
                    # 必须显式 begin_dfx/end_dfx 包住一次调用，再导出成带真实任务名的泳道。
                    import pypto.torch as _pt
                    from dsv4_csa_single_card_bench import _export_swimlane

                    restore(fixture)
                    torch.npu.synchronize()
                    _pt.begin_dfx()
                    try:
                        compiled_call()
                    finally:
                        _pt.end_dfx()
                    torch.npu.synchronize()
                    exported = _export_swimlane(
                        args.output / "dfx",
                        kernel_pattern="_jit__decode_hca_tp1_layer_*/kernel_config.py")
                    report["swimlane"] = exported
                    if not exported.get("exported"):
                        raise RuntimeError(f"泳道导出失败：{exported}")
            finally:
                static_kernel.static_compile = original_compile
            if args.save_state:
                torch.save(collect_state(fixture, output), args.output / "states.pt")
            report["us_p50"] = report["timing"]["us_p50"]
            report["us_mean"] = statistics.mean(report["timing"]["samples_us"])
            report["status"] = "MEASURED"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        fields = ("status", "side", "batch", "history", "us_p50", "us_mean", "compiler", "error")
        print(json.dumps({k: report[k] for k in fields if k in report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
