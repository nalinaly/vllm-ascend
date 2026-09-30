# SPDX-License-Identifier: Apache-2.0
"""HCA 服务图的同地址 metadata A→B→A 重放检查。

A、B 两份 fixture 形状相同（统一页表宽度），历史长度不同，B 的页表行反序。用 A 的地址
捕获服务 custom op 图；每一步把目标状态的 metadata、positions、输入和 cache 初态原地
拷入 A 的地址，先跑 eager、再跑图重放，要求输出与三份完整 allocation 逐 bit 相同，
写保护按目标状态自身的 slot 计算。B 另与在其自身张量上的直接调用逐 bit 对照。
每一步都从同一初态开始，不代表连续多步 decode 轨迹。
"""

import argparse
import os
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_validation import compare_tensor


def _tensor_leaves(value, path, seen, out):
    import inspect

    import torch

    if isinstance(value, torch.Tensor):
        out.append((path, value))
        return
    if value is None or isinstance(value, (int, float, complex, bool, str, bytes)):
        return
    # metadata 里会带上 spec、enum 这类对象引用；类／模块／函数本身没有实例状态，
    # 而且对它们取 vars() 会拿到 __abstractmethods__ 之类的描述符，getattr 直接抛异常。
    if inspect.isclass(value) or inspect.ismodule(value) or inspect.isroutine(value):
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _tensor_leaves(value[key], f"{path}[{key!r}]", seen, out)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _tensor_leaves(item, f"{path}[{index}]", seen, out)
    elif hasattr(value, "__dict__"):
        for key in sorted(vars(value)):
            if key.startswith("__"):
                continue
            try:
                child = getattr(value, key)
            except Exception:
                continue
            _tensor_leaves(child, f"{path}.{key}", seen, out)


def metadata_leaves(fixture):
    out = []
    _tensor_leaves(fixture["metadata"], "metadata", set(), out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"],
                        default="tensormap_and_ringbuffer")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history-a", type=int, default=124)
    parser.add_argument("--history-b", type=int, default=8190)
    parser.add_argument("--device", type=int, choices=[0], default=0)
    parser.add_argument("--operator-source", type=Path)
    parser.add_argument("--weight-nz-mode", type=int, choices=[1, 2], default=2)
    parser.add_argument("--deterministic-level", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"))
    args = parser.parse_args()
    weight_options = {"host_scalars": True} if args.runtime == "host_build_graph" else {}
    if not os.environ.get("TASK_DEVICE"):
        raise RuntimeError("NPU 验证必须通过 task-submit 提交")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = str(args.weight_nz_mode)
    os.environ["HCCL_DETERMINISTIC"] = "true" if args.deterministic_level else "false"
    os.environ["PTO_CSA_RUNTIME"] = args.runtime
    activate()
    if args.operator_source:
        import vllm_ascend.ops.pypto as operator_package

        operator_package.__path__ = [str(args.operator_source.resolve())]
    width = max(args.history_a, args.history_b)
    report = {
        "scope": "单卡服务图同地址 metadata A→B→A；每步同一初态，不是连续 decode 轨迹",
        "status": "RUNNING", "runtime": args.runtime, "batch": args.batch,
        "history_a": args.history_a, "history_b": args.history_b,
        "table_history": width, "task_device": os.environ["TASK_DEVICE"],
        "operator_source": str(args.operator_source.resolve()) if args.operator_source else "当前 worktree",
    }
    try:
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
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
            max_model_len=max(16384, width + 128), max_num_seqs=40, max_num_batched_tokens=256,
            enable_prefix_caching=False, enforce_eager=True, block_size=32,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": args.weight_nz_mode, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        with native_session(config, 0), torch.inference_mode():
            device = torch.device("npu:0")
            layer, details = make_layer(config, args.checkpoint, device, layer_index=3)
            report.update(details)
            fixture_b = make_fixture(config, layer.self_attn, args.batch, args.history_b, 20260929, device,
                                     table_history=width, reverse_pages=True)
            fixture_a = make_fixture(config, layer.self_attn, args.batch, args.history_a, 20260928, device,
                                     table_history=width)
            # 最后构造的 A 已把层内 cache 绑定到 A 的视图；服务路径读取的就是这组固定地址。
            leaves_a, leaves_b = metadata_leaves(fixture_a), metadata_leaves(fixture_b)
            if [p for p, _ in leaves_a] != [p for p, _ in leaves_b]:
                raise ValueError("A/B metadata 结构不同，不能同地址更新")
            for (path, a), (_, b) in zip(leaves_a, leaves_b):
                if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                    raise ValueError(f"A/B metadata 形状不同：{path} {tuple(a.shape)} vs {tuple(b.shape)}")
            for name in fixture_a["groups"]:
                if fixture_a["groups"][name]["allocation"].shape != fixture_b["groups"][name]["allocation"].shape:
                    raise ValueError(f"A/B cache allocation 形状不同：{name}")
            # 两份 fixture 共用或内容相同的叶子（如整张 RoPE 表）无需来回拷贝。
            varying = [index for index, ((_, a), (_, b)) in enumerate(zip(leaves_a, leaves_b))
                       if a.data_ptr() != b.data_ptr() and not torch.equal(a, b)]
            states = {}
            for label, fixture, leaves in (("A", fixture_a, leaves_a), ("B", fixture_b, leaves_b)):
                states[label] = {
                    "fixture": fixture, "hidden": fixture["hidden"].clone(), "positions": fixture["positions"].clone(),
                    "metadata": [leaves[index][1].clone() for index in varying],
                }
            report["metadata_tensors"] = len(leaves_a)
            report["metadata_changed"] = [leaves_a[index][0] for index in varying]
            if not report["metadata_changed"]:
                raise ValueError("A/B metadata 没有差异，不能验证同地址更新")

            def load(label):
                source = states[label]
                fixture_a["hidden"].copy_(source["hidden"])
                fixture_a["positions"].copy_(source["positions"])
                for index, value in zip(varying, source["metadata"]):
                    leaves_a[index][1].copy_(value)
                for name, group in fixture_a["groups"].items():
                    group["allocation"].copy_(source["fixture"]["groups"][name]["initial"].to(device))

            output = torch.empty_like(fixture_a["hidden"])

            def collect():
                return {"output": output.cpu(), **{
                    name: group["allocation"].cpu() for name, group in fixture_a["groups"].items()
                }}

            def guards(label):
                # 写保护与只读检查按目标状态自身的 slot/metadata 计算。
                source = states[label]["fixture"]
                view = {"groups": {name: {"allocation": group["allocation"],
                                          "initial": source["groups"][name]["initial"],
                                          "allowed": source["groups"][name]["allowed"]}
                                   for name, group in fixture_a["groups"].items()},
                        # compact slots 由服务路径在图内重算，fixture 预存的那份不参与本检查。
                        "readonly": {name: (value, source["readonly"][name][1])
                                     for name, (value, _) in fixture_a["readonly"].items()
                                     if not name.endswith("compact_slots")}}
                return guard_checks(view)

            from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.reduction import ATOMIC_ADD
            from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.native_adapter import (
                HCAOperators, NativeHCACall, prepare_weights,
            )

            from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

            pypto.torch.init(device=0, platform="a2a3", runtime=args.runtime, **ring_sizing_kwargs())
            report["atomic_add"] = ATOMIC_ADD
            operators = HCAOperators.register()
            import vllm_ascend.ops.dsv4_hca  # noqa: F401
            from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service import HCAServiceRuntime

            wrapper = layer.self_attn.dsa_attn
            wrapper._pto_hca_layer = (layer,)
            runtime = HCAServiceRuntime(layer.self_attn, operators, 40, layer)
            wrapper._pto_hca_runtime = runtime

            def service_call():
                with set_ascend_forward_context(fixture_a["metadata"], config, num_tokens=fixture_a["tokens"],
                                               num_actual_tokens=fixture_a["tokens"]):
                    from vllm.forward_context import get_forward_context

                    if not runtime.eligible(get_forward_context(), fixture_a["hidden"], fixture_a["positions"]):
                        raise RuntimeError("服务入口未选中 HCA，禁止用 Native fallback 冒充通过")
                    torch.ops.vllm.dsv4_hca_forward(fixture_a["hidden"], fixture_a["positions"], output, wrapper.prefix)

            # B 在其自身张量上的直接调用，作为同地址重放 B 的独立参照。
            direct_output = torch.empty_like(fixture_b["hidden"])
            groups_b = {name: (fixture_b["metadata"][group["prefix"]], tuple(group["views"]))
                        for name, group in fixture_b["groups"].items()}
            direct = NativeHCACall(
                operators, prepare_weights(layer.self_attn, layer, **weight_options), fixture_b["hidden"], fixture_b["positions"],
                groups_b, layer_name=wrapper.dsa_attn.layer_name,
                compact_metadata=fixture_b["compact"]["compressed"], output=direct_output,
            )
            for name, group in fixture_b["groups"].items():
                group["allocation"].copy_(group["initial"].to(device))
            direct()
            torch.npu.synchronize()
            direct_b = {"output": direct_output.cpu(), **{
                name: group["allocation"].cpu() for name, group in fixture_b["groups"].items()
            }}

            load("A")
            service_call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            load("A")
            with torch.npu.graph(graph):
                service_call()
            steps, failed, first_a = [], False, None
            for label in ("A", "B", "A"):
                load(label)
                service_call()
                torch.npu.synchronize()
                eager = collect()
                load(label)
                output.fill_(float("nan"))
                graph.replay()
                torch.npu.synchronize()
                replay = collect()
                step = {
                    "state": label,
                    "graph_eager": {name: compare_tensor(value, eager[name], 0, 0) for name, value in replay.items()},
                    "guards": guards(label),
                    "output_nonfinite": int((~torch.isfinite(replay["output"])).sum()),
                }
                if label == "B":
                    step["graph_direct"] = {name: compare_tensor(value, direct_b[name], 0, 0)
                                            for name, value in replay.items()}
                elif first_a is None:
                    first_a = replay
                else:
                    step["repeat_a"] = {name: compare_tensor(value, first_a[name], 0, 0)
                                        for name, value in replay.items()}
                failed |= any(value["status"] != "PASS" for key in ("graph_eager", "guards", "graph_direct", "repeat_a")
                              for value in step.get(key, {}).values()) or step["output_nonfinite"] != 0
                steps.append(step)
            report["steps"] = steps
            report["status"] = "FAIL" if failed else "PASS"
            if failed:
                raise RuntimeError("同地址 metadata A→B→A 检查失败")
    except BaseException as exc:
        if report.get("status") == "RUNNING":
            report.update(status="FAIL")
        report["error"] = repr(exc)
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
