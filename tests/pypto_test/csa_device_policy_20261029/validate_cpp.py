# SPDX-License-Identifier: Apache-2.0
"""单卡对比合并C++与原对应策略，直接加载已有二进制，禁止JIT重生成。"""

import argparse
import copy
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--case", choices=("C01", "C04", "short_b16", "mixed_padding"), default="C01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "0")))
    parser.add_argument("--selector-probe", action="store_true")
    parser.add_argument("--cross-policy-replay", action="store_true")
    parser.add_argument("--profile-windows", type=int, default=0)
    parser.add_argument("--aclnn-library", type=Path)
    parser.add_argument("--aclnn-vendor", type=Path)
    parser.add_argument("--formal-aclnn-package", type=Path)
    args = parser.parse_args()
    if args.profile_windows and args.profile_windows < 10:
        parser.error("稳定态L1每种实现至少10个窗口")
    if args.aclnn_library:
        if not args.aclnn_vendor or args.profile_windows:
            parser.error("ACLNN验证需要独立vendor路径；不使用Simpler L1计时")
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.aclnn_vendor.resolve())
    if args.formal_aclnn_package:
        if args.aclnn_library or args.profile_windows:
            parser.error("正式ACLNN入口验证不混用直接库或Simpler L1")
        sys.path.insert(0, str(args.formal_aclnn_package.resolve()))
        os.environ["VLLM_ASCEND_PTO_CSA_BACKEND"] = "aclnn"
        os.environ["PTO_CSA_VARIANT"] = "spmd_tnd"
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    report = dict(case=args.case, status="initializing", jit_regeneration=False, model_test=False)

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("CPP94 " + stage, flush=True)

    try:
        import pypto.torch
        import torch
        import torch_npu
        from csa_torch_repro.inputs import make_inputs, restore
        from pypto._kernel_abi import KernelABI
        from pypto.jit._artifact_manifest import BuildKind
        from pypto.runtime._kernel_artifact import KernelArtifact
        from pypto.runtime._prebuilt import load_prebuilt
        from pypto.runtime.kernel.context import get_process_kernel_state
        from pypto.torch.launch import describe_eager_call, invoke

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy

        if args.aclnn_library:
            import ctypes

            _vendor_handle = ctypes.CDLL(
                str(args.aclnn_vendor / "op_api/lib/libcust_opapi.so"), mode=ctypes.RTLD_GLOBAL
            )
            torch.ops.load_library(str(args.aclnn_library))
            report["candidate_backend"] = "ACLNN device policy, fixed tiling key 0"

        device_id = args.device
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(1)
        device = torch.device(f"npu:{device_id}")
        pypto.torch.init(
            device=device_id,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            ring_heap=[64 * 1024**2, 512 * 1024**2, 64 * 1024**2, 32 * 1024**2],
            ring_task_window=[4096] * 4,
            enable_chip_swimlane=int(bool(args.profile_windows)),
            enable_dep_gen=False,
            output_dir=str(args.output / "dfx"),
        )
        bound = get_process_kernel_state().require_config()
        cases = json.loads((repo / "tests/pypto_test/csa_torch_repro/cases.json").read_text())
        case = copy.deepcopy(
            next(
                c
                for c in cases["cases"]
                if c["name"] == ("C04" if args.case in ("short_b16", "mixed_padding") else args.case)
            )
        )
        if args.case == "short_b16":
            case.update(
                name="short_b16",
                history_label=8192,
                table_history=8192,
                history_lengths=[8192] * 16,
                seq_lens=[8198] * 16,
            )
        if args.case == "mixed_padding":
            queries = [1, 2, 3, 4, 5, 6, 1, 2]
            histories = [131072, 65536, 32768, 8192, 98304, 131072, 8192, 16384]
            bounds = [0]
            for query in queries:
                bounds.append(bounds[-1] + query)
            case.update(
                name="mixed_padding",
                active_requests=8,
                query_lengths=queries,
                history_lengths=histories,
                valid_tokens=sum(queries),
                query_start_loc=bounds + [case["token_capacity"]] * 8,
                seq_lens=[h + q for h, q in zip(histories, queries)] + [0] * 8,
            )
        report.update(scenario=case, device=device_id)
        save("preparing_inputs")
        with torch.inference_mode():
            native, groups, readonly, _ = make_inputs(
                case, Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"), cases["layer"], device
            )
            # 旧独立输入生成器仍提供4张NZ；94份产物的ABI已采用7张NZ。
            # WKV和Indexer head权重沿Native的[N,K]方向，转换仅在输入准备期进行。
            for name in ("wkv", "weights_proj"):
                native[name] = torch_npu.npu_format_cast(native[name].T.contiguous(), 29)
            native["idx_wq_b"] = torch_npu.npu_format_cast(native["idx_wq_b"], 29)
            maximum_q = max(case["query_lengths"])
            maximum_l = max(case["seq_lens"])
            prepared = PreparedDynamicCSA(
                {k: v for k, v in native.items() if k not in ("idx_topk", "idx_topk_scores")},
                max_query_tokens=maximum_q,
                max_seq_len=maximum_l,
            )
            policy = select_policy(
                tokens=case["token_capacity"],
                requests=case["request_capacity"],
                max_query_tokens=maximum_q,
                max_seq_len=maximum_l,
                compressed_table_columns=native["cmp_block_table"].shape[1],
            ).constexprs()
            records = json.loads(
                (repo / "tests/pypto_test/csa_policy_compile_20261027/compile_results.json").read_text()
            )["results"]
            reference = next(r for r in records if r["mode"] == "spmd" and r["policy"] == policy)
            report["reference_policy"] = reference["index"]

            def load(path):
                path = Path(path)
                abi = KernelABI.from_record(json.loads((path / "compiled_meta.json").read_text())["kernel_abi"])
                artifact = KernelArtifact(path, abi)
                artifact._loaded = load_prebuilt(path, abi.platform, BuildKind.SINGLE_CHIP, kernel_abi=abi)["."]
                parameters = [prepared.buffers[p.name] for p in abi.parameters]

                def call():
                    # 捕获时须读取当前stream/capture，不能复用eager阶段的CallFrame。
                    frame = describe_eager_call(abi, parameters, bound)
                    return invoke(artifact, frame, bound)

                return call

            reference_call, merged_call = load(reference["artifact"]), load(args.artifact)
            if args.formal_aclnn_package:
                assert prepared.execution_backend == "aclnn" and not prepared.allocations
                assert prepared.policy.constexprs() == {}
                merged_call = prepared
                report["candidate_backend"] = "PreparedDynamicCSA backend=aclnn device_policy"
                report["host_policy_key"] = prepared.policy.constexprs()
            if args.aclnn_library:
                parameters = json.loads((args.artifact / "compiled_meta.json").read_text())["params"]
                candidate_arguments = [prepared.buffers[p["name"]] for p in parameters]

                def merged_call():
                    torch.ops.csa_pto_device.run(candidate_arguments, 0)

            def snapshot():
                values = {"output": native["x_out"].cpu().clone()}
                guards = {}
                for name, group in groups.items():
                    raw = group["allocation"].cpu()
                    guards[name] = bool(((raw == group["initial"]) | group["allowed"]).all())
                    pairs = sorted({(p, r) for p, r in group["slots"].tolist() if p >= 0 and r >= 0})
                    pages, rows = [p for p, _ in pairs], [r for _, r in pairs]
                    for i, view in enumerate(group["views"]):
                        current = raw.view(view.dtype).as_strided(view.shape, view.stride(), view.storage_offset())
                        values[f"{name}.{i}"] = current[pages, rows].clone()
                for name, (value, initial) in readonly.items():
                    guards["readonly/" + name] = torch.equal(value.cpu(), initial)
                return values, guards

            save("reference")
            restore(native, groups)
            reference_call()
            torch.npu.synchronize()
            expected, reference_guards = snapshot()
            save("merged")
            restore(native, groups)
            started = time.perf_counter()
            merged_call()
            torch.npu.synchronize()
            report["first_merged_wall_s"] = time.perf_counter() - started
            if args.selector_probe:
                selected = native["x_out"].view(torch.int32).flatten()[: 72 * 16 : 16].cpu().tolist()
                report["selected_by_core"] = selected
                assert selected == [reference["index"]] * 72, selected
                report.update(status="PASS", selector_only=True)
                save("complete")
                return 0
            actual, guards = snapshot()
            report["bitwise"] = {key: torch.equal(value, expected[key]) for key, value in actual.items()}
            report["guards"] = guards
            report["reference_guards"] = reference_guards
            assert all(report["bitwise"].values()), report["bitwise"]
            assert all(guards.values()) and all(reference_guards.values())
            save("capture_replay")
            restore(native, groups)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                merged_call()
            restore(native, groups)
            graph.replay()
            torch.npu.synchronize()
            replay, replay_guards = snapshot()
            report["replay_bitwise"] = {key: torch.equal(value, expected[key]) for key, value in replay.items()}
            assert all(report["replay_bitwise"].values()) and all(replay_guards.values())
            if args.profile_windows:
                from csa_tnd_nz_weights_20261027.l1 import export

                baseline_graph = torch.npu.NPUGraph()
                restore(native, groups)
                with torch.npu.graph(baseline_graph):
                    reference_call()
                graphs = {"original": baseline_graph, "merged": graph}
                for _ in range(10):
                    for current in graphs.values():
                        restore(native, groups)
                        current.replay()
                        torch.npu.synchronize()
                report["performance"] = dict(
                    warmup_per_side=10, replay_per_window=1, boundary="L1 incore首尾", samples=[]
                )
                save("stable_l1")
                samples = report["performance"]["samples"]
                for sample in range(args.profile_windows):
                    order = ("original", "merged") if sample % 2 == 0 else ("merged", "original")
                    for side in order:
                        restore(native, groups)
                        torch.npu.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            graphs[side].replay()
                        finally:
                            pypto.torch.end_dfx()
                        source = args.output / "dfx"
                        if samples:
                            source /= f"window_{len(samples)}"
                        artifact_dir = Path(reference["artifact"]) if side == "original" else args.artifact
                        samples.append(
                            export(
                                source,
                                args.output / f"l1_{sample:02d}_{side}",
                                side=side,
                                variant="spmd",
                                kernel_config=artifact_dir / "kernel_config.py",
                            )
                        )
                summary = {}
                for side in graphs:
                    values = [s["incore_span_us"] for s in samples if s["side"] == side]
                    summary[side] = dict(
                        min=min(values),
                        median=statistics.median(values),
                        mean=statistics.mean(values),
                        max=max(values),
                        count=len(values),
                    )
                report["performance"]["statistics_us"] = summary
                report["performance"]["median_change_percent"] = (
                    summary["merged"]["median"] / summary["original"]["median"] - 1
                ) * 100
                restore(native, groups)
                graph.replay()
                torch.npu.synchronize()
                last, last_guards = snapshot()
                assert all(torch.equal(value, expected[key]) for key, value in last.items()) and all(
                    last_guards.values()
                )
            if args.cross_policy_replay:
                assert args.case == "C04", "跨档测试固定B16/S6，容量和输入地址保持不变"
                save("cross_policy_inputs")
                alternate = copy.deepcopy(case)
                alternate.update(history_label=8192, history_lengths=[8192] * 16, seq_lens=[8198] * 16)
                _, changed_groups, changed_readonly, _ = make_inputs(
                    alternate, Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"), cases["layer"], device
                )
                # 真实Q/L、位置、slot和RoPE一起更新，避免不自洽的伪跨档输入。
                # 页表和cache沿用128K容量；只有内容变化，不重新捕获图或重新绑定地址。
                for name, (value, _) in tuple(readonly.items()):
                    changed, initial = changed_readonly[name]
                    assert value.shape == changed.shape
                    value.copy_(changed)
                    readonly[name] = (value, initial)
                for name, group in groups.items():
                    for field in ("initial", "slots", "allowed"):
                        group[field] = changed_groups[name][field]
                next_policy = select_policy(
                    tokens=case["token_capacity"],
                    requests=case["request_capacity"],
                    max_query_tokens=maximum_q,
                    max_seq_len=8198,
                    compressed_table_columns=native["cmp_block_table"].shape[1],
                ).constexprs()
                next_reference = next(r for r in records if r["mode"] == "spmd" and r["policy"] == next_policy)
                assert next_reference["index"] != reference["index"]
                report["cross_policy_ids"] = [reference["index"], next_reference["index"]]
                next_call = load(next_reference["artifact"])
                restore(native, groups)
                next_call()
                torch.npu.synchronize()
                short_expected, short_reference_guards = snapshot()
                restore(native, groups)
                save("cross_policy_replay")
                graph.replay()
                torch.npu.synchronize()
                short_actual, short_guards = snapshot()
                report["cross_policy_bitwise"] = {
                    key: torch.equal(value, short_expected[key]) for key, value in short_actual.items()
                }
                report["cross_policy_guards"] = short_guards
                assert (
                    all(report["cross_policy_bitwise"].values())
                    and all(short_guards.values())
                    and all(short_reference_guards.values())
                )
        report["status"] = "PASS"
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    save("complete")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
