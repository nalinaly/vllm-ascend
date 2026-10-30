# SPDX-License-Identifier: Apache-2.0
"""只用后置ACLNN：短历史、全padding捕图再重放真实请求，不初始化Simpler。"""

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ["VLLM_ASCEND_PTO_CSA_BACKEND"] = "aclnn"
os.environ["PTO_CSA_VARIANT"] = "spmd_tnd"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--target", action="store_true", help="128K/B16；默认为短历史混长边界")
    parser.add_argument("--dump-static", action="store_true", help="只采集静态编译输入，离线CPU编译不占卡")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "scope": "单卡实际service/ACLGraphWrapper，不代表整模型验收"}

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(stage, flush=True)

    save("imports")
    try:
        from contextlib import contextmanager

        import torch
        import torch_npu
        from csa_aclnn_backend_20261028.checks import NoPtoExecution
        from csa_mpmd_tnd_20261021.scenarios import make_fixture
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import guard_checks, make_layer, restore
        from vllm.config import CUDAGraphMode
        from vllm.engine.arg_utils import EngineArgs
        from vllm.forward_context import BatchDescriptor, get_forward_context
        from vllm.platforms import current_platform

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        from vllm_ascend.models.pypto_deepseek_v4 import install_csa_forward, prepare_csa_model
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        from vllm_ascend.utils import enable_custom_op

        guard = NoPtoExecution()
        guard.install()
        device_id = int(os.environ["TASK_DEVICE"])
        assert args.device is None or args.device == device_id
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(1)
        current_platform.pre_register_and_update()
        assert enable_custom_op()
        model = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
        config = EngineArgs(
            model=str(model),
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=131200,
            max_num_seqs=40,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        histories = [131072] * 16 if args.target else [0, 0, 2046, 8190]
        queries = [6] * 16 if args.target else [1, 6, 3, 1]
        tokens, requests = (96, 16) if args.target else (24, 5)
        report.update(device=device_id, histories=histories, queries=queries, tokens=tokens, requests=requests)

        @contextmanager
        def dump_static(label):
            if not args.dump_static:
                yield
                return
            from torchair.core import _torchair

            directory = args.output / ("static_" + label)
            directory.mkdir()
            _torchair.AclopStartDumpArgs(1, str(directory.resolve()))
            try:
                yield
                torch.npu.synchronize()
            finally:
                _torchair.AclopStopDumpArgs(1)

        with native_session(config, device_id), torch.inference_mode():
            device = torch.device(f"npu:{device_id}")
            save("load_layer")
            layer, _ = make_layer(config, model, device, 4)
            fixture = make_fixture(
                config,
                layer.self_attn,
                len(queries),
                max(histories),
                20261030,
                device,
                query_lengths=queries,
                history_lengths=histories,
                token_capacity=tokens,
                request_capacity=requests,
            )
            output = torch.empty_like(fixture["hidden"])
            actual = sum(queries)
            descriptor = BatchDescriptor(tokens, num_reqs=requests, uniform=False)

            def context(graph=False):
                extra = dict(aclgraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=descriptor) if graph else {}
                return set_ascend_forward_context(
                    fixture["metadata"],
                    config,
                    num_tokens=tokens,
                    num_actual_tokens=fixture["actual_tokens"],
                    **extra,
                )

            # Native只消费有效token；同一页布局、权重和本轮Q/L。
            save("native")
            restore(fixture)
            from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import native_decode_capacity_views

            native_metadata = {
                group["prefix"]: native_decode_capacity_views(
                    [fixture["metadata"][group["prefix"]]],
                    tokens,
                    getattr(group["spec"], "compress_ratio", 1),
                )[0]
                for group in fixture["groups"].values()
            }
            with (
                dump_static("native"),
                set_ascend_forward_context(native_metadata, config, num_tokens=tokens, num_actual_tokens=actual),
            ):
                _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)
            torch.npu.synchronize()
            native_output = output[:actual].cpu()
            install_csa_forward(layer)
            prepare_csa_model(SimpleNamespace(model=SimpleNamespace(layers=[layer])))
            runtime = layer.self_attn.dsa_attn._pto_csa_runtime
            assert runtime.native_graph_lifecycle and config._pto_csa_device_policy

            def forbidden(*_):
                raise AssertionError("后置ACLNN不允许重算Host graph signature")

            runtime.graph_signature = forbidden
            for name in ("compressed", "indexer"):
                req = fixture["metadata"][fixture["groups"][name]["prefix"]].decode
                req.sas_metadata = req.qli_metadata = None
                req.native_metadata_factory = forbidden

            def forward(hidden, positions, target):
                torch.ops.vllm.dsv4_csa_forward(hidden, positions, target, layer.self_attn.dsa_attn.prefix)
                return target

            wrapper = ACLGraphWrapper(forward, config, CUDAGraphMode.FULL)

            def call(graph=False):
                with context(graph):
                    assert runtime.eligible(get_forward_context(), fixture["hidden"], fixture["positions"])
                    (wrapper if graph else forward)(fixture["hidden"], fixture["positions"], output)
                torch.npu.synchronize()

            def snapshot():
                return {
                    "output": output.cpu(),
                    **{name: group["allocation"].cpu() for name, group in fixture["groups"].items()},
                }

            save("aclnn_eager")
            restore(fixture)
            with dump_static("aclnn"):
                call()
            expected = snapshot()
            a, n = expected["output"][:actual].float(), native_output.float()
            report["native_output_comparison"] = {
                "max_abs": float((a - n).abs().max()),
                "relative_rmse": float((a - n).square().mean().sqrt() / n.square().mean().sqrt()),
                "finite": bool(torch.isfinite(a).all()),
                "bit_equal": torch.equal(a, n),
            }
            assert report["native_output_comparison"]["finite"]
            report["guards"] = guard_checks(fixture)
            assert all(value["status"] == "PASS" for value in report["guards"].values())

            # 同一容量先捕获全padding。Q/L、start_pos、slots均按Native失活语义设置。
            staged = []
            for item in fixture["metadata"].values():
                req = item.decode
                for name in ("seq_lens", "query_start_loc", "start_pos", "slot_mapping"):
                    value = getattr(req, name, None)
                    if value is not None:
                        staged.append((name, value, value.clone()))
            for name, value, _ in staged:
                value.fill_(-1 if name in ("start_pos", "slot_mapping") else 0)
                if name == "query_start_loc":
                    value[-1] = tokens
            fixture["actual_tokens"] = 0
            save("all_padding_capture")
            restore(fixture)
            call(True)
            entry = wrapper.concrete_aclgraph_entries[descriptor]
            assert entry.csa_tnd_contracts is None
            assert not entry.csa_native_metadata_prefixes
            captures = runtime.submissions
            restore(fixture)
            call(True)
            report["padding_cache_unchanged"] = {
                name: torch.equal(group["allocation"].cpu(), group["initial"])
                for name, group in fixture["groups"].items()
            }
            assert all(report["padding_cache_unchanged"].values())
            # 恢复真实请求，仍用同一图；不得重新进入Python/按policy拒绝replay。
            for _, value, prior in staged:
                value.copy_(prior)
            fixture["actual_tokens"] = actual
            restore(fixture)
            save("real_request_replay")
            call(True)
            replay = snapshot()
            report["replay_bit_equal"] = {name: torch.equal(value, expected[name]) for name, value in replay.items()}
            assert all(report["replay_bit_equal"].values())
            assert runtime.submissions == captures
            report["no_python_redispatch"] = True
            report["no_simpler"] = guard.report(args.output / "unused_build")
            assert report["no_simpler"]["passed"]
            report["passed"] = True
            save("complete")
    except BaseException:
        report["error"] = traceback.format_exc()
        save("error")
        raise


if __name__ == "__main__":
    main()
