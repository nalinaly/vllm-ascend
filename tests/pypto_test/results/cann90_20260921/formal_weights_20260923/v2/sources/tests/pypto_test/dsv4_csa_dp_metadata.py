"""Two real TP1/DP2 ranks: native coordination, graph dispatch and PTO probes.

No full model, experts, draft model or checkpoint payloads are loaded. Idle-rank
cases cover coordination only; full CSA and the model runner dummy path remain
separate acceptance requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from dsv4_csa_env import activate, load_native_extension, write_json


def run_worker(args):
    repo = activate()
    report = {"status": "FAIL", "rank": args.rank, "cases": [], "scope": "DP2_metadata_only_no_full_CSA"}
    try:
        import pypto.torch
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_all_groups import run_batch
        from dsv4_csa_metadata_kernel import expand_compressed_slots, metadata_probe
        from dsv4_csa_native_fixture import make_attention, make_cache_groups, make_config, native_session
        from vllm.config import CUDAGraphMode
        from vllm.distributed.parallel_state import get_dp_group, get_tp_group, get_world_group
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
        from vllm_ascend.utils import is_moe_model, should_skip_allreduce_across_dp_group
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        config = make_config(args.checkpoint, dp_size=2, dp_rank=args.rank, dp_port=args.port, full_decode_graph=True)
        device = torch.device(f"npu:{args.rank}")
        with native_session(config, args.rank):
            assert get_world_group().world_size == get_dp_group().world_size == 2
            assert get_tp_group().world_size == 1
            assert is_moe_model(config), "must retain the actual DeepSeek MoE classification"
            skip = should_skip_allreduce_across_dp_group(config)
            assert not skip, "this fixture must execute the real DP collective"
            attention = make_attention(config, device)
            groups = make_cache_groups(config, device, attention)
            support = AscendDSAMetadataBuilder.get_cudagraph_support(config, groups["compressed"]["spec"])
            mode = config.compilation_config.resolve_cudagraph_mode_and_sizes(
                support, "AscendDSA", uniform_decode_query_len=6, tensor_parallel_size=1
            )
            dispatcher = CudagraphDispatcher(config)
            dispatcher.initialize_cudagraph_keys(mode, uniform_decode_query_len=6)
            # Only the fields read by this real runner method are needed. The
            # method calls the real DP CPU process group and torch all_reduce.
            runner_fields = SimpleNamespace(dp_size=2, dp_rank=args.rank, vllm_config=config)
            pypto.torch.init(device=args.rank, platform="a2a3", runtime="tensormap_and_ringbuffer")
            probe = pypto.torch.register(metadata_probe, "dsv4_csa_test::dp_probe")
            expand = pypto.torch.register(expand_compressed_slots, "dsv4_csa_test::dp_expand")
            report.update(
                world_size=2,
                tp_size=1,
                dp_size=2,
                moe_classification=True,
                skip_dp_allreduce=skip,
                dp_cpu_backend=torch.distributed.get_backend(get_dp_group().cpu_group),
                world_device_backend=torch.distributed.get_backend(get_world_group().device_group),
                capture_sizes=config.compilation_config.cudagraph_capture_sizes,
                kv_transfer_config=None,
                note="real MoE synchronization branch; PD consumer and EP16 communication selection not covered",
            )
            pairs = [(4, 40), (40, 4), (8, 24), (16, 32), (0, 4), (0, 40), (4, 40)]
            for step, pair in enumerate(pairs):
                batch = pair[args.rank]
                local_tokens = batch * 6
                runtime_mode, local_descriptor = dispatcher.dispatch(local_tokens, uniform_decode=bool(batch))
                # An idle rank must participate in coordination. Explicit NONE
                # also exercises the same downgrade when any rank cannot graph.
                if not batch or step == len(pairs) - 1 and args.rank == 0:
                    runtime_mode = CUDAGraphMode.NONE
                modes = [None, None]
                torch.distributed.all_gather_object(modes, runtime_mode.value, group=get_dp_group().cpu_group)
                expected_mode = CUDAGraphMode(min(modes))
                case = {"step": step, "batch_pair": pair, "local_mode": runtime_mode.name, "sync": []}
                for allow_padding in (False, True):
                    maximum, counts, synced = NPUModelRunner._sync_metadata_across_dp(
                        runner_fields, local_tokens, cudagraph_mode=runtime_mode, allow_dp_padding=allow_padding
                    )
                    expected_counts = [max(pair) * 6] * 2 if allow_padding else [b * 6 for b in pair]
                    assert maximum == max(pair) * 6
                    assert counts.tolist() == expected_counts
                    assert synced == expected_mode
                    case["sync"].append({"allow_padding": allow_padding, "maximum": maximum, "counts": counts.tolist()})
                selected_mode, descriptor = dispatcher.dispatch(maximum, uniform_decode=True, valid_modes={synced})
                case.update(
                    synced_mode=synced.name,
                    selected_mode=selected_mode.name,
                    local_descriptor=asdict(local_descriptor),
                    common_descriptor=asdict(descriptor),
                )
                if batch:
                    case["probe"] = run_batch(
                        config, device, batch, groups, probe, expand, graph_capacity=descriptor.num_tokens
                    )
                    # run_batch checks its own eager and probe-only graph even
                    # when the model dispatcher selected NONE. Do not conflate.
                    case["probe_scope"] = "independent_probe_graph_at_coordinated_capacity"
                else:
                    case["probe_scope"] = "idle_coordination_only_model_dummy_path_NOT_RUN"
                report["cases"].append(case)
                write_json(args.output_dir / f"dp_metadata_rank{args.rank}.json", report)
                print(f"DP2 rank={args.rank} pair={pair} mode={synced.name}: PASS", flush=True)
            report["status"] = "PASS"
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / f"dp_metadata_rank{args.rank}.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--port", type=int)
    parser.add_argument("--device", help="Physical device list appended by task-submit; must match the wrapper mask")
    args = parser.parse_args()
    if args.device is not None and args.device != os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("--device must match the two-device ASCEND_RT_VISIBLE_DEVICES mask set by the queue wrapper")
    if args.rank is not None:
        run_worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    workers = []
    files = []
    try:
        for rank in range(2):
            log = (args.output_dir / f"rank{rank}.log").open("w")
            files.append(log)
            workers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--checkpoint",
                        str(args.checkpoint),
                        "--output-dir",
                        str(args.output_dir),
                        "--rank",
                        str(rank),
                        "--port",
                        str(port),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [worker.wait(timeout=1200) for worker in workers]
        if any(codes):
            raise RuntimeError(f"DP workers failed: {codes}; see {args.output_dir}/rank*.log")
        results = [json.loads((args.output_dir / f"dp_metadata_rank{rank}.json").read_text()) for rank in range(2)]
        assert all(result["status"] == "PASS" for result in results)
        write_json(args.output_dir / "dp_metadata_summary.json", {"status": "PASS", "ranks": results})
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for log in files:
            log.close()


if __name__ == "__main__":
    main()
