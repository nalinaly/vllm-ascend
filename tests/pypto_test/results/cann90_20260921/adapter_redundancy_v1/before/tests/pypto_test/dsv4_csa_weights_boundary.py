"""Observe Native linear and replay the shared production head-weight helpers."""

import argparse
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_full_compare import compare_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[24, 48, 96, 144, 192, 240])
    parser.add_argument("--observe-native", action="store_true")
    args = parser.parse_args()
    activate()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import contextlib

    import pypto.torch
    import torch
    import torch_npu
    from dsv4_csa_precision_kernels import T_PAD, diagnose_indexer_weights

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_indexer import WEIGHTS_SCALE

    saved = torch.load(args.capture, map_location="cpu", weights_only=True)
    captured_rows = saved["hidden"].shape[0]
    if any(rows > T_PAD or rows % captured_rows for rows in args.rows):
        raise ValueError("Observation rows must fit the ABI and repeat whole captured batches")
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    native_weight = saved["weights_proj"].T.contiguous().to(device)
    pto_weight = saved["weights_proj"].to(device)
    report = {"scope": "Captured failing hidden values repeated across six shapes; not full CSA validation", "cases": {}}
    inputs = {}
    outputs = {}
    profile = (
        torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1, record_op_args=True, op_attr=True
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output_dir / "native_profile")),
        )
        if args.observe_native
        else contextlib.nullcontext()
    )
    with torch.inference_mode():
        for rows in args.rows:
            repeat = rows // captured_rows
            inputs[rows] = saved["hidden"].repeat(repeat, 1).to(device)
        with profile:
            for rows, x in inputs.items():
                raw = torch.nn.functional.linear(x, native_weight)
                scaled = (raw * WEIGHTS_SCALE).half()
                outputs[rows] = {"native.raw_weights": raw, "native.weights": scaled}
            torch.npu.synchronize()
        if captured_rows in outputs:
            report["native_linear_vs_captured_module"] = compare_tensor(
                outputs[captured_rows]["native.raw_weights"], saved["native.raw_weights"], 0, 0
            )
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        for rows, x in inputs.items():
            repeat = rows // captured_rows
            scales = saved["native.query_scale"].repeat(repeat, 1)
            scale_pad = torch.zeros((T_PAD * 64, 1), dtype=torch.float32, device=device)
            scale_pad[: rows * 64].copy_(scales.reshape(-1, 1).to(device))
            positions = saved["positions"].repeat(repeat).to(device)
            weights = torch.empty((T_PAD, 64), dtype=torch.float32, device=device)
            coefficients = torch.empty((T_PAD, 64), dtype=torch.float16, device=device)
            diagnose_indexer_weights(x, pto_weight, positions, scale_pad, weights, coefficients)
            torch.npu.synchronize()
            expected_weights = outputs[rows]["native.weights"].cpu()
            expected_coefficients = (expected_weights.float() * scales.float()).half()
            outputs[rows].update({"pto.weights": weights[:rows].cpu(), "pto.coefficients": coefficients[:rows].cpu()})
            checks = {
                "weights": compare_tensor(weights[:rows], expected_weights.float(), 0, 0),
                "coefficients_vs_cpu_rounding": compare_tensor(coefficients[:rows], expected_coefficients, 0, 0),
            }
            report["cases"][rows] = checks
            print(rows, {name: value["mismatches"] for name, value in checks.items()}, flush=True)
        report["status"] = "PASS" if all(
            check["status"] == "PASS" for case in report["cases"].values() for check in case.values()
        ) and report.get("native_linear_vs_captured_module", {"status": "PASS"})["status"] == "PASS" else "FAIL"
        torch.save({rows: {name: value.cpu() for name, value in values.items()} for rows, values in outputs.items()},
                   args.output_dir / "weights.pt")
        write_json(args.output_dir / "weights.json", report)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
