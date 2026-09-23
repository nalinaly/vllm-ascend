"""Compare CSA projection primitives on synthetic ND inputs, without a checkpoint."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def compare(actual, expected, *, atol, rtol):
    """Count every element outside the frozen elementwise tolerance."""
    import torch

    left = actual.detach().cpu().float()
    right = expected.detach().cpu().float()
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    error = (left - right).abs()
    finite = torch.isfinite(left) & torch.isfinite(right)
    mismatch = ~finite | (error > atol + rtol * right.abs())
    return {
        "status": "PASS" if not mismatch.any() else "FAIL",
        "shape": list(left.shape),
        "dtype": str(actual.dtype),
        "atol": atol,
        "rtol": rtol,
        "mismatches": int(mismatch.sum()),
        "elements": left.numel(),
        "max_abs_error": float(error.max()),
        "nonfinite": int((~finite).sum()),
    }


def run_case(tokens, seed, kernels, device, result):
    import torch
    import torch_npu

    generator = torch.Generator().manual_seed(seed)

    def normal(shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(dtype=torch.bfloat16, device=device)

    hidden = normal((tokens, 4096))
    qa_weight = normal((4096, 1024), 0.015625)
    kv_weight = normal((4096, 512), 0.015625)
    gamma = (0.5 + torch.rand(1024, generator=generator)).to(dtype=torch.bfloat16, device=device)
    qb_weight = torch.randint(-127, 128, (1024, 32768), generator=generator, dtype=torch.int8).to(device)
    qb_scale = (0.001 + torch.rand(32768, generator=generator) * 0.01).to(dtype=torch.bfloat16, device=device)
    qa_native = hidden @ qa_weight
    kv_native = hidden @ kv_weight
    qa_pto = torch.empty_like(qa_native)
    kv_pto = torch.empty_like(kv_native)
    kernels["bf16"](hidden, qa_weight, qa_pto)
    kernels["bf16"](hidden, kv_weight, kv_pto)

    qr_native, scale_native = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(qa_native, gamma, epsilon=1e-6)
    qr_pto = torch.empty_like(qr_native)
    scale_pto = torch.empty((tokens, 1), dtype=torch.float32, device=device)
    kernels["rms"](qa_native, gamma, qr_pto, scale_pto)
    qr_chain = torch.empty_like(qr_native)
    scale_chain = torch.empty_like(scale_pto)
    kernels["rms"](qa_pto, gamma, qr_chain, scale_chain)

    qb_native = torch_npu.npu_quant_matmul(
        qr_native, qb_weight, qb_scale, pertoken_scale=scale_native, output_dtype=torch.bfloat16
    )
    qb_pto = torch.empty_like(qb_native)
    kernels["int8"](qr_native, qb_weight, scale_native.reshape(tokens, 1), qb_scale, qb_pto)
    qb_chain = torch.empty_like(qb_native)
    kernels["int8"](qr_chain, qb_weight, scale_chain, qb_scale, qb_chain)
    torch.npu.synchronize()

    bf16_rule = {"atol": 1e-2, "rtol": 1e-2}
    exact = {"atol": 0.0, "rtol": 0.0}
    scale_rule = {"atol": 1e-6, "rtol": 1e-5}
    checks = {
        "qa_projection": compare(qa_pto, qa_native, **bf16_rule),
        "kv_projection": compare(kv_pto, kv_native, **bf16_rule),
        "qr_isolated": compare(qr_pto, qr_native, **exact),
        "qr_scale_isolated": compare(scale_pto, scale_native.reshape(tokens, 1), **scale_rule),
        "qb_isolated": compare(qb_pto, qb_native, **bf16_rule),
        "qr_chained": compare(qr_chain, qr_native, **exact),
        "qr_scale_chained": compare(scale_chain, scale_native.reshape(tokens, 1), **scale_rule),
        "qb_chained": compare(qb_chain, qb_native, **bf16_rule),
    }
    result["checks"] = checks
    positions = torch.arange(131072, 131072 + tokens, dtype=torch.float32)
    frequencies = torch.exp(-torch.arange(32, dtype=torch.float32) / 32 * 9.210340371976184)
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos().repeat_interleave(2, dim=1).to(device)
    sin = angles.sin().repeat_interleave(2, dim=1).to(device)
    for name, native_input, chained_input, norm_weight in (
        ("q", qb_native.reshape(tokens, 64, 512), qb_chain.reshape(tokens, 64, 512), torch.ones(512)),
        (
            "kv",
            kv_native.reshape(tokens, 1, 512),
            kv_pto.reshape(tokens, 1, 512),
            0.5 + torch.rand(512, generator=generator),
        ),
    ):
        norm_weight = norm_weight.to(dtype=torch.bfloat16, device=device)
        native_rope, _ = torch_npu.npu_rms_norm(native_input, norm_weight, epsilon=1e-6)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            native_rope.unsqueeze(1),
            cos.reshape(tokens, 1, 1, 64),
            sin.reshape(tokens, 1, 1, 64),
            rotary_mode="interleave",
            partial_slice=[448, 512],
        )
        pto_rope = torch.empty_like(native_rope)
        kernels["rope"](native_input, norm_weight, cos, sin, pto_rope)
        checks[name + "_norm_rope_isolated"] = compare(pto_rope, native_rope, **bf16_rule)
        kernels["rope"](chained_input, norm_weight, cos, sin, pto_rope)
        checks[name + "_norm_rope_chained"] = compare(pto_rope, native_rope, **bf16_rule)
    layouts = {
        name: int(torch_npu.get_npu_format(tensor))
        for name, tensor in {
            "hidden": hidden,
            "qa_weight": qa_weight,
            "kv_weight": kv_weight,
            "qb_weight": qb_weight,
            "qa_pto": qa_pto,
            "qb_pto": qb_pto,
            "qa_native": qa_native,
            "kv_native": kv_native,
            "qb_native": qb_native,
        }.items()
    }
    assert all(layout in (0, 2) for layout in layouts.values()), layouts
    return {
        "tokens": tokens,
        "seed": seed,
        "status": "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL",
        "checks": checks,
        "npu_formats": layouts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[24, 48, 96, 144, 192, 240])
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(tokens <= 0 or tokens > 400 for tokens in args.tokens):
        parser.error("tokens must be between 1 and 400")
    repo = activate()
    report = {
        "status": "FAIL",
        "scope": "synthetic_ND_projection_primitives_only_not_complete_CSA_or_target_checkpoint",
        "checkpoint": None,
        "cases": [],
        "contract": {
            "rms_epsilon": 1e-6,
            "bf16": {"atol": 1e-2, "rtol": 1e-2},
            "int8": "exact",
            "token_scale_fp32": {"atol": 1e-6, "rtol": 1e-5},
            "quant_rounding": "FP32_RINT_to_INT32_then_FP16_then_INT8_TRUNC",
            "weight_scale": "BF16_as_consumed_by_native_projection",
            "accepted_base_formats": [0, 2],
            "rope": "FP32_interleaved_cos_sin_BF16_round_after_norm_and_after_rope",
        },
    }
    output_file = args.output_dir / f"projection_compare_device{args.device}.json"
    write_json(output_file, report)
    try:
        import pypto.torch
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_norm_rope_kernel import norm_rope
        from dsv4_csa_qkv_kernel import bf16_projection, q_rms_quant, w8a8_projection

        report["extension"] = str(load_native_extension(repo))
        torch.npu.set_device(args.device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        kernels = {
            "bf16": pypto.torch.register(bf16_projection, "dsv4_csa_test::bf16_projection"),
            "rms": pypto.torch.register(q_rms_quant, "dsv4_csa_test::q_rms_quant"),
            "int8": pypto.torch.register(w8a8_projection, "dsv4_csa_test::w8a8_projection"),
            "rope": pypto.torch.register(norm_rope, "dsv4_csa_test::norm_rope"),
        }
        for tokens in args.tokens:
            result = {"tokens": tokens, "seed": args.seed, "status": "FAIL"}
            report["cases"].append(result)
            result.update(run_case(tokens, args.seed, kernels, torch.device(f"npu:{args.device}"), result))
            write_json(output_file, report)
            print(f"projection primitives T={tokens}: {result['status']}", flush=True)
        report["peak_npu_allocated_bytes"] = torch.npu.max_memory_allocated()
        assert all(case["status"] == "PASS" for case in report["cases"]), "projection numerical comparison failed"
        report["status"] = "PASS"
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(output_file, report)


if __name__ == "__main__":
    main()
# Retired diagnostic entry; not part of the CSA validation matrix.
