# SPDX-License-Identifier: Apache-2.0
"""Compile the real HBG eager kernel ABI on CPU, without an NPU allocation."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true", help="检查原有 Tensor-only ABI 的编译兼容性")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dsv4_csa_env import activate

    activate()
    import pypto.language as pl
    from pypto.ir._kernel_compile import kernel_abi_for_program, validate_hbg_kernel_orchestration
    from pypto.ir.compile import _compile_impl
    from pypto.pypto_core import passes
    from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.decode_hca import (
        decode_hca_tp1_layer_hbg, decode_hca_tp1_layer_test,
    )

    root = decode_hca_tp1_layer_test if args.legacy else decode_hca_tp1_layer_hbg
    specialization, _ = root._resolve_specialization((), {}, allow_signature_mode=True)
    program = root._compile_to_program(
        specialization.tensor_meta, specialization.scalar_dtypes,
        specialization.constexpr_values, specialization.per_func_dyn, pl,
    )
    runtime = "tensormap_and_ringbuffer" if args.legacy else "host_build_graph"
    abi = kernel_abi_for_program(program, platform="a2a3", runtime=runtime)
    if not args.legacy:
        validate_hbg_kernel_orchestration(program)
        print("HBG_HOST_CONTRACT_PASS", flush=True)
    kind = passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER if args.legacy else passes.RuntimeKind.HOST_BUILD_GRAPH
    with passes.PassContext([], runtime=kind):
        _compile_impl(program, output_dir=str(args.output), platform="a2a3", _kernel_abi=abi)
    print(f"{runtime}: COMPILE_PASS", flush=True)


if __name__ == "__main__":
    main()
