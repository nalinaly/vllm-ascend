"""CPU-only observation of the installed CANN MatMulV2 code generator."""

import json
from pathlib import Path

from impl.ops_legacy.util.util_gemm import _binary_constant_branch
from tbe.common.buildcfg import build_config
from tbe.common.context import op_context
from tbe.common.platform import set_current_compile_soc_info


def descriptor(shape):
    return {"shape": shape, "ori_shape": shape, "format": "ND", "ori_format": "ND", "dtype": "bfloat16"}


set_current_compile_soc_info("Ascend910_9392")
root = Path(__file__).resolve().parent
records = {}
for rows in (24, 48, 96, 144, 192, 240):
    output = root / f"m{rows}"
    output.mkdir(parents=True, exist_ok=True)
    params = {
        "op_type": "MatMulV2",
        "trans_a": False,
        "trans_b": True,
        "kernel_name": f"csa_native_qa_m{rows}",
        "offset_a": 0,
        "offset_b": None,
        "tensor_c": None,
    }
    try:
        with op_context.OpContext("static"):
            with build_config(save_temp_cce_file=True, kernel_meta_parent_dir=str(output)):
                _binary_constant_branch(
                    descriptor([rows, 4096]), descriptor([1024, 4096]), descriptor([rows, 1024]), params, None
                )
        records[rows] = {"status": "PASS", "sources": [str(p) for p in output.rglob("*.cce")]}
    except Exception as error:
        records[rows] = {"status": "FAIL", "error": repr(error)}
    print(rows, records[rows], flush=True)
    (root / "compile_report.json").write_text(json.dumps(records, indent=2) + "\n")
