"""Run the CSA eager kernel's HBG compile gate without allocating an NPU."""
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[1]
sys.path.insert(0, str(TESTS))
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"] = "0"
os.environ["PTO_CSA_VARIANT"] = "performance"
from dsv4_csa_env import activate

activate()
import vllm_ascend.ops.pypto as ops
ops.__path__ = [str(HERE / "frozen_ops_pypto")]
import pypto.language as pl
from pypto.ir._kernel_compile import kernel_abi_for_program
from pypto.ir.compile import _compile_impl

report = {"runtime": "host_build_graph", "execution": "CPU compile gate only; no NPU initialized", "variant": "performance", "nz_mode": 2, "atomic_add": 0}
try:
    module = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa")
    root = module.decode_csa_tp1_layer_test
    root._get_dep_graph()
    print("CSA dependency graph parsed", flush=True)
    specialization, _ = root._resolve_specialization((), {}, allow_signature_mode=True)
    program = root._compile_to_program(specialization.tensor_meta, specialization.scalar_dtypes, specialization.constexpr_values, specialization.per_func_dyn, pl)
    print("CSA raw program constructed", flush=True)
    abi = kernel_abi_for_program(program, platform="a2a3", runtime="host_build_graph")
    print("CSA HBG kernel ABI derived; entering actual eager compiler gate", flush=True)
    _compile_impl(program, output_dir=str(HERE / "compile_output"), platform="a2a3", _kernel_abi=abi)
    report["status"] = "COMPILE_PASS"
except Exception as exc:
    report.update(status="FAIL", exception_type=type(exc).__name__, error=str(exc))
    traceback.print_exc()
finally:
    (HERE / "compile_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if report["status"] == "FAIL":
    sys.exit(1)
