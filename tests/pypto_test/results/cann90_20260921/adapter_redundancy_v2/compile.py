import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, 'tests/pypto_test')
from dsv4_csa_env import activate
activate()
from pypto.runtime import RunConfig
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_csa import decode_csa_tp1_attention_test
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_metadata import commit_state_window
from dsv4_csa_compressor_boundary import diagnose_indexer_compressor
from dsv4_csa_precision_kernels import diagnose_sparse_attention
root = Path(__file__).parent
report = {'status': 'FAIL', 'scope': 'CPU compile including PTOAS; no device numerical result', 'kernels': []}
try:
    for path in json.loads((root / 'before_sources.json').read_text()):
        ast.parse(Path(path).read_text(), filename=path)
    for kernel in (commit_state_window, decode_csa_tp1_attention_test, diagnose_indexer_compressor, diagnose_sparse_attention):
        print('Compiling', kernel.__name__, flush=True)
        artifact = kernel.compile(config=RunConfig(platform='a2a3'))
        report['kernels'].append({'name': kernel.__name__, 'output_dir': str(artifact.output_dir), 'parameters': kernel.param_names, 'mutable_and_output': kernel.output_param_names})
    report['status'] = 'PASS'
except BaseException as exc:
    report['error'] = repr(exc)
    raise
finally:
    (root / 'compile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
