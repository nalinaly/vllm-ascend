# SPDX-License-Identifier: Apache-2.0
"""仅构建本地 HC_pre PyTorch 绑定，不安装到共用环境。"""
import argparse
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--recipe-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.recipe_root.resolve(strict=True)
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MAX_JOBS', '4')
import torch_npu
from torch.utils.cpp_extension import load
npu = Path(torch_npu.__file__).parent
source = root / 'ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_hc_pre.cpp'
library = load(name='hc_pre_recipes_reference', sources=[str(Path(__file__).with_name('registration.cpp')), str(source),
                                                       str(source.with_name('ops_common.cpp'))],
               extra_include_paths=[str(npu / 'include'), str(npu / 'include/third_party/acl/inc')],
               extra_cflags=['-O2', '-Wno-deprecated-declarations'],
               extra_ldflags=[f'-L{npu / "lib"}', '-ltorch_npu'],
               build_directory=str(out), is_python_module=False, verbose=True)
record = {'recipe_root': str(root), 'revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
          'source': str(source), 'library': str(library),
          'operator': 'custom::npu_hc_pre', 'scope': '上游融合 HcPre；pre_mix=None、pre_out=None，仅三份输出'}
(out / 'reference.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(record, ensure_ascii=False))
