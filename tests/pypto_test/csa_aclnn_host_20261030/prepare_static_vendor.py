# SPDX-License-Identifier: Apache-2.0
"""静态编译试验仅修改私有Native OPP源码副本，复用原二进制。"""

import shutil
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[2]
source = repo / ".cache/csa/csa-native-ops-install/vendors/custom_transformer"
target = here / "results/static_vendor/custom_transformer"
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, target, symlinks=True)
header = target / "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/compressor_metadata/compressor_metadata.h"
old = "Init(CompressorMetadataTilingData* tilingData, TPipe* pipe)"
text = header.read_text()
assert old in text
header.write_text(text.replace(old, "Init(const CompressorMetadataTilingData* tilingData, TPipe* pipe)"))
print(target, flush=True)
