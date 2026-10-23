# SPDX-License-Identifier: Apache-2.0
"""Python entry for HC_pre; build the extension and configure OPP before import."""
from pathlib import Path

import torch
import torch_npu  # noqa: F401

torch.ops.load_library(str(Path(__file__).with_name("bytenn_pto_ops_npu.so")))


def hc_pre_pto_bnn(x, weight, scale, base):
    return torch.ops.bytenn_pto_ops.hc_pre_pto_bnn(x, weight, scale, base)
