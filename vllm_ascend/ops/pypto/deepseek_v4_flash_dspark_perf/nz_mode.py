# SPDX-License-Identifier: Apache-2.0
"""复用CSA/HCA公共基础模块nz_mode，不另维护副本。"""

from ..deepseek_v4_flash_dspark.nz_mode import *  # noqa: F401,F403
from ..deepseek_v4_flash_dspark.nz_mode import (  # noqa: F401
    BF16_WEIGHT_LAYOUT,
    BF16_WEIGHT_NZ,
    QUANT_WEIGHT_LAYOUT,
    QUANT_WEIGHT_NZ,
    WEIGHT_NZ_MODE,
    WO_A_WEIGHT_LAYOUT,
)
