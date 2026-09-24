# SPDX-License-Identifier: Apache-2.0
"""从精度版重导出：本模块不含任何数值实现，两套必须完全一致。



档位、形状与图重放闸门只管几何，不管数值；而且 model_runner_v1.py 直接从精度版导入这些闸门，两套若各有一份就会在判据上分叉。
"""

from ..deepseek_v4_flash_dspark.service_config import *  # noqa: F401,F403
from ..deepseek_v4_flash_dspark import service_config as _source

__all__ = [name for name in dir(_source) if not name.startswith("_")]
