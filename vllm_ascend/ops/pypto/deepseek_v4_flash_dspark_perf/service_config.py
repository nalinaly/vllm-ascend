# SPDX-License-Identifier: Apache-2.0
"""复用CSA/HCA公共基础模块service_config，不另维护副本。"""

from ..deepseek_v4_flash_dspark import service_config as _source
from ..deepseek_v4_flash_dspark.service_config import *  # noqa: F401,F403

__all__ = [name for name in dir(_source) if not name.startswith("_")]
