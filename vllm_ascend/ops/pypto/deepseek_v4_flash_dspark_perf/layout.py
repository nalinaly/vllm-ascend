# SPDX-License-Identifier: Apache-2.0
"""复用CSA/HCA公共基础模块layout，不另维护副本。"""

from ..deepseek_v4_flash_dspark import layout as _source
from ..deepseek_v4_flash_dspark.layout import *  # noqa: F401,F403

__all__ = [name for name in dir(_source) if not name.startswith("_")]

# Cache policy is specified at the corresponding kernel load sites. ND/NZ
# variants now use BYPASS selectively; the old global "NZ disabled/no BYPASS"
# description no longer describes the retained implementation.
