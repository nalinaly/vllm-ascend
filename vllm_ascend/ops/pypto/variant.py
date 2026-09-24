# SPDX-License-Identifier: Apache-2.0
"""在 CSA 的精度版与性能版之间选择实现。

两套算子并存：性能版（`deepseek_v4_flash_dspark_perf`）改用 pypto-lib 上游的数值
写法以追平其性能，精度版（`deepseek_v4_flash_dspark`）与 Native 逐 bit 一致。

选择走环境变量而不是 `additional_config`：切换的用途是对照实验，环境变量最不侵入
生产配置结构，测试驱动也便于按轮次切换。

**默认是性能版**（用户 2026-09-24 定）。这意味着不设该变量时走的不再是与 Native
逐 token 一致的那一套，做精度验收必须显式 `PTO_CSA_VARIANT=precision`。
"""

import os

_PRECISION = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark"
_PERFORMANCE = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf"
_ENV = "PTO_CSA_VARIANT"
_ALIASES = {"precision": "precision", "prec": "precision",
            "performance": "performance", "perf": "performance"}


def selected_variant() -> str:
    """返回 "precision" 或 "performance"；无法识别的取值必须报错而不是静默回退。"""
    value = os.environ.get(_ENV, "performance").strip().lower()
    if value not in _ALIASES:
        raise ValueError(f"{_ENV} must be one of {sorted(_ALIASES)}, got {value!r}")
    return _ALIASES[value]


def variant_package() -> str:
    return _PRECISION if selected_variant() == "precision" else _PERFORMANCE
