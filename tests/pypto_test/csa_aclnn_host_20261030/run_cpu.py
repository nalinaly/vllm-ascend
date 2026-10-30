# SPDX-License-Identifier: Apache-2.0
"""只运行受影响的CPU合同；不调用或初始化Simpler。"""

import os

import pytest

os.environ["VLLM_ASCEND_PTO_CSA_BACKEND"] = "python"  # 旧接口UT各自覆盖模式；本进程不执行算子。
raise SystemExit(
    pytest.main(
        [
            "-q",
            "--noconftest",
            "tests/ut/attention/test_dsa_native_metadata.py",
            "tests/ut/ops/test_csa_aclnn_backend.py",
            "tests/ut/ops/test_csa_aclnn_native_graph.py",
        "tests/ut/ops/test_csa_superkernel_config.py",
        ]
    )
)
