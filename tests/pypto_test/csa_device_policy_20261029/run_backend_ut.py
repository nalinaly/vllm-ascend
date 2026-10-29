# SPDX-License-Identifier: Apache-2.0
"""公共纯PyTorch bootstrap内运行后端合同UT，不导入vLLM引擎或占卡。"""

import pytest

raise SystemExit(pytest.main(["-q", "--noconftest", "tests/ut/ops/test_csa_aclnn_backend.py"]))
