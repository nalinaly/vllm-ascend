# SPDX-License-Identifier: Apache-2.0
"""Native/PTO共用SK编译配置，默认行为不变。"""

from types import SimpleNamespace as NS

import pytest

from vllm_ascend.ascend_config import AscendCompilationConfig
from vllm_ascend.compilation.compiler_interface import _configure_backend


def test_superkernel_requires_static_and_propagates_to_both_compilers(monkeypatch):
    monkeypatch.setattr("vllm_ascend.utils.is_310p", lambda: False)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="enable_static_kernel"):
        AscendCompilationConfig(enable_super_kernel=True)
    config = AscendCompilationConfig(enable_static_kernel=True, enable_super_kernel=True)
    engine = NS(
        speculative_config=NS(num_speculative_tokens=5),
        scheduler_config=NS(max_num_seqs=16),
        compilation_config=NS(cudagraph_capture_sizes=[6, 96]),
    )
    received = {}
    _configure_backend(NS(), config, engine, lambda _, kwargs: received.update(kwargs["options"]))
    assert received["static_kernel_compile"] and received["super_kernel_optimize"] and received["inplace_pass"]
    assert received["_vllm_aclnn_static_kernel_sym_range"] == [6, 96]
    fallback = NS(debug=NS(aclgraph=NS()), experimental_config=NS(aclgraph=NS()))
    _configure_backend(fallback, config, engine)
    assert fallback.experimental_config.aclgraph._super_kernel_optimize
    assert fallback.experimental_config.aclgraph._aclnn_static_shape_kernel
    assert fallback.debug.aclgraph.disable_reinplace_inplaceable_ops_pass is False
    assert AscendCompilationConfig().enable_super_kernel is False


def test_superkernel_does_not_reinterpret_existing_positional_arguments(monkeypatch):
    monkeypatch.setattr("vllm_ascend.utils.is_310p", lambda: False)
    config = AscendCompilationConfig(True, False, True, False)
    assert config.fuse_norm_quant and not config.fuse_qknorm_rope
    assert not config.enable_super_kernel
