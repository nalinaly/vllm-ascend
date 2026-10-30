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


@pytest.mark.parametrize("requires_replay_sync,enable_enpu", [(None, False), (False, False), (True, True)])
def test_capture_completes_static_kernels_once_without_leaking_metadata_cache(
    monkeypatch, requires_replay_sync, enable_enpu
):
    from contextlib import nullcontext
    from unittest.mock import Mock

    import torch
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    from vllm_ascend.compilation import acl_graph

    context = NS(
        batch_descriptor=BatchDescriptor(num_tokens=96),
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        attn_metadata={},
        additional_kwargs={"num_actual_tokens": 96},
        capturing=False,
    )
    steps = []
    result = torch.empty(1)
    graph = NS(replay=lambda: steps.append("replay"), super_kernel_optimize=lambda: steps.append("optimize"))

    def run():
        # The metadata producer must be recorded again, not reused from warmup.
        assert "compact_metadata" not in context.additional_kwargs
        context.additional_kwargs["compact_metadata"] = object()
        steps.append("capture" if context.capturing else "compile_warmup")
        return result

    compile_call = Mock(side_effect=lambda runnable, args, kwargs: runnable(*args, **kwargs))
    monkeypatch.setattr(acl_graph, "_compile_capture_kernels", compile_call)
    monkeypatch.setattr(acl_graph, "get_forward_context", lambda: context)
    monkeypatch.setattr(acl_graph, "_EXTRA_CTX", NS(is_draft_model=False))
    monkeypatch.setattr(acl_graph, "validate_cudagraph_capturing_enabled", lambda: None)
    monkeypatch.setattr(acl_graph, "weak_ref_tensors", lambda value: value)
    monkeypatch.setattr(acl_graph, "weak_ref_workspaces", lambda _: None)
    monkeypatch.setattr(acl_graph.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(torch.npu, "NPUGraph", lambda: graph)
    monkeypatch.setattr(torch.npu, "graph", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.npu, "current_stream", lambda: NS(synchronize=lambda: steps.append("synchronize")))
    monkeypatch.setattr("vllm.model_executor.offloader.base.get_offloader", lambda: Mock())
    engine = NS(compilation_config=NS(), additional_config={"ascend_compilation_config": {"enable_super_kernel": True}})
    options = {} if requires_replay_sync is None else {"requires_replay_sync": requires_replay_sync}
    wrapper = acl_graph.ACLGraphWrapper(run, engine, CUDAGraphMode.FULL, enable_enpu=enable_enpu, **options)
    wrapper.is_debugging_mode = False
    assert wrapper() is result
    context.capturing = False
    context.additional_kwargs = {"num_actual_tokens": 96}
    assert wrapper() is result
    expected = ["compile_warmup", "capture", "optimize"]
    if requires_replay_sync is not False and not enable_enpu:
        expected.append("synchronize")
    assert steps == [*expected, "replay"]
    compile_call.assert_called_once()
