# SPDX-License-Identifier: Apache-2.0
"""CPU ABI and graph-replay contract for the independent CSA HBG entry."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.host_metadata import (
    CSAHostMetadata,
    validate_graph_replay,
)


def test_hbg_takes_native_host_int_without_reading_device():
    native = SimpleNamespace(max_seq_lens=131078, seq_lens=object())
    assert CSAHostMetadata.from_native(native).max_seq_len == 131078
    for value in (True, 1.0, None, torch.tensor(8192)):
        with pytest.raises(TypeError, match="Host int"):
            CSAHostMetadata(value)
    for value in (-1, 2**31):
        with pytest.raises(ValueError, match="INT32"):
            CSAHostMetadata(value)


def test_topology_boundaries_and_model_graph_guard():
    expected = {0: 0, 8191: 0, 8192: 1, 32771: 1, 32772: 2, 131078: 2}
    assert {length: CSAHostMetadata(length).graph_key() for length in expected} == expected
    captured = {"indexer": CSAHostMetadata(8198)}
    current = {"indexer": SimpleNamespace(decode=SimpleNamespace(max_seq_lens=8202))}
    validate_graph_replay(captured, current)
    current["indexer"].decode.max_seq_lens = 8190
    with pytest.raises(ValueError, match="changed Score/Top-K topology"):
        validate_graph_replay(captured, current)
    with pytest.raises(ValueError, match="missing Native decode metadata"):
        validate_graph_replay(captured, {})


def test_separate_torch_schemas_preserve_legacy_tensor_abi():
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.decode_csa import (
        _decode_csa_tp1_layer,
        _decode_csa_tp1_layer_hbg,
    )
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.native_adapter import CSAOperators, HBGCSAOperators

    legacy = CSAOperators.register().attention._schema
    hbg = HBGCSAOperators.register().attention._schema
    assert len(legacy.arguments) == 56
    assert all(str(arg.type) == "Tensor" for arg in legacy.arguments)
    assert [a.name for a in hbg.arguments[:-1]] == [a.name for a in legacy.arguments]
    scalar = hbg.arguments[-1]
    assert scalar.name == "host_max_seq_len" and str(scalar.type) == "int"
    assert not scalar.has_default_value()
    assert inspect.signature(_decode_csa_tp1_layer_hbg).parameters[scalar.name].default is inspect.Parameter.empty
    from dsv4_csa_replay import argument_roles

    assert list(argument_roles(_decode_csa_tp1_layer)) == [a.name for a in legacy.arguments]


def test_model_wrapper_checks_host_values_before_replay(monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    from vllm_ascend.compilation import acl_graph

    descriptor = BatchDescriptor(24)
    replay = []
    entry = acl_graph.ACLGraphEntry(
        descriptor, aclgraph=SimpleNamespace(replay=lambda: replay.append(True)), output="output",
        csa_hbg_host_metadata={"indexer": CSAHostMetadata(8198)},
    )
    context = SimpleNamespace(batch_descriptor=descriptor, cudagraph_runtime_mode=CUDAGraphMode.FULL,
                              attn_metadata={"indexer": SimpleNamespace(decode=SimpleNamespace(max_seq_lens=8190))})
    monkeypatch.setattr(acl_graph, "get_forward_context", lambda: context)
    monkeypatch.setattr(acl_graph, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False))
    wrapper = object.__new__(acl_graph.ACLGraphWrapper)
    wrapper.runtime_mode = CUDAGraphMode.FULL
    wrapper.concrete_aclgraph_entries = {descriptor: entry}
    wrapper.is_debugging_mode = False
    wrapper.enable_enpu = True
    wrapper.use_eagle = False
    with pytest.raises(ValueError, match="changed Score/Top-K topology"):
        wrapper()
    assert not replay
    context.attn_metadata["indexer"].decode.max_seq_lens = 8202
    assert wrapper() == "output"
    assert replay == [True]
