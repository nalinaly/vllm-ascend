# SPDX-License-Identifier: Apache-2.0
"""CSA后端选择与模型准备：CPU检查，不加载wheel或初始化设备运行时。"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.ops.pypto.variant import selected_csa_backend

_TND_PACKAGE = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd"
_COMMON_PACKAGE = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa"


@pytest.mark.parametrize("variant", ["performance", "perf", "spmd_tnd", "mpmd_tnd", "pkg:experiment"])
def test_backend_defaults_to_python_for_existing_variants(monkeypatch, variant):
    monkeypatch.delenv("VLLM_ASCEND_PTO_CSA_BACKEND", raising=False)
    monkeypatch.setenv("PTO_CSA_VARIANT", variant)
    assert selected_csa_backend() == "python"


@pytest.mark.parametrize("backend", ["aclnn", " ACLNN "])
def test_aclnn_backend_selects_only_tnd_spmd(monkeypatch, backend):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", backend)
    monkeypatch.setenv("PTO_CSA_VARIANT", "spmd_tnd")
    assert selected_csa_backend() == "aclnn"


@pytest.mark.parametrize("variant", ["performance", "perf", "mpmd_tnd", "pkg:experiment"])
def test_aclnn_rejects_other_algorithm_variants(monkeypatch, variant):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", "aclnn")
    monkeypatch.setenv("PTO_CSA_VARIANT", variant)
    with pytest.raises(ValueError, match="仅支持PTO_CSA_VARIANT=spmd_tnd"):
        selected_csa_backend()


@pytest.mark.parametrize("backend", ["", "auto", "native", "unknown"])
def test_unknown_backend_does_not_silently_select_python(monkeypatch, backend):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", backend)
    monkeypatch.setenv("PTO_CSA_VARIANT", "spmd_tnd")
    with pytest.raises(ValueError, match="VLLM_ASCEND_PTO_CSA_BACKEND只接受"):
        selected_csa_backend()


@pytest.fixture
def model_preparation(monkeypatch):
    import vllm.config

    import vllm_ascend.ascend_config
    from vllm_ascend.models import pypto_deepseek_v4

    monkeypatch.setenv("PTO_CSA_VARIANT", "spmd_tnd")
    monkeypatch.setenv("PTO_CSA_RUNTIME", "tensormap_and_ringbuffer")
    calls = []

    def fake_module(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class Service:
        def __init__(self, attention, operators, max_num_seqs, layer):
            calls.append("service")
            assert operators is None
            self.attention = attention
            self.max_num_seqs = max_num_seqs
            self.layer = layer

    install_check = Mock(side_effect=lambda: calls.append("wheel_check"))
    runtime_init = Mock(side_effect=lambda: calls.append("python_init"))
    root = object()
    layout = Mock(return_value={"wkv": "NZ"})
    nz_check = Mock()
    fake_module(
        _TND_PACKAGE + ".backend",
        validate_backend_installation=install_check,
        uses_device_policy=Mock(return_value=False),
    )
    fake_module(_TND_PACKAGE + ".service", CSAServiceRuntime=Service)
    fake_module(_TND_PACKAGE + ".entry", decode_csa_spmd=root)
    fake_module(
        _TND_PACKAGE + ".service_config",
        minimum_sequence_length=lambda variant: 1 if variant == "spmd_tnd" else 8192,
        MAX_SEQUENCE_LENGTH=1048576,
    )
    fake_module(_COMMON_PACKAGE + ".nz_mode", root_weight_layouts=layout, validate_weight_nz_mode=nz_check)
    fake_module(_COMMON_PACKAGE + ".reduction", ATOMIC_ADD=0)
    monkeypatch.setattr(pypto_deepseek_v4, "init_pto_runtime", runtime_init)
    monkeypatch.setattr(vllm_ascend.ascend_config, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=2))
    config = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        model_config=SimpleNamespace(max_model_len=131072),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    target_layer = SimpleNamespace(self_attn=SimpleNamespace(compress_ratio=4, dsa_attn=SimpleNamespace()))
    other_layer = SimpleNamespace(self_attn=SimpleNamespace(compress_ratio=128, dsa_attn=SimpleNamespace()))
    model = SimpleNamespace(model=SimpleNamespace(layers=[target_layer, other_layer]))
    return SimpleNamespace(
        run=pypto_deepseek_v4.prepare_csa_model,
        model=model,
        target_layer=target_layer,
        other_layer=other_layer,
        install_check=install_check,
        runtime_init=runtime_init,
        layout=layout,
        nz_check=nz_check,
        root=root,
        calls=calls,
    )


@pytest.mark.parametrize(
    "backend,expected", [("python", ["python_init", "service"]), ("aclnn", ["wheel_check", "service"])]
)
def test_model_preparation_chooses_runtime_without_changing_service_contract(
    monkeypatch, model_preparation, backend, expected
):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", backend)
    prepared = model_preparation
    prepared.run(prepared.model)

    assert prepared.calls == expected
    prepared.layout.assert_called_once_with(prepared.root)
    prepared.nz_check.assert_called_once_with(2)
    service = prepared.target_layer.self_attn.dsa_attn._pto_csa_runtime
    assert service.attention is prepared.target_layer.self_attn
    assert service.layer is prepared.target_layer
    assert service.max_num_seqs == 16
    assert not hasattr(prepared.other_layer.self_attn.dsa_attn, "_pto_csa_runtime")
    assert prepared.model._pto_csa_tnd_capture_sequence_length == 131072


@pytest.mark.parametrize("error", [ModuleNotFoundError("CSA wheel missing"), RuntimeError("CSA ABI mismatch")])
def test_aclnn_installation_failure_stops_before_runtime_or_layer_preparation(monkeypatch, model_preparation, error):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", "aclnn")
    prepared = model_preparation
    prepared.install_check.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        prepared.run(prepared.model)
    prepared.install_check.assert_called_once_with()
    prepared.runtime_init.assert_not_called()
    prepared.layout.assert_not_called()
    assert not prepared.calls
    assert not hasattr(prepared.target_layer.self_attn.dsa_attn, "_pto_csa_runtime")


def test_python_backend_does_not_require_an_aclnn_wheel(monkeypatch, model_preparation):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", "python")
    prepared = model_preparation
    prepared.install_check.side_effect = ModuleNotFoundError("CSA wheel must not be used")
    prepared.run(prepared.model)
    prepared.install_check.assert_not_called()
    prepared.runtime_init.assert_called_once_with()


@pytest.mark.parametrize("backend", ["python", "aclnn"])
def test_tnd_hbg_combination_fails_before_backend_setup(monkeypatch, model_preparation, backend):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", backend)
    monkeypatch.setenv("PTO_CSA_RUNTIME", "host_build_graph")
    prepared = model_preparation
    with pytest.raises(ValueError, match="TND CSA不支持PTO_CSA_RUNTIME=host_build_graph"):
        prepared.run(prepared.model)
    prepared.install_check.assert_not_called()
    prepared.runtime_init.assert_not_called()
    assert not prepared.calls
