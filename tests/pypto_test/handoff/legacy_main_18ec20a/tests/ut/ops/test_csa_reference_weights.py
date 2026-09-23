# SPDX-License-Identifier: Apache-2.0
"""Validate the standalone CSA reference loader's checkpoint normalization."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize("scale_name,scale_shape", [("scale", (4,)), ("weight_scale", (4, 1))])
def test_reference_channel_scale_loading(tmp_path, monkeypatch, scale_name, scale_shape):
    test_root = Path(__file__).resolve().parents[2] / "pypto_test"
    monkeypatch.syspath_prepend(str(test_root))
    from dsv4_csa_native_layout import load_layer_weights

    attention = torch.nn.Module()
    attention.wq_b = torch.nn.Module()
    attention.wq_b.weight = torch.nn.Parameter(torch.empty((4, 2), dtype=torch.int8), requires_grad=False)
    attention.wq_b.weight_scale = torch.nn.Parameter(torch.empty((4, 1)), requires_grad=False)
    attention.wq_b.weight_offset = torch.nn.Parameter(torch.full((4, 1), float("nan")), requires_grad=False)
    weight = torch.arange(8, dtype=torch.int8).reshape(4, 2)
    scale = torch.tensor([0.125, 0.25, 0.5, 1.0]).reshape(scale_shape)
    values = {"layers.2.attn.wq_b.weight": weight, f"layers.2.attn.wq_b.{scale_name}": scale}
    save_file(values, tmp_path / "reference.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "reference.safetensors" for name in values}})
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"config_groups": {"group_0": {"weights": {"symmetric": True}}}}})
    )

    records, methods = load_layer_weights(attention, tmp_path)
    torch.testing.assert_close(attention.wq_b.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(attention.wq_b.weight_scale, scale.reshape(4, 1), rtol=0, atol=0)
    assert torch.count_nonzero(attention.wq_b.weight_offset) == 0
    scale_record = next(record for record in records if record.get("parameter_name") == "wq_b.weight_scale")
    assert scale_record["shape"] == list(scale_shape)
    assert scale_record["loaded_shape"] == [4, 1]
    assert methods == {}
