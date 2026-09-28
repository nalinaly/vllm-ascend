# SPDX-License-Identifier: Apache-2.0
"""显式选择 HCA PTO 的 Native 模型；C4/SWA 层与 FFN 沿用 Native。"""

from types import MethodType

import torch
from vllm.logger import logger

from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM
from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service_config import validate_configuration


def hca_layer_forward(layer, positions, hidden_states, residual, llama_4_scaling=None):
    output = torch.empty_like(hidden_states)
    torch.ops.vllm.dsv4_hca_forward(hidden_states, positions, output, layer.self_attn.dsa_attn.prefix)
    # attention 半层完成后，按 Native 的顺序执行 FFN 半层。
    residual = output.clone()
    hidden_states, post, comb = layer.hc_pre(output, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.hc_post(hidden_states, residual, post, comb)
    return hidden_states, residual


def install_hca_forward(layer):
    import vllm_ascend.ops.dsv4_hca  # noqa: F401

    if layer.self_attn.compress_ratio == 128:
        wrapper = layer.self_attn.dsa_attn
        wrapper._pto_hca_runtime = None
        # tuple 避免将父 layer 再登记成 wrapper 的子模块而造成循环。
        wrapper._pto_hca_layer = (layer,)
        layer.forward = MethodType(hca_layer_forward, layer)


class PyptoHCADeepseekV4ForCausalLM(AscendDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        validate_configuration(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            install_hca_forward(layer)

    def process_weights_after_loading(self):
        import pypto.torch
        from vllm.config import get_current_vllm_config

        from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.native_adapter import HCAOperators
        from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service import HCAServiceRuntime

        pypto.torch.init(device=torch.npu.current_device(), platform="a2a3", runtime="tensormap_and_ringbuffer")
        operators = HCAOperators.register()
        capacity = get_current_vllm_config().scheduler_config.max_num_seqs
        count = 0
        for layer in self.model.layers:
            if layer.self_attn.compress_ratio == 128:
                layer.self_attn.dsa_attn._pto_hca_runtime = HCAServiceRuntime(layer.self_attn, operators, capacity, layer)
                count += 1
        if not count:
            raise ValueError("模型中未找到 C128 HCA 层")
        logger.info("PTO HCA 已绑定 %d 个 C128 层，TP1/S6，公共计算使用 CSA 性能版", count)
