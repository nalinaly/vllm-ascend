# SPDX-License-Identifier: Apache-2.0
"""Opt-in Native DeepSeek V4 model with a complete PTO CSA decode call."""

from types import MethodType

import torch
from vllm.logger import logger

from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service_config import validate_configuration


def csa_attention_forward(attention, positions, hidden_states, llama_4_scaling):
    # The Native DSA wrapper owns the prefix and all six cache tensors. Keep
    # positions explicit in the custom-op ABI, including during torch.compile.
    output = torch.empty_like(hidden_states)
    torch.ops.vllm.dsv4_csa_forward(hidden_states, positions, output, attention.dsa_attn.prefix)
    return output


def install_csa_forward(attention):
    import vllm_ascend.ops.dsv4_csa  # noqa: F401

    if attention.compress_ratio == 4:
        attention.dsa_attn._pto_csa_runtime = None
        attention.forward = MethodType(csa_attention_forward, attention)


def prepare_csa_model(model):
    # The opt-in runner hook runs after Native per-layer quant finalization.
    # No PyPTO initialization or NPU allocation occurs during model inspection.
    import pypto.torch
    from vllm.config import get_current_vllm_config
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_adapter import CSAOperators
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import CSAServiceRuntime

    pypto.torch.init(device=torch.npu.current_device(), platform="a2a3", runtime="tensormap_and_ringbuffer")
    operators = CSAOperators.register()
    max_num_seqs = get_current_vllm_config().scheduler_config.max_num_seqs
    count = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        if attention.compress_ratio == 4:
            attention.dsa_attn._pto_csa_runtime = CSAServiceRuntime(attention, operators, max_num_seqs)
            count += 1
    if not count:
        raise ValueError("No target C4 attention layers found for PTO CSA")
    logger.info("Prepared PTO CSA for %d target C4 layers; prefill and unsupported decode batches use Native", count)


class PyptoCSADeepseekV4ForCausalLM(AscendDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        validate_configuration(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            install_csa_forward(layer.self_attn)

    def process_weights_after_loading(self):
        prepare_csa_model(self)
