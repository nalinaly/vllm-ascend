# SPDX-License-Identifier: Apache-2.0
"""Opaque service forward boundary; Native remains the fallback implementation."""

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


@eager_break_during_capture
def dsv4_csa_forward(hidden_states: torch.Tensor, positions: torch.Tensor,
                     output: torch.Tensor, layer_name: str) -> None:
    from vllm_ascend.ops.dsa import _build_kv_cache, dsa_forward

    context = get_forward_context()
    wrapper = context.no_compile_layers[layer_name]
    runtime = getattr(wrapper, "_pto_csa_runtime", None)
    if runtime is None or not runtime.eligible(context, hidden_states, positions):
        dsa_forward(hidden_states, wrapper.need_gather_q_kv, output, layer_name)
        return
    runtime(context, hidden_states, positions, output, _build_kv_cache(wrapper, context))


def dsv4_csa_forward_fake(hidden_states: torch.Tensor, positions: torch.Tensor,
                          output: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="dsv4_csa_forward", op_func=dsv4_csa_forward,
    mutates_args=["output"], fake_impl=dsv4_csa_forward_fake, dispatch_key="PrivateUse1",
)
