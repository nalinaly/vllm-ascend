# SPDX-License-Identifier: Apache-2.0
"""Host-only selection rules for the opt-in CSA service integration."""

MODEL_ARCHITECTURE = "PyptoCSADeepseekV4ForCausalLM"
SUPPORTED_BATCHES = (1, 4, 8, 16, 24, 32, 40)
QUERY_TOKENS = 6
SUPPORTED_GRAPH_TOKENS = tuple(batch * QUERY_TOKENS for batch in SUPPORTED_BATCHES)


def is_csa_model(config) -> bool:
    return MODEL_ARCHITECTURE in getattr(config.model_config.hf_config, "architectures", ())


def can_replay_csa_graph(*, num_tokens, num_reqs, uniform_decode, padded_tokens) -> bool:
    """A graph captured with PTO has no device-side mask for dummy requests.

    Other buckets capture Native and retain Native's padding support. The
    decision uses scheduler-owned host counts, never device tensor values.
    """
    return padded_tokens not in SUPPORTED_GRAPH_TOKENS or (
        uniform_decode and num_tokens == padded_tokens and num_tokens == num_reqs * QUERY_TOKENS
    )


def validate_configuration(config):
    import torch
    from vllm.config import CUDAGraphMode
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.utils import enable_dsa_cp, oproj_tp_enable, should_skip_allreduce_across_dp_group

    hf = config.model_config.hf_config
    expected = {"hidden_size": 4096, "num_attention_heads": 64, "head_dim": 512,
                "qk_rope_head_dim": 64, "q_lora_rank": 1024, "o_lora_rank": 1024,
                "o_groups": 8, "index_n_heads": 64, "index_head_dim": 128,
                "index_topk": 512, "sliding_window": 128, "rms_norm_eps": 1e-6}
    mismatches = {key: (getattr(hf, key, None), value) for key, value in expected.items()
                  if getattr(hf, key, None) != value}
    if hf.model_type != "deepseek_v4" or mismatches:
        raise ValueError(f"PTO CSA requires the DeepSeek V4 Flash configuration: {mismatches}")
    if config.model_config.dtype != torch.bfloat16 or config.model_config.quantization != "ascend":
        raise ValueError("PTO CSA requires Native ModelSlim W8A8 loading with BF16 activations")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("PTO CSA currently requires TP=1 and PP=1")
    if enable_dsa_cp() or oproj_tp_enable():
        raise ValueError("PTO CSA does not support DSA context parallelism or O-projection TP")
    if config.cache_config.block_size != 32 or config.attention_config.indexer_kv_dtype != "int8":
        raise ValueError("PTO CSA requires 32-token cache blocks and indexer_kv_dtype=int8")
    if get_ascend_config().weight_nz_mode != 0 or get_ascend_config().enable_kv_nz:
        raise ValueError("PTO CSA requires weight_nz_mode=0 and enable_kv_nz=false")
    if getattr(hf, "use_index_cache", False) or config.lora_config is not None:
        raise ValueError("PTO CSA does not support IndexCache reuse or LoRA")
    spec = config.speculative_config
    if spec is None or spec.method != "dspark" or spec.num_speculative_tokens != QUERY_TOKENS - 1:
        raise ValueError("PTO CSA requires target DSpark decoding with five speculative tokens")
    if config.use_v2_model_runner:
        raise ValueError("PTO CSA service graph dispatch currently requires Model Runner V1")
    graph_mode = config.compilation_config.cudagraph_mode
    if graph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        raise ValueError("PTO CSA supports eager or FULL_DECODE_ONLY graph mode")
    if graph_mode != CUDAGraphMode.NONE:
        if config.compilation_config.cudagraph_num_of_warmups < 1:
            raise ValueError("PTO CSA requires a warmup call before each graph capture")
        if parallel.data_parallel_size > 1 and not should_skip_allreduce_across_dp_group(config):
            raise ValueError("PTO CSA full graphs require Native to skip DP padding; use eager until P4 is validated")
