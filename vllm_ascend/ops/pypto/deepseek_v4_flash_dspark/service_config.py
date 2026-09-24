# SPDX-License-Identifier: Apache-2.0
"""Host-only selection rules for the opt-in CSA service integration."""

from .config import DECODE_BATCH

MODEL_ARCHITECTURE = "PyptoCSADeepseekV4ForCausalLM"
MAX_BATCH_SIZE = DECODE_BATCH
QUERY_TOKENS = 6


def is_csa_model(config) -> bool:
    return MODEL_ARCHITECTURE in getattr(config.model_config.hf_config, "architectures", ())


def can_replay_csa_graph(*, num_tokens, num_reqs, uniform_decode, padded_tokens) -> bool:
    """补位请求由 kernel 内的 seq_lens 判据屏蔽，因此补位档位也可以重放。

    原先还要求 ``num_tokens == padded_tokens``，即整档不得含补位；那条一旦命中就
    把整步退回 eager 并换用未补齐的 BatchDescriptor，等于把补位本身消掉，
    PTO 因而从未真正走过补位路径。现在 CSA kernel 会按 ``seq_lens == 0`` 跳过
    补位请求的 compact 行推算与页表读取，可以直接重放。

    ``num_tokens``／``num_reqs`` 是本步的**实际**量，``padded_tokens`` 是档位容量。
    判定只用调度器持有的主机计数，绝不读设备张量的值。
    """
    captures_pto = 0 < padded_tokens <= MAX_BATCH_SIZE * QUERY_TOKENS and padded_tokens % QUERY_TOKENS == 0
    return not captures_pto or (
        uniform_decode and num_tokens <= padded_tokens and num_tokens == num_reqs * QUERY_TOKENS
    )


def validate_configuration(config):
    import torch
    from vllm.config import CUDAGraphMode
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.utils import enable_dsa_cp, oproj_tp_enable

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
    # Release Native selects INT8 indexer storage directly on A3; its upstream
    # AttentionConfig does not accept the main-branch indexer_kv_dtype='int8'.
    # NativeCSACall validates the actual key/scale dtype and shared storage.
    if config.cache_config.block_size != 32:
        raise ValueError("PTO CSA requires 32-token cache blocks")
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
        # 这里原本要求 should_skip_allreduce_across_dp_group 为真才允许 DP>1 的图模式，
        # 即只在"DP 补齐不会发生"的配置下放行，否则拒绝启动。那是算子 padding 尚未
        # 完善时的临时保护。现在补位处理已经就位——kernel 按 seq_lens == 0 跳过补位
        # 请求，compact 行号按表的真实行数兜住，三道 host 闸门也已放开——DP 补齐
        # 场景下的 aclgraph 本来就是目标，故移除该拒绝。
