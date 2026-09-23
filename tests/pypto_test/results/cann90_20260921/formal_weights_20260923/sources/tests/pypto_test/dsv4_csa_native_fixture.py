"""Real vLLM configuration, distributed groups and metadata for standalone CSA tests.

No engine or full model is loaded. Metadata-only attention buffer owners are
ordinary torch modules; native builders and BlockTable operations are unmodified.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path


def make_config(checkpoint: Path, *, dp_size=1, dp_rank=0, dp_port=None, full_decode_graph=False):
    from dsv4_csa_formal_weights import is_modelslim_checkpoint
    from vllm_ascend.utils import adapt_patch

    # Match CLI pre-registration, including the native INT8 indexer dtype.
    adapt_patch(is_global_patch=True)
    from vllm.config import (
        AttentionConfig,
        CacheConfig,
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    model = ModelConfig(
        model=str(checkpoint),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        quantization="ascend" if is_modelslim_checkpoint(checkpoint) else None,
        max_model_len=1048576,
        hf_overrides={"sliding_window": 128},
    )
    parallel_args = dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        data_parallel_size_local=dp_size,
        data_parallel_rank=dp_rank,
        data_parallel_rank_local=dp_rank,
        enable_expert_parallel=dp_size > 1,
    )
    if dp_size > 1:
        if dp_port is None:
            raise ValueError("DP workers require the same explicit rendezvous port")
        parallel_args.update(
            data_parallel_master_ip="127.0.0.1",
            data_parallel_master_port=dp_port,
            _data_parallel_master_port_list=[dp_port],
        )
    parallel = ParallelConfig(**parallel_args)
    speculative = SpeculativeConfig(
        method="dspark",
        num_speculative_tokens=5,
        target_model_config=model,
        target_parallel_config=parallel,
        enforce_eager=True,
    )
    return VllmConfig(
        model_config=model,
        parallel_config=parallel,
        speculative_config=speculative,
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=1048576,
            max_num_batched_tokens=400,
            max_num_seqs=40,
            async_scheduling=True,
        ),
        cache_config=CacheConfig(block_size=32, enable_prefix_caching=True),
        # Current native A3 QLI uses INT8 K plus FP16 scales. "auto" resolves
        # to activation BF16 here, not to the checkpoint li_cache_scheme.
        attention_config=AttentionConfig(indexer_kv_dtype="int8"),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE if full_decode_graph else CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY if full_decode_graph else CUDAGraphMode.NONE,
        ),
        additional_config={"weight_nz_mode": 0},
    )


@contextmanager
def native_session(config, device_index: int):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_ascend.ascend_config import init_ascend_config

    torch.npu.set_device(device_index)
    with tempfile.TemporaryDirectory(prefix="csa_tp1_") as temporary, set_current_vllm_config(config):
        init_ascend_config(config)
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device_index,
                distributed_init_method=f"file://{temporary}/distributed",
                backend="hccl",
            )
            initialize_model_parallel(backend="hccl")
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def register_rope(config, prefix: str):
    import torch

    from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

    # Native model loading constructs modules under the target device context.
    # RoPE's inv_freq inherits that context, while its positions are explicit NPU.
    with torch.device(f"npu:{torch.npu.current_device()}"):
        return ComplexExpRotaryEmbedding(
            vllm_config=config,
            layername=prefix,
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=65536,
            base=160000,
            scaling_factor=16,
            beta_fast=32,
            beta_slow=1,
            rope_groups=["default", "c4"],
        )


def make_swa_builder(config, device):
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
    from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache

    prefix = "model.layers.2.attn.swa_cache"
    owner = AscendDeepseekV4SWACache(
        head_dim=512,
        window_size=128,
        dtype=config.model_config.dtype,
        prefix=prefix,
        cache_config=config.cache_config,
    )
    spec = owner.get_kv_cache_spec(config)
    builder = AscendDSAMetadataBuilder(spec, [prefix], config, device)
    return owner, spec, builder


def make_attention(config, device):
    """Construct just the real CSA layer, retaining checkpoint quantization."""
    import copy

    import torch

    from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention
    from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
    from vllm_ascend.utils import register_ascend_customop

    register_ascend_customop(config)

    # The checkpoint uses layers.*; the model's live prefixes are model.layers.*.
    # This standalone constructor performs the same prefix mapping explicitly.
    if config.model_config.quantization == "ascend":
        # Use the config loaded by Native VllmConfig from ModelSlim's own
        # description, including its Native DeepSeek prefix mapping.
        quant = config.quant_config
    else:
        description = copy.deepcopy(config.model_config.hf_config.quantization_config)
        description["ignore"] = ["model." + name for name in description["ignore"]]
        quant = AscendCompressedTensorsConfig.from_config(description)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(config.model_config.dtype)
        with torch.device(device):
            return DeepseekV4Attention(
                config,
                config.model_config.hf_config,
                max_position_embeddings=65536,
                cache_config=config.cache_config,
                quant_config=quant,
                prefix="model.layers.2.attn",
            )
    finally:
        torch.set_default_dtype(old_dtype)


def make_cache_groups(config, device, attention):
    """Use all five native owners and their own native specs/builders."""
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

    owners = {
        "swa": attention.dsa_attn.swa_cache_layer,
        "compressed": attention.dsa_attn.dsa_attn,
        "state": attention.compressor.state_cache,
        "indexer": attention.indexer.k_cache,
        "indexer_state": attention.indexer.compressor.state_cache,
    }
    groups = {}
    for kind, owner in owners.items():
        prefix = owner.layer_name if kind == "compressed" else owner.prefix
        spec = owner.get_kv_cache_spec(config)
        builder = AscendDSAMetadataBuilder(spec, [prefix], config, device)
        builder.enable_device_metadata()
        groups[kind] = {"owner": owner, "prefix": prefix, "spec": spec, "builder": builder}
    return groups
