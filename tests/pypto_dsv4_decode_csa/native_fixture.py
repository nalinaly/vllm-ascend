# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Synthetic native/PyPTO fixture for the DeepSeek-V4 decode CSA operator.

This module intentionally lives in ``tests``.  It builds the same TP1 ratio-4
attention module, ModelSlim quant methods, metadata builders and six cache
families used by vLLM, but does not load a checkpoint or construct the rest of
the model.  Heavy vLLM/NPU imports are kept inside the NPU-only constructors so
the physical-layout and ownership contracts remain testable on a Host runner.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from tests.pypto_dsv4_decode_csa.fixtures import (
    _make_strided_tensor,
    _native_swa_slot_mapping,
    _paged_slot_mapping,
    _physical_block_table,
    _position_rows,
    _state_metadata,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.config import BLOCK_SIZE, FLASH

DEFAULT_LAYER_PREFIX = "model.layers.2.self_attn"
STATE_NAMES = (
    "compressed_kv",
    "swa_kv",
    "compressor_state",
    "indexer_compressor_state",
    "indexer_k",
    "indexer_scale",
)


def ratio4_quant_description(prefix: str = DEFAULT_LAYER_PREFIX) -> Mapping[str, str]:
    """Return the exact real quant/FLOAT split required by the 44-argument ABI."""
    description = {
        "model_quant_type": "W8A8_DYNAMIC",
        f"{prefix}.wq_a.weight": "W8A8_DYNAMIC",
        f"{prefix}.wq_b.weight": "W8A8_DYNAMIC",
        f"{prefix}.wkv.weight": "W8A8_DYNAMIC",
        f"{prefix}.indexer.wq_b.weight": "W8A8_DYNAMIC",
        f"{prefix}.compressor.wkv.weight": "FLOAT",
        f"{prefix}.compressor.wgate.weight": "FLOAT",
        f"{prefix}.indexer.weights_proj.weight": "FLOAT",
        f"{prefix}.indexer.compressor.wkv.weight": "FLOAT",
        f"{prefix}.indexer.compressor.wgate.weight": "FLOAT",
        f"{prefix}.wo_a.weight": "FLOAT",
        f"{prefix}.wo_b.weight": "FLOAT",
    }
    return MappingProxyType(description)


def ratio4_quant_description_for_prefixes(prefixes: Sequence[str]) -> Mapping[str, str]:
    """Combine the exact ratio-4 description for independent real layers."""
    values = tuple(prefixes)
    if not values:
        raise ValueError("at least one ratio-4 layer prefix is required")
    if any(not isinstance(prefix, str) or not prefix for prefix in values):
        raise ValueError("every ratio-4 layer prefix must be a non-empty string")
    if len(set(values)) != len(values):
        raise ValueError("ratio-4 layer prefixes must be unique")
    combined = {"model_quant_type": "W8A8_DYNAMIC"}
    for prefix in values:
        for name, method in ratio4_quant_description(prefix).items():
            if name != "model_quant_type":
                combined[name] = method
    return MappingProxyType(combined)


def metadata_family_names(prefix: str = DEFAULT_LAYER_PREFIX) -> tuple[str, ...]:
    """Names in the lexical order consumed by ``ops.dsa.filter_metadata``."""
    names = (
        f"{prefix}.attn",
        f"{prefix}.compressor.state_cache",
        f"{prefix}.indexer.compressor.state_cache",
        f"{prefix}.indexer.k_cache",
        f"{prefix}.swa_cache",
    )
    assert names == tuple(sorted(names))
    return names


@dataclass(frozen=True, slots=True)
class DecodeCSAStateWorld:
    """Strong owner for one independently mutable six-cache universe."""

    spec: DecodeCSAProgramSpec
    raw_cache_tuple: tuple[torch.Tensor, ...]
    storage_owners: tuple[torch.Tensor, ...]
    block_tables: Mapping[str, torch.Tensor]
    positions: torch.Tensor
    start_positions: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    swa_slot_mapping: torch.Tensor
    linear_slot_mappings: Mapping[str, torch.Tensor]

    def snapshot(self, *, device: torch.device | str = "cpu") -> Mapping[str, torch.Tensor]:
        """Copy the six logical states; this is stronger than comparing written slots only."""
        target = torch.device(device)
        return MappingProxyType(
            {
                name: tensor.detach().to(target).clone()
                for name, tensor in zip(STATE_NAMES, self.raw_cache_tuple, strict=True)
            }
        )


def _canonical_starts(spec: DecodeCSAProgramSpec, values: Sequence[int] | None) -> tuple[int, ...]:
    starts = (0,) * spec.batch if values is None else tuple(int(value) for value in values)
    if len(starts) != spec.batch:
        raise ValueError(f"expected {spec.batch} start positions, got {len(starts)}")
    if min(starts) < 0 or max(starts) + spec.seq > FLASH.max_position_embeddings:
        raise ValueError(f"start positions out of range: {starts}")
    return starts


def build_decode_csa_state_world(
    spec: DecodeCSAProgramSpec,
    *,
    device: torch.device | str,
    start_positions: Sequence[int] | None = None,
) -> DecodeCSAStateWorld:
    """Allocate production A3 layouts without creating a PyPTO program or weights."""
    target = torch.device(device)
    starts = _canonical_starts(spec, start_positions)
    physical = spec.physical_layout

    compressed_kv = torch.zeros(
        (spec.compressed_blocks, BLOCK_SIZE, 1, FLASH.head_dim),
        dtype=torch.bfloat16,
        device=target,
    )
    swa_kv = torch.zeros(
        (spec.swa_blocks, BLOCK_SIZE, 1, FLASH.head_dim),
        dtype=torch.bfloat16,
        device=target,
    )
    main_storage, main_state = _make_strided_tensor(
        shape=(spec.main_state_blocks, 2, 1, 4 * FLASH.head_dim),
        strides=(physical.main_state_strides[0], physical.main_state_strides[1], physical.main_state_strides[1], 1),
        dtype=torch.float32,
        device=target,
    )
    inner_storage, inner_state = _make_strided_tensor(
        shape=(spec.inner_state_blocks, 2, 1, 4 * FLASH.index_head_dim),
        strides=(
            physical.inner_state_strides[0],
            physical.inner_state_strides[1],
            physical.inner_state_strides[1],
            1,
        ),
        dtype=torch.float32,
        device=target,
    )
    indexer_k_storage, indexer_k = _make_strided_tensor(
        shape=(spec.indexer_blocks, BLOCK_SIZE, 1, FLASH.index_head_dim),
        strides=physical.indexer_k_strides,
        dtype=torch.int8,
        device=target,
    )
    indexer_scale_storage, indexer_scale = _make_strided_tensor(
        shape=(spec.indexer_blocks, BLOCK_SIZE, 1, 1),
        strides=physical.indexer_scale_strides,
        dtype=torch.float16,
        device=target,
    )
    indexer_scale.fill_(1.0)

    per_request = {
        "compressed": spec.compressed_blocks // spec.batch,
        "swa": spec.swa_blocks // spec.batch,
        "main_state": (spec.main_state_blocks - 1) // spec.batch,
        "inner_state": (spec.inner_state_blocks - 1) // spec.batch,
        "indexer": spec.indexer_blocks // spec.batch,
    }
    if min(per_request.values()) <= 0:
        raise ValueError(f"each request needs a disjoint physical partition, got {per_request}")

    table_specs = {
        "compressed": (spec.compressed_table_width, per_request["compressed"], spec.compressed_blocks),
        "main_state": (spec.main_state_table_width, per_request["main_state"], spec.main_state_blocks),
        "inner_state": (spec.inner_state_table_width, per_request["inner_state"], spec.inner_state_blocks),
        "indexer": (spec.indexer_table_width, per_request["indexer"], spec.indexer_blocks),
        "swa": (spec.swa_table_width, per_request["swa"], spec.swa_blocks),
    }
    host_tables = {
        name: _physical_block_table(
            batch=spec.batch,
            width=width,
            blocks_per_request=blocks_per_request,
            physical_blocks=physical_blocks,
            first_physical_block=1 if name in {"main_state", "inner_state"} else 0,
        )
        for name, (width, blocks_per_request, physical_blocks) in table_specs.items()
    }
    tables = {name: value.to(target) for name, value in host_tables.items()}
    host_positions = _position_rows(starts, seq=spec.seq)
    positions = host_positions.to(target)
    starts_tensor = torch.tensor(starts, dtype=torch.int32, device=target)
    seq_lens = positions[:, -1].contiguous() + 1
    query_start_loc = torch.arange(
        0,
        spec.tokens + 1,
        spec.seq,
        dtype=torch.int32,
        device=target,
    )
    native_swa_slots = _native_swa_slot_mapping(
        positions=host_positions,
        block_table=host_tables["swa"],
    ).to(target)
    linear_slots = {
        "compressed": _paged_slot_mapping(
            positions=host_positions,
            block_table=host_tables["compressed"],
            position_divisor=4,
        ).to(dtype=torch.int32, device=target),
        "main_state": _state_metadata(
            positions=host_positions,
            block_table=host_tables["main_state"],
            state_block_size=2,
        ).to(dtype=torch.int32, device=target),
        "inner_state": _state_metadata(
            positions=host_positions,
            block_table=host_tables["inner_state"],
            state_block_size=2,
        ).to(dtype=torch.int32, device=target),
        "indexer": _paged_slot_mapping(
            positions=host_positions,
            block_table=host_tables["indexer"],
            position_divisor=4,
        ).to(dtype=torch.int32, device=target),
        "swa": (native_swa_slots[:, 0] * BLOCK_SIZE + native_swa_slots[:, 1]).contiguous(),
    }
    return DecodeCSAStateWorld(
        spec=spec,
        raw_cache_tuple=(compressed_kv, swa_kv, main_state, inner_state, indexer_k, indexer_scale),
        storage_owners=(main_storage, inner_storage, indexer_k_storage, indexer_scale_storage),
        block_tables=MappingProxyType(tables),
        positions=positions,
        start_positions=starts_tensor,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        swa_slot_mapping=native_swa_slots,
        linear_slot_mappings=MappingProxyType(linear_slots),
    )


def build_twin_decode_csa_state_worlds(
    spec: DecodeCSAProgramSpec,
    *,
    device: torch.device | str,
    start_positions: Sequence[int] | None = None,
) -> tuple[DecodeCSAStateWorld, DecodeCSAStateWorld]:
    """Return equal initial contents with no shared mutable tensor storage."""
    left = build_decode_csa_state_world(spec, device=device, start_positions=start_positions)
    right = build_decode_csa_state_world(spec, device=device, start_positions=start_positions)
    for name, left_tensor, right_tensor in zip(STATE_NAMES, left.raw_cache_tuple, right.raw_cache_tuple, strict=True):
        if left_tensor.data_ptr() == right_tensor.data_ptr():
            raise AssertionError(f"twin cache {name} unexpectedly aliases storage")
        right_tensor.copy_(left_tensor)
    return left, right


def retarget_decode_csa_state_world(
    world: DecodeCSAStateWorld,
    *,
    start_positions: Sequence[int],
) -> DecodeCSAStateWorld:
    """Reuse one mutable cache world with metadata for the next decode step."""
    spec = world.spec
    starts = _canonical_starts(spec, start_positions)
    target = world.positions.device
    host_positions = _position_rows(starts, seq=spec.seq)
    host_tables = {name: table.detach().cpu() for name, table in world.block_tables.items()}
    native_swa_slots = _native_swa_slot_mapping(
        positions=host_positions,
        block_table=host_tables["swa"],
    ).to(target)
    linear_slots = {
        "compressed": _paged_slot_mapping(
            positions=host_positions,
            block_table=host_tables["compressed"],
            position_divisor=4,
        ).to(dtype=torch.int32, device=target),
        "main_state": _state_metadata(
            positions=host_positions,
            block_table=host_tables["main_state"],
            state_block_size=2,
        ).to(dtype=torch.int32, device=target),
        "inner_state": _state_metadata(
            positions=host_positions,
            block_table=host_tables["inner_state"],
            state_block_size=2,
        ).to(dtype=torch.int32, device=target),
        "indexer": _paged_slot_mapping(
            positions=host_positions,
            block_table=host_tables["indexer"],
            position_divisor=4,
        ).to(dtype=torch.int32, device=target),
        "swa": (native_swa_slots[:, 0] * BLOCK_SIZE + native_swa_slots[:, 1]).contiguous(),
    }
    positions = host_positions.to(target)
    return DecodeCSAStateWorld(
        spec=spec,
        raw_cache_tuple=world.raw_cache_tuple,
        storage_owners=world.storage_owners,
        block_tables=world.block_tables,
        positions=positions,
        start_positions=torch.tensor(starts, dtype=torch.int32, device=target),
        seq_lens=positions[:, -1].contiguous() + 1,
        query_start_loc=world.query_start_loc,
        swa_slot_mapping=native_swa_slots,
        linear_slot_mappings=MappingProxyType(linear_slots),
    )


@dataclass(slots=True)
class SyntheticVllmConfigOwner:
    """Retains the temporary local HF config backing a real ``ModelConfig``."""

    config_dir: tempfile.TemporaryDirectory[str]
    vllm_config: Any
    hf_config: Any
    quant_config: Any

    def close(self) -> None:
        self.config_dir.cleanup()


def _flash_hf_config() -> Any:
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        max_position_embeddings=FLASH.max_position_embeddings,
        rope_theta=FLASH.rope_theta,
        rope_parameters={
            "factor": FLASH.rope_factor,
            "beta_fast": FLASH.beta_fast,
            "beta_slow": FLASH.beta_slow,
            "original_max_position_embeddings": FLASH.original_max_position_embeddings,
            "rope_theta": FLASH.rope_theta,
            "rope_type": "deepseek_yarn",
        },
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=FLASH.hidden_size,
        num_attention_heads=FLASH.num_attention_heads,
        head_dim=FLASH.head_dim,
        qk_rope_head_dim=FLASH.qk_rope_head_dim,
        q_lora_rank=FLASH.q_lora_rank,
        o_lora_rank=FLASH.o_lora_rank,
        o_groups=FLASH.o_groups,
        sliding_window=FLASH.sliding_window,
        rms_norm_eps=FLASH.rms_norm_eps,
        vocab_size=FLASH.vocab_size,
        moe_intermediate_size=FLASH.moe_intermediate_size,
        n_routed_experts=FLASH.n_routed_experts,
        n_shared_experts=FLASH.n_shared_experts,
        num_experts_per_tok=FLASH.num_experts_per_tok,
        scoring_func=FLASH.scoring_func,
        routed_scaling_factor=FLASH.routed_scaling_factor,
        swiglu_limit=FLASH.swiglu_limit,
        num_hidden_layers=FLASH.num_hidden_layers,
        num_hash_layers=FLASH.num_hash_layers,
        num_redundant_experts=0,
        compress_ratios=list(FLASH.compress_ratios),
        index_n_heads=FLASH.index_n_heads,
        index_head_dim=FLASH.index_head_dim,
        index_topk=FLASH.index_topk,
        index_topk_freq=1,
        use_index_cache=False,
        hc_mult=FLASH.hc_mult,
        hc_sinkhorn_iters=FLASH.hc_sinkhorn_iters,
        hc_eps=FLASH.hc_eps,
        compress_rope_theta=FLASH.compress_rope_theta,
        hidden_act="silu",
        norm_topk_prob=False,
        torch_dtype="bfloat16",
    )


def build_synthetic_vllm_config(
    *,
    batch: int = 4,
    prefix: str | Sequence[str] = DEFAULT_LAYER_PREFIX,
) -> SyntheticVllmConfigOwner:
    """Build real vLLM/ModelSlim config objects from a temporary local config."""
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

    config_dir = tempfile.TemporaryDirectory(prefix="pypto-dsv4-csa-")
    hf_config = _flash_hf_config()
    hf_config.save_pretrained(config_dir.name)
    model_config = ModelConfig(
        model=config_dir.name,
        tokenizer=config_dir.name,
        tokenizer_mode="deepseek_v4",
        skip_tokenizer_init=True,
        dtype=torch.bfloat16,
        max_model_len=FLASH.max_position_embeddings,
        enforce_eager=True,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    cache_config.num_gpu_blocks = 512
    cache_config.num_cpu_blocks = 0
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=batch,
        max_num_batched_tokens=batch * 8,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    quant_description = (
        ratio4_quant_description(prefix) if isinstance(prefix, str) else ratio4_quant_description_for_prefixes(prefix)
    )
    quant_config = AscendModelSlimConfig(dict(quant_description))
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(device="npu"),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
        quant_config=quant_config,
        additional_config={"multistream_dsv4_dsa_overlap": False},
    )
    return SyntheticVllmConfigOwner(config_dir, vllm_config, hf_config, quant_config)


def initialize_synthetic_weights(module: torch.nn.Module, *, seed: int = 20260902) -> None:
    """Fill all real parameters before invoking vLLM's post-load finalizer."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "weight_offset" in name:
                parameter.zero_()
            elif "weight_scale" in name:
                parameter.fill_(2.0e-3)
            elif parameter.dtype == torch.int8:
                parameter.random_(-3, 4)
            elif name.endswith("attn_sink"):
                parameter.fill_(4.0)
            elif any(token in name for token in ("q_norm.weight", "kv_norm.weight", ".norm.weight")):
                parameter.fill_(1.0)
            elif name.endswith(".ape"):
                parameter.normal_(mean=0.0, std=2.0e-3)
            elif parameter.is_floating_point():
                parameter.uniform_(-2.0e-3, 2.0e-3)
            else:
                raise TypeError(f"unsupported synthetic parameter {name}: {parameter.dtype}")
        nonzero_offsets = {
            name: int(torch.count_nonzero(parameter).item())
            for name, parameter in module.named_parameters()
            if "weight_offset" in name and int(torch.count_nonzero(parameter).item()) != 0
        }
        if nonzero_offsets:
            raise AssertionError(f"synthetic W8 weights must remain symmetric: {nonzero_offsets}")


def build_real_ratio4_attention(
    config_owner: SyntheticVllmConfigOwner,
    *,
    device: torch.device | str,
    prefix: str = DEFAULT_LAYER_PREFIX,
    seed: int = 20260902,
) -> Any:
    """Construct one real ``DeepseekV4Attention`` and finalize its real methods."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_ascend.models.deepseek_v4 import DeepseekV4Attention
    from vllm_ascend.ops.linear import AscendColumnParallelLinear, AscendUnquantizedLinearMethod
    from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod

    target = torch.device(device)
    # Production model loading constructs parameters under model_config.dtype;
    # explicit FP32 parameters (APE, sinks) keep their declared dtype.
    with (
        set_current_vllm_config(config_owner.vllm_config),
        set_default_torch_dtype(config_owner.vllm_config.model_config.dtype),
        torch.device(target),
    ):
        attention = DeepseekV4Attention(
            vllm_config=config_owner.vllm_config,
            config=config_owner.vllm_config.model_config.hf_config,
            max_position_embeddings=FLASH.max_position_embeddings,
            cache_config=config_owner.vllm_config.cache_config,
            quant_config=config_owner.quant_config,
            prefix=prefix,
        )
        initialize_synthetic_weights(attention, seed=seed)
        # AscendColumnParallelLinear's real checkpoint loader turns wo_a from
        # [groups*o_lora, group_input] into the grouped BMM layout
        # [groups, group_input, o_lora].  Filling Parameter storage directly
        # would bypass this semantically significant load step.
        wo_a = attention.dsa_attn.wo_a
        if not isinstance(wo_a, AscendColumnParallelLinear):
            raise AssertionError(
                "wo_a must be the Ascend production linear so its grouped-layout "
                f"checkpoint loader is exercised, got {type(wo_a).__name__}"
            )
        wo_a.weight_loader(wo_a.weight, wo_a.weight.detach().clone())
        expected_wo_a_shape = (
            FLASH.o_groups,
            FLASH.num_attention_heads * FLASH.head_dim // FLASH.o_groups,
            FLASH.o_lora_rank,
        )
        if tuple(wo_a.weight.shape) != expected_wo_a_shape:
            raise AssertionError(
                f"wo_a production loader must create grouped layout {expected_wo_a_shape}, "
                f"got {tuple(wo_a.weight.shape)}"
            )
        process_weights_after_loading(attention, config_owner.vllm_config.model_config, target)

    wrapper = attention.dsa_attn
    quantized = (wrapper.wq_a, wrapper.wq_b, wrapper.wkv, wrapper.indexer.wq_b)
    for linear in quantized:
        inner = getattr(getattr(linear, "quant_method", None), "quant_method", None)
        if not isinstance(inner, AscendW8A8DynamicLinearMethod):
            raise AssertionError(f"expected a real dynamic-W8 method for {linear.prefix}, got {type(inner).__name__}")
    floating = (
        wrapper.compressor.wkv,
        wrapper.compressor.wgate,
        wrapper.indexer.compressor.wkv,
        wrapper.indexer.compressor.wgate,
        # Although this module passes quant_config=None in the model source,
        # AscendWorker's pluggable-layer registry constructs the Ascend linear
        # and therefore its real unquantized method as well.
        wrapper.indexer.weights_proj,
        wrapper.wo_a,
        wrapper.wo_b,
    )
    for linear in floating:
        method = getattr(linear, "quant_method", None)
        if not isinstance(method, AscendUnquantizedLinearMethod):
            raise AssertionError(f"expected a real FLOAT method for {linear.prefix}, got {type(method).__name__}")
    norm_weights = {
        "q_norm": wrapper.q_norm.weight,
        "kv_norm": wrapper.kv_norm.weight,
        "compressor.norm": wrapper.compressor.norm.weight,
        "indexer.compressor.norm": wrapper.indexer.compressor.norm.weight,
    }
    wrong_norm_dtypes = {name: tensor.dtype for name, tensor in norm_weights.items() if tensor.dtype != torch.bfloat16}
    if wrong_norm_dtypes:
        raise AssertionError(
            f"A3 Flash norm weights must follow the BF16 model construction dtype: {wrong_norm_dtypes}"
        )
    return attention


def bind_state_world(wrapper: Any, world: DecodeCSAStateWorld) -> None:
    """Rebind all six vLLM cache owners to one independent state universe."""
    compressed, swa, main, inner, indexer_k, indexer_scale = world.raw_cache_tuple
    wrapper.dsa_attn.kv_cache = compressed
    wrapper.swa_cache_layer.kv_cache = swa
    wrapper.compressor.state_cache.kv_cache = main
    wrapper.indexer.compressor.state_cache.kv_cache = inner
    wrapper.indexer.k_cache.kv_cache = (indexer_k, indexer_scale)


@dataclass(frozen=True, slots=True)
class NativeMetadataBundle:
    """Five sorted metadata families and every builder/buffer that owns them."""

    metadata: tuple[Any, ...]
    by_name: Mapping[str, Any]
    builders: tuple[Any, ...]
    common_metadata: tuple[Any, ...]


def _common_metadata(world: DecodeCSAStateWorld, *, family: str, block_table: torch.Tensor) -> Any:
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

    spec = world.spec
    starts_cpu = world.start_positions.cpu()
    query_start_cpu = world.query_start_loc.cpu()
    seq_lens_cpu = world.seq_lens.cpu()
    if family == "swa":
        slot_mapping = world.linear_slot_mappings["swa"]
    else:
        slot_mapping = world.linear_slot_mappings[family]
    return AscendCommonAttentionMetadata(
        query_start_loc=world.query_start_loc,
        query_start_loc_cpu=query_start_cpu,
        seq_lens=world.seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=starts_cpu,
        num_computed_tokens_cpu=starts_cpu,
        num_reqs=spec.batch,
        num_actual_tokens=spec.tokens,
        max_query_len=spec.seq,
        max_seq_len=int(max(world.seq_lens.cpu().tolist())),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        positions=world.positions.reshape(-1),
        positions_cpu=world.positions.cpu().reshape(-1),
        # split_decodes_and_prefills combines this with query_start_loc_cpu;
        # production metadata keeps both on Host even when all paging tensors
        # are device resident.
        is_prefilling=torch.zeros(spec.batch, dtype=torch.bool, device="cpu"),
        decode_token_per_req=spec.seq,
        actual_seq_lengths_q=[spec.seq] * spec.batch,
        attn_state=AscendAttentionState.DecodeOnly,
        graph_pad_size=-1,
        num_input_tokens=spec.tokens,
    )


def build_real_native_metadata(
    wrapper: Any,
    world: DecodeCSAStateWorld,
    vllm_config: Any,
) -> NativeMetadataBundle:
    """Run the five real ``AscendDSAMetadataBuilder`` instances on device."""
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

    names = metadata_family_names(wrapper.prefix)
    cache_modules = (
        wrapper.dsa_attn,
        wrapper.compressor.state_cache,
        wrapper.indexer.compressor.state_cache,
        wrapper.indexer.k_cache,
        wrapper.swa_cache_layer,
    )
    table_families = ("compressed", "main_state", "inner_state", "indexer", "swa")
    shared_metadata: dict[str, Any] = {}
    metadata: list[Any] = []
    builders: list[Any] = []
    common_owners: list[Any] = []
    for name, cache_module, family in zip(names, cache_modules, table_families, strict=True):
        cache_spec = cache_module.get_kv_cache_spec(vllm_config)
        common = _common_metadata(world, family=family, block_table=world.block_tables[family])
        builder = AscendDSAMetadataBuilder(cache_spec, [name], vllm_config, world.positions.device)
        # S=8 is a decode bucket here, not an eight-token prefill.
        builder.decode_threshold = world.spec.seq
        built = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            block_size=cache_spec.block_size,
            common_ratio_to_sas_metadata=shared_metadata,
            num_reqs_actual=world.spec.batch,
        )
        metadata.append(built)
        builders.append(builder)
        common_owners.append(common)
    metadata_tuple = tuple(metadata)
    return NativeMetadataBundle(
        metadata=metadata_tuple,
        by_name=MappingProxyType(dict(zip(names, metadata_tuple, strict=True))),
        builders=tuple(builders),
        common_metadata=tuple(common_owners),
    )


__all__ = [
    "DEFAULT_LAYER_PREFIX",
    "STATE_NAMES",
    "DecodeCSAStateWorld",
    "NativeMetadataBundle",
    "SyntheticVllmConfigOwner",
    "bind_state_world",
    "build_decode_csa_state_world",
    "build_real_native_metadata",
    "build_real_ratio4_attention",
    "build_synthetic_vllm_config",
    "build_twin_decode_csa_state_worlds",
    "initialize_synthetic_weights",
    "metadata_family_names",
    "ratio4_quant_description",
    "ratio4_quant_description_for_prefixes",
    "retarget_decode_csa_state_world",
]
