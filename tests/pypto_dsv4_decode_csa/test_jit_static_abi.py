# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only guards for the public PyPTO L1 JIT ABI."""

from __future__ import annotations

import inspect
from typing import get_args

import pypto.language as pl
import pytest
import regex as re
from pypto.language.typing.array import Array
from pypto.language.typing.dynamic import DynVar
from pypto.language.typing.tensor import Tensor

from vllm_ascend.ops._pypto_dsv4_csa import (
    DecodeCSAProgramSpec,
    decode_sparse_attn_csa,
    make_decode_csa_l1_program,
)
from vllm_ascend.ops._pypto_dsv4_csa.backend import (
    EXPECTED_ABI_PARAMETER_COUNT,
    EXPECTED_OUTPUT_INDICES,
    EXPECTED_PARAMETER_NAMES,
    EXPECTED_TENSOR_COUNT,
)

PAGE_STRIDE_SCALARS = (
    "main_state_page_stride",
    "inner_state_page_stride",
    "indexer_k_page_stride",
    "indexer_scale_page_stride",
)


def _dynamic_annotation_dims(function: object) -> list[tuple[str, int, str]]:
    dynamic: list[tuple[str, int, str]] = []
    for parameter in inspect.signature(function).parameters.values():
        annotation = parameter.annotation
        if not isinstance(annotation, Tensor) or annotation.shape is None:
            continue
        for axis, extent in enumerate(annotation.shape):
            if isinstance(extent, DynVar):
                dynamic.append((parameter.name, axis, extent.name))
    return dynamic


def _annotation_contains_array(annotation: object) -> bool:
    return isinstance(annotation, Array) or any(
        _annotation_contains_array(argument) for argument in get_args(annotation)
    )


@pytest.mark.parametrize("batch", (4, 8, 12, 16))
def test_public_entry_is_positive_static_and_page_strides_are_runtime(
    batch: int,
) -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=batch))
    signature = inspect.signature(program._func)

    assert _dynamic_annotation_dims(program._func) == []
    for parameter in signature.parameters.values():
        annotation = parameter.annotation
        if isinstance(annotation, Tensor) and annotation.shape is not None:
            assert all(isinstance(extent, int) and extent > 0 for extent in annotation.shape)
    for name in PAGE_STRIDE_SCALARS:
        assert signature.parameters[name].default is pl.RUNTIME


def test_inline_dependencies_cannot_propagate_dynamic_dims_to_l1_entry() -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=4))
    dependencies, _, _, _ = program._get_dep_graph()

    assert dependencies
    offenders = {
        dependency.__name__: _dynamic_annotation_dims(dependency._func)
        for dependency in dependencies
        if _dynamic_annotation_dims(dependency._func)
    }
    assert offenders == {}


@pytest.mark.parametrize("batch", (4, 8, 12, 16))
def test_public_entry_consumes_request_start_positions_not_flat_positions(
    batch: int,
) -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=batch))
    signature = inspect.signature(program._func)

    assert "position_ids" not in signature.parameters
    start_positions = signature.parameters["start_positions"].annotation
    assert tuple(start_positions.shape) == (batch,)
    assert str(start_positions.dtype) == str(pl.INT32)


def test_public_metadata_abi_matches_native_a3_tensors() -> None:
    spec = DecodeCSAProgramSpec(batch=4)
    program = make_decode_csa_l1_program(spec)
    parameters = inspect.signature(program._func).parameters

    assert tuple(parameters) == EXPECTED_PARAMETER_NAMES
    assert len(parameters) == EXPECTED_ABI_PARAMETER_COUNT == 44
    assert EXPECTED_TENSOR_COUNT == 40
    assert EXPECTED_OUTPUT_INDICES == (39,)
    for removed in (
        "ori_slot_mapping",
        "window_swa_indices",
        "window_swa_lens",
        "cmp_slot_mapping",
        "idx_slot_mapping",
        "state_slot_mapping",
        "inner_state_slot_mapping",
        "swa_slot_mapping",
    ):
        assert removed not in parameters

    expected_tables = {
        "swa_block_table": spec.swa_table_width,
        "cmp_block_table": spec.compressed_table_width,
        "compress_state_block_table": spec.main_state_table_width,
        "inner_compress_state_block_table": spec.inner_state_table_width,
        "idx_block_table": spec.indexer_table_width,
    }
    for name, width in expected_tables.items():
        annotation = parameters[name].annotation
        assert tuple(annotation.shape) == (spec.batch, width)
        assert str(annotation.dtype) == str(pl.INT32)


def test_replay_varying_seq_lens_reaches_every_padded_row_side_effect_child() -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=4))
    dependencies, _, _, _ = program._get_dep_graph()
    by_name = {dependency.__name__: inspect.signature(dependency._func) for dependency in dependencies}

    for child in (
        "compressor_ratio4",
        "indexer",
        "indexer_compressor",
        "sparse_attn_csa_heads",
        "sparse_attn_csa_local_o_proj",
        "sparse_attn_csa",
    ):
        assert "kv_seq_lens" in by_name[child].parameters, child


def test_swa_cache_read_depends_on_writeback_without_persistent_self_copy() -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=4))
    dependencies, _, _, _ = program._get_dep_graph()
    by_name = {dependency.__name__: dependency for dependency in dependencies}

    for child in ("sparse_attn_csa_heads", "sparse_attn_csa"):
        assert "writeback_dep" in inspect.signature(by_name[child]._func).parameters

    heads_source = "".join(inspect.getsource(by_name["sparse_attn_csa_heads"]._func).split())
    entry_source = "".join(inspect.getsource(program._func).split())
    module_source = inspect.getsource(decode_sparse_attn_csa)

    assert "deps=[qk_plan_tid,writeback_dep]" in heads_source
    assert "pl.system.available_cluster_count()" in heads_source
    assert "qk_core_count=pl.tile.get_block_num()" in heads_source
    assert "NUM_QK_CORES" not in module_source
    assert "aswriteback_dep:" in entry_source
    assert "sparse_attn_csa(" in entry_source and "writeback_dep," in entry_source
    assert "kv_touch" not in module_source
    assert "ori_kv_flat[0:T,0:HEAD_DIM]=ori_kv_flat[0:T,0:HEAD_DIM]" not in "".join(module_source.split())


def test_partial_bucket_guards_do_not_use_runtime_continue() -> None:
    from vllm_ascend.ops._pypto_dsv4_csa import (
        decode_compressor_ratio4,
        decode_indexer_compressor,
    )

    for module in (decode_compressor_ratio4, decode_indexer_compressor):
        source = inspect.getsource(module)
        assert re.search(r"(?m)^\s*continue\s*$", source) is None
        assert "if pl.read(kv_seq_lens, [c_idx]) > 0:" in source


def test_scatter_pool_is_one_serial_core_group_task() -> None:
    from vllm_ascend.ops._pypto_dsv4_csa import (
        decode_compressor_ratio4,
        decode_indexer_compressor,
    )

    for module in (decode_compressor_ratio4, decode_indexer_compressor):
        source = "".join(inspect.getsource(module).split())
        assert "forc_idxinpl.range(b_dim):" not in source
        assert 'withpl.at(level=pl.Level.CORE_GROUP,name_hint="scatter_softmax_pool"):' not in source
        assert 'forc_idxinpl.spmd(b_dim,name_hint="scatter_softmax_pool",allow_early_resolve=True,):' in source


def test_indexer_cache_scatter_has_one_aiv_write_owner() -> None:
    from vllm_ascend.ops._pypto_dsv4_csa import decode_indexer_compressor

    source = "".join(inspect.getsource(decode_indexer_compressor).split())
    assert "cache_write_aiv=pl.tile.get_subblock_idx()" in source
    assert "forinnerinpl.range(wr_blk_rows*(1-cache_write_aiv)):" in source


def test_short_context_indexer_fast_path_is_device_gated_and_internal_only() -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=4))
    dependencies, _, _, _ = program._get_dep_graph()
    by_name = {dependency.__name__: dependency for dependency in dependencies}

    entry_source = "".join(inspect.getsource(program._func).split())
    indexer_source = "".join(inspect.getsource(by_name["indexer"]._func).split())
    indexer_signature = inspect.signature(by_name["indexer"]._func)

    # The gate and narrowed top-k scratch stay inside the fused CSA callable;
    # neither changes the public 44-slot production ABI.
    assert tuple(inspect.signature(program._func).parameters) == EXPECTED_PARAMETER_NAMES
    assert "need_index_score=pl.create_tensor([1],dtype=pl.INT32)" in entry_source
    assert "idx_topk_full=pl.create_tensor([tokens,INDEXER_TOPK],dtype=pl.INT32)" in entry_source
    assert "need_index_score" in indexer_signature.parameters
    assert "index_score_gate_dep" in indexer_signature.parameters
    assert "hadamard_idx,rope_swap_idx_t,idx_kv_unused" in entry_source

    # All six remaining score-only child dispatches name the gate producer
    # directly; the two fixed index tables are generated by the existing
    # metadata task rather than an eighth AICore child.  Hadamard matmul and
    # quantization are one mixed child task.
    # top-k remains live and waits for score's dispatched-or-retired TaskId.
    assert indexer_source.count("predicate=(need_index_score[0]>0)") == 6
    assert 'name_hint="qr_rope_swap_idx"' not in indexer_source
    assert "rope_swap_idx_t" in indexer_signature.parameters
    assert "score_reorder_idx_t" not in indexer_signature.parameters
    assert 'name_hint="qr_hadamard_quant_mixed"' in indexer_source
    assert "deps=[_score_tid]" in indexer_source
    assert "ifvisible_len_t<=IDX_TOPK:" in indexer_source
    assert "idx_all=pl.arange(0,[1,IDX_TOPK],dtype=pl.INT32)" in indexer_source


@pytest.mark.parametrize("batch", (4, 8, 12, 16))
def test_indexer_reuses_qkv_swap_storage_and_takes_a_static_32_row_view_at_use(
    batch: int,
) -> None:
    spec = DecodeCSAProgramSpec(batch=batch)
    program = make_decode_csa_l1_program(spec)
    dependencies, _, _, _ = program._get_dep_graph()
    by_name = {dependency.__name__: dependency for dependency in dependencies}

    entry_source = "".join(inspect.getsource(program._func).split())
    qkv_source = "".join(inspect.getsource(by_name["qkv_proj_rope"]._func).split())
    indexer_source = "".join(inspect.getsource(by_name["indexer"]._func).split())
    indexer_annotation = inspect.signature(by_name["indexer"]._func).parameters["rope_swap_idx_t"].annotation

    assert tuple(inspect.signature(program._func).parameters) == EXPECTED_PARAMETER_NAMES
    assert len(inspect.signature(program._func).parameters) == EXPECTED_ABI_PARAMETER_COUNT == 44
    hidden_annotation = inspect.signature(program._func).parameters["hidden_states"].annotation
    assert tuple(hidden_annotation.shape) == (spec.tokens, 4096)
    assert "rope_swap_idx_t=pl.create_tensor([tokens,ROPE_HEAD_DIM],dtype=pl.INT32)" in entry_source
    assert "hadamard_idx,rope_swap_idx_t,idx_kv_unused" in entry_source
    assert indexer_annotation is pl.Tensor
    assert "rope_swap_idx=rope_swap_idx_t[0:QR_FUSED_HEAD_TILE,0:ROPE_HEAD_DIM]" in indexer_source
    assert 'name_hint="idx_qr_proj_dequant"' not in indexer_source
    assert 'name_hint="qr_rope"' not in indexer_source
    assert 'name_hint="idx_qr_dequant_rope"' in indexer_source
    assert "qr_proj=pl.create_tensor" not in indexer_source
    assert "rope_prepare(rope_cos,rope_sin,q_rope_cos_il,q_rope_sin_signed,q_rope_swap_idx)" in qkv_source
    assert 'name_hint="qr_rope_swap_idx"' not in indexer_source


def test_score_reorder_table_is_owned_by_the_matching_predicated_reduce() -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=4))
    dependencies, _, _, _ = program._get_dep_graph()
    by_name = {dependency.__name__: dependency for dependency in dependencies}
    indexer_source = "".join(inspect.getsource(by_name["indexer"]._func).split())

    assert indexer_source.count("score_reorder_idx_t=pl.create_tensor") == 1
    reduce_scope = indexer_source.index('name_hint="weights_proj_reduce"')
    reorder_write = indexer_source.index("score_reorder_idx_t[0:1,0:SCORE_REORDER_SIZE]=")
    score_scope = indexer_source.index('name_hint="score"')
    assert reduce_scope < reorder_write < score_scope
    assert indexer_source.count("predicate=(need_index_score[0]>0)") == 6
    assert "deps=[_qh_quant_tid,_weights_reduce_tid,index_score_gate_dep]" in indexer_source


def test_score_reorder_formula_is_cache_major_to_token_major() -> None:
    import torch

    from vllm_ascend.ops._pypto_dsv4_csa import decode_indexer

    assert decode_indexer.SCORE_LANE_CACHE == 16
    assert decode_indexer.SCORE_LANE_TOKENS == 8
    assert decode_indexer.SCORE_REORDER_SIZE == 128
    j = torch.arange(decode_indexer.SCORE_REORDER_SIZE, dtype=torch.int64)
    actual = (j % decode_indexer.SCORE_LANE_CACHE) * decode_indexer.SCORE_LANE_TOKENS + (
        j // decode_indexer.SCORE_LANE_CACHE
    )
    expected = torch.arange(128, dtype=torch.int64).reshape(16, 8).T.contiguous().flatten()
    assert torch.equal(actual, expected)


def test_qk_plan_balances_optional_items_before_kept_seed_blocks() -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=4))
    dependencies, _, _, _ = program._get_dep_graph()
    by_name = {dependency.__name__: dependency for dependency in dependencies}
    heads_source = "".join(inspect.getsource(by_name["sparse_attn_csa_heads"]._func).split())

    optional_loop = "forplan_sbinpl.unroll(1,SPARSE_BLOCKS):"
    seed_item = "pl.cast(plan_t*SPARSE_BLOCKS,pl.INT32)"
    assert optional_loop in heads_source
    assert "ifpl.read(valid_block_mask,[plan_t,plan_sb])>0:" in heads_source
    assert "qk_plan_all_full" not in heads_source
    assert "full_visible_min=pl.min(full_visible_min,full_visible)" in heads_source
    assert "iffull_visible_min>=IDX_TOPK:" in heads_source
    assert "iffull_visible_min<IDX_TOPK:" in heads_source
    assert "pl.store(full_indices,[bias_t0,0],cmp_sparse_indices)" in heads_source
    assert "pl.assemble(cmp_sparse_indices,c_out_i32,[bias_t0,0])" in heads_source
    assert "full_order_slot=full_plan_t*(SPARSE_BLOCKS-1)+full_plan_sb-1" in heads_source
    assert "qk_wcur=pl.create_tensor([2],dtype=pl.INT32)" in heads_source
    assert "pl.write(qk_wcur,[0],pl.cast(qk_items,pl.INT32))" in heads_source
    assert "pl.write(qk_wcur,[1],pl.cast(1,pl.INT32))" in heads_source
    assert "dtype=pl.FP32,value=0.0" in heads_source
    assert seed_item in heads_source
    assert heads_source.index(optional_loop) < heads_source.index(seed_item)
    assert "pl.write(qk_wcur,[0],pl.cast(0,pl.INT32))" in heads_source
    assert "pl.write(qk_wcur,[1],pl.cast(0,pl.INT32))" in heads_source
    assert "plan_w=pl.read(qk_wcur,[0])" in heads_source
    assert "qk_valid_items=pl.read(qk_wcur,[0])" in heads_source
    assert "qk_all_full=pl.read(qk_wcur,[1])" in heads_source
    assert "qk_block_valid=qk_all_full" in heads_source
    assert "ifqk_all_full<1:qk_block_valid=pl.read(valid_block_mask,[qk_t,qk_sb])" in heads_source
    assert "m_all_full=pl.read(qk_wcur,[1])" in heads_source
    assert "ifm_all_full<1:m_block_valid=pl.read(valid_block_mask,[m_t,m_sb])" in heads_source
    assert "qk_lane_iters=(qk_valid_items-qk_core+qk_core_count-1)//qk_core_count" in heads_source
    assert "emptytilesappended" not in heads_source
    assert "c_blk_valid=pl.row_max(c_mask[:,c_s0:c_s0+ATTN_K_TILE])" in heads_source
    assert "ifqk_cmp_k<CMP_TOPK:" in heads_source
    assert "ifqk_ridx>=0:" in heads_source
    assert "qk_bias_row=pl.full([1,ATTN_K_TILE],dtype=pl.FP32,value=0.0)" in heads_source
    assert "ifqk_all_full<1:qk_bias_row=sparse_bias[" in heads_source
    assert "qk_scores=pl.col_expand_add(qk_scaled,qk_bias_row)" in heads_source


def test_qproj_uses_one_32_row_pass_per_supported_decode_bucket() -> None:
    from vllm_ascend.ops._pypto_dsv4_csa import qkv_proj_rope

    assert qkv_proj_rope.QPROJ_M_TILE == 32
    for batch in (4, 8, 12, 16):
        assert (batch * 8) % qkv_proj_rope.QPROJ_M_TILE == 0


def test_merge_norm_split_keeps_task_id_arrays_function_local() -> None:
    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=4))
    dependencies, _, _, _ = program._get_dep_graph()
    by_name = {dependency.__name__: dependency for dependency in dependencies}
    heads = by_name["sparse_attn_csa_heads"]
    projection = by_name["sparse_attn_csa_local_o_proj"]
    wrapper = by_name["sparse_attn_csa"]

    # ArrayNotEscaped: inline boundaries carry scalar TaskIds only. TaskId
    # arrays remain local orchestration storage on either side of the boundary.
    for dependency in (heads, projection, wrapper):
        signature = inspect.signature(dependency._func)
        assert not _annotation_contains_array(signature.return_annotation)
        assert not any(_annotation_contains_array(parameter.annotation) for parameter in signature.parameters.values())

    merge_head_tiles = decode_sparse_attn_csa.MERGE_HEAD_TILES
    return_types = get_args(inspect.signature(heads._func).return_annotation)
    assert len(return_types) == 1 + merge_head_tiles
    assert return_types[0] is Tensor
    assert all(getattr(annotation, "dtype", None) == pl.TASK_ID for annotation in return_types[1:])

    projection_parameters = inspect.signature(projection._func).parameters
    assert [name for name in projection_parameters if name.startswith("heads_dep_")] == [
        f"heads_dep_{index}" for index in range(merge_head_tiles)
    ]
    assert all(
        projection_parameters[f"heads_dep_{index}"].annotation.dtype == pl.TASK_ID for index in range(merge_head_tiles)
    )

    heads_source = "".join(inspect.getsource(heads._func).split())
    projection_source = "".join(inspect.getsource(projection._func).split())
    wrapper_source = "".join(inspect.getsource(wrapper._func).split())
    assert "merge_tids=pl.array.create(MERGE_HEAD_TILES,pl.TASK_ID)" in heads_source
    assert "form_h_idxinpl.parallel(MERGE_HEAD_TILES):" in heads_source
    assert ('withpl.spmd(t_dim,name_hint="merge_norm",deps=[_qk_tid,rope_cs_tid],)asmerge_tid:') in heads_source
    assert 'name_hint="rope_cs",allow_early_resolve=True,)asrope_cs_tid:' in heads_source
    assert "m_idx=m_t*MERGE_HEAD_TILES+m_h_idx" in heads_source
    assert "m_blk_base=m_idx*SPARSE_BLOCKS*H_TILE" in heads_source
    assert "merge_tids[m_h_idx]=merge_tid" in heads_source
    assert "t_hblocks" not in heads_source
    for index in range(merge_head_tiles):
        assert f"merge_tids[{index}]" in heads_source
        assert f"heads_deps[{index}]=heads_dep_{index}" in projection_source
    assert "deps=[heads_deps[g//GROUPS_PER_MERGE_HEAD_TILE]]" in projection_source
    assert "deps=[heads_deps]" not in projection_source
    assert ("return(o_packed_heads,merge_tids[0],merge_tids[1],merge_tids[2],merge_tids[3],)") in heads_source
    assert (
        "(o_packed_heads,heads_dep_0,heads_dep_1,heads_dep_2,heads_dep_3,)=sparse_attn_csa_heads("
    ) in wrapper_source
    assert (
        "sparse_attn_csa_local_o_proj(o_packed_heads,wo_a,wo_b,attn_out,"
        "kv_seq_lens,heads_dep_0,heads_dep_1,heads_dep_2,heads_dep_3,)"
    ) in wrapper_source


def test_merge_norm_split_preserves_token_major_layout_and_group_ownership() -> None:
    tokens = DecodeCSAProgramSpec(batch=4).tokens
    heads = decode_sparse_attn_csa.H
    head_dim = decode_sparse_attn_csa.HEAD_DIM
    head_tile = decode_sparse_attn_csa.H_TILE
    heads_per_group = decode_sparse_attn_csa.HEADS_PER_GROUP
    groups = decode_sparse_attn_csa.O_GROUPS
    merge_head_tiles = decode_sparse_attn_csa.MERGE_HEAD_TILES
    groups_per_merge = decode_sparse_attn_csa.GROUPS_PER_MERGE_HEAD_TILE

    assert heads % head_tile == 0
    assert head_tile % heads_per_group == 0
    assert merge_head_tiles == heads // head_tile == 4
    assert groups_per_merge == head_tile // heads_per_group == 2
    assert merge_head_tiles * groups_per_merge == groups

    # Task submission becomes head-tile-major, while sparse scratch remains
    # token-major. The new m_idx expression must cover the identical coordinates.
    monolithic = {
        (m_idx // merge_head_tiles, m_idx % merge_head_tiles, m_idx) for m_idx in range(tokens * merge_head_tiles)
    }
    split = {
        (token, tile, token * merge_head_tiles + tile) for tile in range(merge_head_tiles) for token in range(tokens)
    }
    assert split == monolithic

    owners: dict[int, int] = {}
    packed_chunk_starts: set[tuple[int, int]] = set()
    for tile in range(merge_head_tiles):
        owned_groups: set[int] = set()
        for token in range(tokens):
            for local_head in range(head_tile):
                head = tile * head_tile + local_head
                group = head // heads_per_group
                owned_groups.add(group)
                coordinate = (
                    group * tokens + token,
                    (head % heads_per_group) * head_dim,
                )
                assert coordinate not in packed_chunk_starts
                packed_chunk_starts.add(coordinate)
        for group in owned_groups:
            assert group not in owners
            owners[group] = tile

    assert owners == {group: group // groups_per_merge for group in range(groups)}
    assert len(packed_chunk_starts) == tokens * heads


def test_proj_a_b4_s8_uses_one_exact_m32_row_tile() -> None:
    tokens = DecodeCSAProgramSpec(batch=4).tokens
    row_tile = decode_sparse_attn_csa.PROJ_A_ROW_TILE
    n_fragments = decode_sparse_attn_csa.PA_N_FRAGS

    # The measured B4/S8 specialization has exactly 32 token rows.  M32 keeps
    # the same K-fragment accumulation and BF16 cast boundary while halving
    # proj-a blocks and repeated wo_a scans relative to the former M16 layout.
    assert tokens == 32
    assert all(DecodeCSAProgramSpec(batch=batch).tokens % row_tile == 0 for batch in (4, 8, 12, 16))
    assert row_tile == tokens
    assert (tokens + row_tile - 1) // row_tile == 1
    assert n_fragments == 8
    assert decode_sparse_attn_csa.O_GROUPS * n_fragments == 64

    program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=4))
    dependencies, _, _, _ = program._get_dep_graph()
    by_name = {dependency.__name__: dependency for dependency in dependencies}
    projection_source = "".join(inspect.getsource(by_name["sparse_attn_csa_local_o_proj"]._func).split())
    assert "proj_a_rows=(t_dim+PROJ_A_ROW_TILE-1)//PROJ_A_ROW_TILE" in projection_source
    assert "withpl.spmd(proj_a_rows*PA_N_FRAGS" in projection_source
    assert "valid_shape=[pa_rows,A_K_TILE]" in projection_source
    assert 'name_hint="proj_a_bf16"' in projection_source
    assert "o_r_pad=pl.create_tensor([O_GROUPS*t_dim,O_LORA]" in projection_source
    assert "o_r_bf16_pad=pl.create_tensor([O_GROUPS*t_dim,O_LORA]" in projection_source
    assert "pl.assemble(o_r_pad,acc_a,[pa_src0,n0])" in projection_source
    assert "o_r_bf16_pad[row_base_o:row_base_o+t_dim,0:B_K_TILE]" in projection_source
    assert "forg_orderinpl.parallel(O_GROUPS)" in projection_source
    assert "pair_order=g_order//GROUPS_PER_MERGE_HEAD_TILE" in projection_source
    assert "(MERGE_HEAD_TILES-1-pair_order)*GROUPS_PER_MERGE_HEAD_TILE+group_in_pair" in projection_source
    # wo_b keeps the public [D, G*K] ABI but is cold-packed tile-major.  The
    # hot cube path must address its BK-wide physical view; falling back to a
    # logical strided slice silently restores the expensive [8192, 1] GM
    # stride that this optimization removes.
    assert "wo_b_tile_rows=pl.reshape(" in projection_source
    assert "weight_row0=(" in projection_source
    assert "b_weight=wo_b_tile_rows[" in projection_source
    assert "wo_b[n0:n0+PROJ_B_MM_N_TILE" not in projection_source
    assert "g*O_LORA+local_k0" not in projection_source
    assert "partials=pl.create_tensor([t_dim,O_GROUPS*D],dtype=pl.FP32)" in projection_source
    assert 'name_hint="proj_b_zero"' not in projection_source
    assert "deps=[cast_tid]" in projection_source
    assert "atomic=pl.AtomicType.Add" not in projection_source
    assert "partials=pl.assemble(partials,acc_b,[0,g*D+n0]" in projection_source
    assert "foract_ginpl.pipeline(O_GROUPS,stage=2)" in projection_source
    assert "out_col_g" not in projection_source
