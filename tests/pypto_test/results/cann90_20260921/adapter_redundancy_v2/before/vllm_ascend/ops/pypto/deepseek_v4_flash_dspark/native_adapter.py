# SPDX-License-Identifier: Apache-2.0
"""Prepared Native-storage invocation of the reference-derived CSA chain.

The caller retains Native metadata waits and cache lifecycle hooks. Allocation,
weight preparation and operator registration happen before graph capture.
"""

from dataclasses import dataclass
from typing import Any

import torch

from . import native_metadata
from .decode_csa import decode_csa_tp1_attention_test
from .native_storage import indexer_storage, physical_pages, table_storage


@dataclass(frozen=True)
class CSAOperators:
    attention: Any
    token_metadata: Any
    compressed_metadata: Any
    gather_state: Any
    commit_state: Any

    @classmethod
    def register(cls) -> "CSAOperators":
        import pypto.torch

        def register(kernel, name):
            return pypto.torch.register(kernel, f"dsv4_csa::{name}")

        return cls(
            register(decode_csa_tp1_attention_test, "attention"),
            register(native_metadata.prepare_token_metadata, "token_metadata"),
            register(native_metadata.prepare_compressed_metadata, "compressed_metadata"),
            register(native_metadata.gather_state_window, "gather_state"),
            register(native_metadata.commit_state_window, "commit_state"),
        )


def prepare_weights(attention, hadamard: torch.Tensor) -> dict[str, torch.Tensor]:
    """Prepare the TP1 ABI from already-loaded Native parameters exactly once."""
    import torch_npu

    if attention.compress_ratio != 4 or attention.n_local_heads != 64 or attention.n_local_groups != 8:
        raise ValueError("CSA specialization requires C4 and TP1 with 64 heads / 8 output groups")

    def weight(module, shape, dtype, transpose=False):
        value = module.weight.detach()
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Unexpected loaded weight: {value.shape}/{value.dtype}; expected {shape}/{dtype}")
        if torch_npu.get_npu_format(value) not in (0, 2):
            raise ValueError("CSA weights must already be Native ND after loading")
        if transpose:
            value = value.transpose(-1, -2)
        return value.contiguous()

    def scale(module, width):
        result = module.weight_scale.detach().reshape(-1)
        if result.numel() != width:
            raise ValueError("Unexpected quantized channel-scale count")
        offset = getattr(module, "weight_offset", None)
        if offset is not None and bool(torch.count_nonzero(offset).cpu()):
            raise ValueError("The reference CSA chain requires symmetric INT8 weights")
        return result.float().contiguous()

    bf16, int8 = torch.bfloat16, torch.int8
    main, indexer = attention.compressor, attention.indexer
    inner = indexer.compressor
    return {
        "wq_a": weight(attention.wq_a, (1024, 4096), bf16, True),
        "wq_b": weight(attention.wq_b, (1024, 32768), int8),
        "wq_b_scale": scale(attention.wq_b, 32768),
        "wkv": weight(attention.wkv, (512, 4096), bf16, True),
        "gamma_cq": weight(attention.q_norm, (1024,), bf16),
        "gamma_ckv": weight(attention.kv_norm, (512,), bf16),
        "cmp_wkv": weight(main.wkv, (1024, 4096), bf16),
        "cmp_wgate": weight(main.wgate, (1024, 4096), bf16),
        "cmp_ape": main.ape.detach().float().contiguous(),
        "cmp_norm_w": weight(main.norm, (512,), torch.float32),
        "idx_wq_b": weight(indexer.wq_b, (1024, 8192), int8),
        "idx_wq_b_scale": scale(indexer.wq_b, 8192),
        "weights_proj": weight(indexer.weights_proj, (64, 4096), bf16, True),
        "hadamard_idx": hadamard.detach().T.to(bf16).contiguous(),
        "inner_wkv": weight(inner.wkv, (256, 4096), bf16),
        "inner_wgate": weight(inner.wgate, (256, 4096), bf16),
        "inner_ape": inner.ape.detach().float().contiguous(),
        "inner_norm_w": weight(inner.norm, (128,), torch.float32),
        "attn_sink": attention.attn_sink.detach().contiguous(),
        "wo_a": weight(attention.wo_a, (8, 4096, 1024), bf16, True),
        "wo_b": weight(attention.wo_b, (8192, 4096), int8, True),
        "wo_b_scale": scale(attention.wo_b, 4096),
    }


class NativeCSACall:
    """Fixed-address eager/graph call for uniform six-token target requests.

    Padding/dummy handling and model forward dispatch remain separate gates;
    this initial contract rejects those shapes instead of silently dropping rows.
    """

    def __init__(self, ops, weights, hidden, positions, groups, *, layer_name: str):
        # Each entry contains its own metadata and Native cache views. No shared
        # synthetic page table can stand in for another cache group.
        self.ops = ops
        self.groups = groups
        self.positions = positions
        self.req = {name: value[0].req_metadata for name, value in groups.items()}
        self.views = {name: value[1] for name, value in groups.items()}
        batch = self.req["swa"].seq_lens.numel()
        tokens = hidden.shape[0]
        if batch not in (4, 8, 16, 24, 32, 40) or tokens != batch * 6:
            raise ValueError("Initial CSA call requires an unpadded formal P2 batch with six rows per request")
        if hidden.shape[1] != 4096 or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            raise ValueError("CSA expects contiguous BF16 normalized hidden states [T, 4096]")
        if positions.dtype != torch.int64 or tuple(positions.shape) != (tokens,):
            raise ValueError("CSA expects the Native INT64 target position vector")
        for name, (metadata, _) in groups.items():
            if metadata.num_prefills or metadata.num_actual_tokens != tokens:
                raise ValueError(f"{name}: requires target decode metadata without padding")
            req = self.req[name]
            if req.query_start_loc.numel() != batch + 1 or req.seq_lens.numel() != batch:
                raise ValueError(f"{name}: inconsistent Native request capacity")
            if req.ori_win_right not in (None, 0):
                raise ValueError("Noncausal drafter windows are outside the target CSA contract")

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=hidden.device)

        self.args = dict(weights)
        self.args.update(
            x_normed_t=hidden,
            kv_cache=self.views["swa"][0],
            cmp_kv=self.views["compressed"][0],
            idx_kv_cache=indexer_storage(*self.views["indexer"]),
            cmp_block_table=table_storage(self.req["compressed"].block_table),
            idx_block_table=table_storage(self.req["indexer"].block_table),
            kv_seq_lens=self.req["indexer"].seq_lens,
            position_ids=empty((tokens,), torch.int32),
            window_swa_indices=empty((tokens, 128), torch.int32),
            compress_state=empty((batch * 7, 2, 2048), torch.float32),
            inner_compress_state=empty((batch * 7, 2, 512), torch.float32),
            compress_state_block_table=empty((batch, 7), torch.int32),
            idx_topk_scores=empty((tokens, 512), torch.float32),
            idx_topk=empty((tokens, 512), torch.int32),
            attn_out=empty((tokens, 4096), torch.bfloat16),
        )
        self.args["inner_compress_state_block_table"] = self.args["compress_state_block_table"]
        for name in (
            "ori_slot_mapping",
            "cmp_slot_mapping",
            "idx_slot_mapping",
            "state_slot_mapping",
            "inner_state_slot_mapping",
        ):
            self.args[name] = empty((tokens,), torch.int64)
        for name in ("cmp_freqs", "inner_freqs"):
            for trig in ("cos", "sin"):
                self.args[f"{name}_{trig}"] = empty((tokens, 64), torch.float32)
        self.state_storage = {name: physical_pages(self.views[name][0]) for name in ("state", "indexer_state")}
        self.tables = {name: table_storage(req.block_table) for name, req in self.req.items()}
        for name in ("kv_cache", "cmp_kv"):
            if not self.args[name].is_contiguous() or self.args[name].dtype != torch.bfloat16:
                raise ValueError(f"{name}: Native BF16 32-token pages must have no additional page padding")
        main = self.req["compressed"]
        self.native_cos = main.cos[layer_name][:tokens].view(tokens, 64)
        self.native_sin = main.sin[layer_name][:tokens].view(tokens, 64)
        # All CSA consumers use Native interleaved FP32 frequency columns.
        # Keep the Native buffers and their producer waits; no device conversion.
        self.args["freqs_cos"] = self.native_cos
        self.args["freqs_sin"] = self.native_sin
        self.core_args = tuple(self.args[name] for name in decode_csa_tp1_attention_test.param_names)

    def __call__(self):
        a, r = self.args, self.req
        swa = r["swa"]
        self.ops.token_metadata(
            swa.query_start_loc,
            swa.seq_lens,
            self.positions,
            self.tables["swa"],
            swa.slot_mapping,
            r["state"].slot_mapping,
            r["indexer_state"].slot_mapping,
            a["position_ids"],
            a["ori_slot_mapping"],
            a["state_slot_mapping"],
            a["inner_state_slot_mapping"],
            a["window_swa_indices"],
            a["compress_state_block_table"],
        )
        for name, slot_name, rope_name in (
            ("compressed", "cmp_slot_mapping", "cmp_freqs"),
            ("indexer", "idx_slot_mapping", "inner_freqs"),
        ):
            req = r[name]
            cos, sin, slots = req.compressor_metadata
            self.ops.compressed_metadata(
                req.query_start_loc,
                req.seq_lens,
                self.positions,
                slots,
                cos.view(-1, 64),
                sin.view(-1, 64),
                a[slot_name],
                a[f"{rope_name}_cos"],
                a[f"{rope_name}_sin"],
            )
        for name, state in (("state", "compress_state"), ("indexer_state", "inner_compress_state")):
            req = r[name]
            self.ops.gather_state(
                self.state_storage[name], self.tables[name], req.query_start_loc, req.seq_lens, a[state]
            )
        self.ops.attention(*self.core_args)
        for name, state, slots in (
            ("state", "compress_state", "state_slot_mapping"),
            ("indexer_state", "inner_compress_state", "inner_state_slot_mapping"),
        ):
            req = r[name]
            self.ops.commit_state(
                a[state], req.slot_mapping, a[slots], req.query_start_loc, req.seq_lens, self.state_storage[name]
            )
        return a["attn_out"]
