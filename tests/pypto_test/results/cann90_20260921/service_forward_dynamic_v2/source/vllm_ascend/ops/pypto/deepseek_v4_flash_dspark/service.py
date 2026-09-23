# SPDX-License-Identifier: Apache-2.0
"""Per-layer PTO resources and binding of the current Native service buffers."""

from dataclasses import replace

import torch

from .native_adapter import NativeCSACall, prepare_weights
from .service_config import MAX_BATCH_SIZE, QUERY_TOKENS


class CSAServiceRuntime:
    def __init__(self, attention, operators, max_num_seqs):
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.operators = operators
        self.weights = prepare_weights(attention, None)
        self.prefixes = {
            "swa": self.wrapper.swa_cache_layer.prefix,
            "compressed": self.layer_name,
            "state": attention.compressor.state_cache.prefix,
            "indexer": attention.indexer.k_cache.prefix,
            "indexer_state": attention.indexer.compressor.state_cache.prefix,
        }
        self.batch_capacity = min(max_num_seqs, MAX_BATCH_SIZE)
        tokens = self.batch_capacity * QUERY_TOKENS
        device = attention.wq_a.weight.device
        # Allocate once after weight loading; these buffers are private to one
        # layer. Output belongs to the Native forward call / captured graph.
        self.scores = torch.empty((tokens, 512), dtype=torch.float32, device=device)
        self.topk = torch.empty((tokens, 512), dtype=torch.int32, device=device)
        self._hadamard = None

    def eligible(self, context, hidden, positions):
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        if hidden.ndim != 2 or hidden.shape[1] != 4096 or hidden.dtype != torch.bfloat16:
            return False
        tokens = hidden.shape[0]
        if tokens % QUERY_TOKENS or not 1 <= tokens // QUERY_TOKENS <= self.batch_capacity:
            return False
        if not hidden.is_contiguous() or positions.dtype != torch.int64 or positions.shape != (tokens,):
            return False
        batch = tokens // QUERY_TOKENS
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.num_actual_tokens != tokens:
                return False
            if item.max_query_len != QUERY_TOKENS or item.num_decodes != batch:
                return False
            req = item.req_metadata
            if req is None or req.seq_lens.numel() != batch or req.query_start_loc.numel() != batch + 1:
                return False
            if req.num_actual_reqs not in (None, batch) or req.ori_win_right not in (None, 0):
                return False
            if req.vision_swa_indices is not None or req.dspark_swa_indices is not None:
                return False
        swa = metadata[self.prefixes["swa"]].req_metadata
        return swa.ori_win_left in (None, 127)

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector, notify_kv_cache_written, wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
            record_attention_compute_start,
        )
        from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata

        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        # Native creates this Hadamard table when it builds attention metadata.
        # Transform once in an ordinary warmup call, never during capture/replay.
        hadamard = metadata["indexer"].hadamard
        if hadamard is None:
            raise ValueError("Native indexer metadata did not provide the Hadamard matrix")
        if self._hadamard is not hadamard:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("PTO CSA requires Native metadata warmup before graph capture")
            self.weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
            self._hadamard = hadamard

        wait_for_kv_layer_from_connector(self.layer_name)
        for name, compressor in (("compressed", self.wrapper.compressor),
                                 ("indexer", self.wrapper.indexer.compressor)):
            # Reuse Native's producer and its COMPRESSOR event wait, including
            # the eager metadata path when no executor output was precomputed.
            item = metadata[name]
            compact = compressor._compute_metadata(item.req_metadata)
            metadata[name] = replace(item, req_metadata=replace(item.req_metadata, compressor_metadata=compact))
        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(metadata["indexer"].req_metadata.qli_metadata))
        wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(metadata["compressed"].req_metadata.sas_metadata))
        compressed, swa, state, indexer_state, indexer_key, indexer_scale = kv_cache
        groups = {
            "swa": (metadata["swa"], (swa,)),
            "compressed": (metadata["compressed"], (compressed,)),
            "state": (metadata["state"], (state,)),
            "indexer": (metadata["indexer"], (indexer_key, indexer_scale)),
            "indexer_state": (metadata["indexer_state"], (indexer_state,)),
        }
        tokens = hidden.shape[0]
        call = NativeCSACall(
            self.operators, self.weights, hidden, positions, groups, layer_name=self.layer_name,
            buffers={"idx_topk_scores": self.scores[:tokens], "idx_topk": self.topk[:tokens], "attn_out": output},
        )
        # A single fused call publishes all KV writes. Notify the connector on
        # the same stream after that call; no global synchronization is needed.
        record_attention_compute_start()
        call()
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
