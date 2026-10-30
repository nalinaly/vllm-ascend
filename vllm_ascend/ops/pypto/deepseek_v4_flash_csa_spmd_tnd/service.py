# SPDX-License-Identifier: Apache-2.0
"""TND单SPMD服务：复用Native输入和图生命周期，调用内申请工作区。

沿用Native的cache connector生命周期及同一步跨层compact metadata复用。
与旧多task入口分开，绝不从token数反算请求数。已有Host字段只用于支持范围
和旧后端策略选择；设备选档ACLNN直接读取device Q/L，不做Host长度门槛。

准备对象按根Tensor描述符和策略缓存，不持有外部工作区；capture可替换同描述符
的输入/输出地址。临时Tensor及信号的初始化和生命周期均在PTO调用内部。图重放不会经过Python判定，因此
旧后端保留图合同；统一ACLNN不注册或逐层重算合同，由Native管理图容量与地址。
"""

import torch

from ..deepseek_v4_flash_csa.native_adapter import prepare_weights
from ..deepseek_v4_flash_csa.service import AttentionServiceBase
from ..deepseek_v4_flash_csa.validation import validation_error
from .binding import PreparedDynamicCSA
from .entry import decode_csa_spmd
from .graph_contract import TNDGraphContract, compact_metadata_capacity, graph_compact_request, tensor_descriptor
from .native_adapter import bind_native_inputs, token_capacity_view
from .policy import DeviceSelectedPolicy, select_policy
from .service_config import MAX_QUERY_TOKENS, native_query_bound, supported_descriptor

_TND_COMPACT_METADATA_CACHE = "pto_csa_tnd_compact_compressor_metadata"


def _descriptor_key(native):
    """只用描述符作工作区键；地址可以rebind，Tensor内容不得进入Host控制流。"""
    return tuple(
        (name, tuple(value.shape), value.stride(), value.dtype, value.device) for name, value in sorted(native.items())
    )


def _rope_tensor(request, kind, layer_name):
    """Native uses RopeDataProxy rather than a dict; both expose subscription."""
    mapping = getattr(request, kind, None)
    if mapping is None:
        return None
    try:
        value = mapping[layer_name]
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, torch.Tensor) else None


def _native_token_views(requests, layer_name, tokens):
    """Validate the initialized Native capacity behind shortened decode views."""
    result = []
    for name in ("swa", "state", "indexer_state"):
        slots = requests[name].slot_mapping
        if slots is None or slots.dtype != torch.int32:
            raise ValueError(validation_error("Native slots须为INT32", group=name, slots=slots, tokens=tokens))
        result.append((name + ".slots", token_capacity_view(slots, tokens, 2)))
    for kind in ("cos", "sin"):
        value = _rope_tensor(requests["compressed"], kind, layer_name)
        if value is None or value.dtype != torch.float32:
            raise ValueError(
                validation_error("Native RoPE须为FP32", kind=kind, value=value, layer_name=layer_name, tokens=tokens)
            )
        result.append((kind, token_capacity_view(value, tokens, 64)))
    return result


class CSAServiceRuntime(AttentionServiceBase):
    native_graph_lifecycle = False

    @staticmethod
    def _supported_descriptor(**descriptors):
        return supported_descriptor(**descriptors)

    @staticmethod
    def _prepare_weights(attention, layer):
        return prepare_weights(attention, None, layer, root_function=decode_csa_spmd)

    @staticmethod
    def _select_policy(**descriptors):
        from ..variant import selected_csa_backend

        if selected_csa_backend() == "aclnn":
            from .backend import uses_device_policy
            from .policy import DeviceSelectedPolicy

            if uses_device_policy():
                return DeviceSelectedPolicy()
        return select_policy(**descriptors)

    @staticmethod
    def _prepare_binding(native, **options):
        return PreparedDynamicCSA(native, **options)

    def __init__(self, attention, operators, max_num_seqs, layer=None):
        # operators参数保留模型准备钩子的统一调用约定；本实现不注册多task算子。
        if operators is not None:
            raise ValueError(
                validation_error(
                    "TND CSA直接提交，不接受旧BSH CSAOperators", operators=operators, max_num_seqs=max_num_seqs
                )
            )
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.weights = self._prepare_weights(attention, layer)
        self.prefixes = {
            "swa": self.wrapper.swa_cache_layer.prefix,
            "compressed": self.layer_name,
            "state": attention.compressor.state_cache.prefix,
            "indexer": attention.indexer.k_cache.prefix,
            "indexer_state": attention.indexer.compressor.state_cache.prefix,
        }
        self.batch_capacity = max_num_seqs
        from ..variant import selected_csa_backend

        if selected_csa_backend() == "aclnn":
            from .backend import uses_device_policy

            self.native_graph_lifecycle = uses_device_policy()
        self._device_policy = DeviceSelectedPolicy() if self.native_graph_lifecycle else None
        self._hadamard = None
        self._bindings = {}
        self.submissions = 0
        self.empty_calls = 0
        self.last_binding = None

    def record_graph_route(self, context, hidden, positions):
        """先记下未确认路由；只有实际单SPMD提交路径才能覆盖为可重放合同。"""
        if self.native_graph_lifecycle:
            return
        if getattr(context, "capturing", False):
            context.additional_kwargs.setdefault("pto_csa_tnd_graph_contracts", {})[self.layer_name] = TNDGraphContract(
                self, hidden.shape[0], None
            )

    def _compact_metadata(self, context, req, tokens):
        """沿Native设备生产者计算，固定图容量，同步跨层复用一次结果。"""
        cache = context.additional_kwargs.setdefault(_TND_COMPACT_METADATA_CACHE, {})
        key = (id(req), tokens)
        cached = cache.get(key)
        if cached is not None and cached[0] is req:
            return cached[1]
        adapted = graph_compact_request(req, tokens, validate=not self.native_graph_lifecycle)
        value = self.wrapper.dsa_attn.impl._compute_compressor_metadata(adapted)
        cache[key] = (req, value)
        return value

    def graph_signature(self, context, tokens):
        """不执行metadata生产者或读取Tensor内容，避免旧图绕过真实服务判定。"""
        from vllm_ascend.ops.dsa import _build_kv_cache

        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return None
        descriptors = []
        reference_counts = None
        for name, prefix in self.prefixes.items():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.decode is None:
                return None
            req = item.decode
            requests = req.seq_lens.numel()
            query_bound = native_query_bound(req)
            if not self._supported_descriptor(
                tokens=tokens,
                requests=requests,
                actual_tokens=item.num_actual_tokens,
                max_query_tokens=query_bound,
                max_seq_len=req.max_seq_lens,
                batch_capacity=self.batch_capacity,
            ):
                return None
            if (
                item.num_decodes != requests
                or req.query_start_loc.shape != (requests + 1,)
                or req.seq_lens.dtype != torch.int32
                or req.query_start_loc.dtype != torch.int32
                or req.seq_lens.device != req.query_start_loc.device
            ):
                return None
            if req.num_reqs_actual is not None and not 1 <= req.num_reqs_actual <= requests:
                return None
            counts = (requests, item.num_actual_tokens, query_bound, req.max_seq_lens)
            if reference_counts is not None and counts != reference_counts:
                return None
            reference_counts = counts
            if req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None:
                return None
            descriptors.append(
                (
                    name,
                    tuple(
                        (field, tensor_descriptor(getattr(req, field, None)))
                        for field in (
                            "seq_lens",
                            "query_start_loc",
                            "block_table",
                            "start_pos",
                        )
                    ),
                )
            )
            if name in ("compressed", "indexer"):
                # The producer consumes device Q/L at the captured capacity.
                # Actual request count and shortened token views may change.
                # Slots/RoPE are checked below through initialized capacity views.
                descriptors.append(
                    (
                        name + ".compact_producer",
                        (requests, compact_metadata_capacity(tokens, requests), req.block_size),
                        tensor_descriptor(getattr(req, "full_compress_cos", None)),
                        tensor_descriptor(getattr(req, "full_compress_sin", None)),
                    )
                )
        try:
            views = _native_token_views(
                {name: metadata[prefix].decode for name, prefix in self.prefixes.items()},
                self.layer_name,
                tokens,
            )
        except ValueError:
            return None
        descriptors.extend((name, tensor_descriptor(value)) for name, value in views)
        if metadata[self.prefixes["swa"]].decode.ori_win_left not in (None, 127):
            return None
        indexer = metadata[self.prefixes["indexer"]]
        descriptors.append(("hadamard", tensor_descriptor(indexer.hadamard)))
        kv_cache = _build_kv_cache(self.wrapper, context)
        descriptors.append(("kv_cache", tuple(tensor_descriptor(value) for value in kv_cache)))
        compressed_table = metadata[self.prefixes["compressed"]].decode.block_table
        if compressed_table is None or compressed_table.ndim != 2:
            return None
        policy = self._select_policy(
            tokens=tokens,
            requests=indexer.decode.seq_lens.numel(),
            max_query_tokens=MAX_QUERY_TOKENS,
            max_seq_len=indexer.decode.max_seq_lens,
            compressed_table_columns=compressed_table.shape[1],
        )
        return (tuple(descriptors), policy)

    def handle_empty_call(self, context, hidden, positions, output):
        """空decode不启动全核协议，不访问或更新Native cache/state。"""
        if hidden.numel() == 0:
            output.copy_(hidden)
            self.empty_calls += 1
            return True
        if self.native_graph_lifecycle:
            # 有图容量时仍捕获真实ACLNN，让device Q/L处理全padding。
            # 不能捕获一次Host判空的零输出分支后用于后续有效请求。
            return False
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.num_actual_tokens != 0:
                return False
        # dummy输出不作为有效token使用；保留输入值便于检查无额外状态副作用。
        output.copy_(hidden)
        self.empty_calls += 1
        return True

    def eligible(self, context, hidden, positions):
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        if self.native_graph_lifecycle:
            # 沿Native的prefill/decode路由；长度、padding、策略由device处理。
            # 首次绑定校验ABI；图replay不会再进入逐层描述符扫描。
            item = metadata[self.prefixes["compressed"]]
            return not item.num_prefills and item.decode is not None
        if hidden.ndim != 3 or hidden.shape[1:] != (4, 4096):
            return False
        if hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            return False
        tokens = hidden.shape[0]
        if (
            positions.shape != (tokens,)
            or positions.dtype != torch.int64
            or positions.device != hidden.device
            or not positions.is_contiguous()
        ):
            return False
        reference_counts = None
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.decode is None:
                return False
            req = item.decode
            requests = req.seq_lens.numel()
            query_bound = native_query_bound(req)
            if not self._supported_descriptor(
                tokens=tokens,
                requests=requests,
                actual_tokens=item.num_actual_tokens,
                max_query_tokens=query_bound,
                max_seq_len=req.max_seq_lens,
                batch_capacity=self.batch_capacity,
            ):
                return False
            if item.num_decodes != requests or req.query_start_loc.shape != (requests + 1,):
                return False
            if req.seq_lens.dtype != torch.int32 or req.query_start_loc.dtype != torch.int32:
                return False
            if req.seq_lens.device != hidden.device or req.query_start_loc.device != hidden.device:
                return False
            counts = (requests, item.num_actual_tokens, query_bound, req.max_seq_lens)
            if reference_counts is not None and counts != reference_counts:
                return False
            reference_counts = counts
            if req.num_reqs_actual is not None and not 1 <= req.num_reqs_actual <= requests:
                return False
            if req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None:
                return False
        try:
            _native_token_views(
                {name: metadata[prefix].decode for name, prefix in self.prefixes.items()},
                self.layer_name,
                tokens,
            )
        except ValueError:
            return False
        swa = metadata[self.prefixes["swa"]].decode
        return swa.ori_win_left in (None, 127)

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            notify_kv_cache_written,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        capturing = torch.npu.is_current_stream_capturing()
        hadamard = metadata["indexer"].hadamard
        if hadamard is None:
            raise ValueError(
                validation_error(
                    "Native Indexer未提供Hadamard",
                    layer_name=self.layer_name,
                    hidden=hidden,
                    positions=positions,
                    output=output,
                    capturing=capturing,
                    metadata=metadata,
                )
            )
        if self._hadamard is not hadamard:
            if capturing:
                raise RuntimeError(
                    validation_error(
                        "TND CSA须在捕获前完成Native Hadamard warmup",
                        layer_name=self.layer_name,
                        hadamard=hadamard,
                        cached_hadamard=self._hadamard,
                        capturing=capturing,
                    )
                )
            self.weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
            self._hadamard = hadamard

        wait_for_kv_layer_from_connector(self.layer_name)
        tokens = hidden.shape[0]
        compact = {
            name: self._compact_metadata(context, metadata[name].decode, tokens) for name in ("compressed", "indexer")
        }
        compressed, swa, state, indexer_state, indexer_key, indexer_scale = kv_cache
        groups = {
            "swa": (metadata["swa"], (swa,)),
            "compressed": (metadata["compressed"], (compressed,)),
            "state": (metadata["state"], (state,)),
            "indexer": (metadata["indexer"], (indexer_key, indexer_scale)),
            "indexer_state": (metadata["indexer_state"], (indexer_state,)),
        }
        native = bind_native_inputs(
            self.weights,
            hidden,
            positions,
            groups,
            layer_name=self.layer_name,
            compact_metadata=compact,
            output=output,
            validate=not self.native_graph_lifecycle,
        )
        requests = native["kv_seq_lens"].numel()
        maximum = metadata["indexer"].decode.max_seq_lens
        # 固定的是本服务允许的query上界，真实Q/L仍可在1..6之间逐请求变化。
        # 不因当前max_q较小而把图容量误判为T > B*max_q。
        if self.native_graph_lifecycle:
            policy = self._device_policy
            key = (tokens, requests)
        else:
            policy = self._select_policy(
                tokens=tokens,
                requests=requests,
                max_query_tokens=MAX_QUERY_TOKENS,
                max_seq_len=maximum,
                compressed_table_columns=native["cmp_block_table"].shape[1],
            )
            key = (_descriptor_key(native), policy)
        if key not in self._bindings:
            if capturing:
                raise RuntimeError(
                    validation_error(
                        "TND CSA须在捕获前warmup当前描述符与策略",
                        layer_name=self.layer_name,
                        native=native,
                        tokens=tokens,
                        requests=requests,
                        max_query_tokens=MAX_QUERY_TOKENS,
                        max_seq_len=maximum,
                        policy=policy.constexprs(),
                        prepared_bindings=len(self._bindings),
                    )
                )
            self._bindings[key] = self._prepare_binding(
                native,
                max_query_tokens=MAX_QUERY_TOKENS,
                max_seq_len=maximum,
                policy=policy,
            )
        else:
            if self.native_graph_lifecycle:
                self._bindings[key].rebind(native, validate=False)
            else:
                self._bindings[key].rebind(native)
        prepared = self._bindings[key]
        if capturing and not self.native_graph_lifecycle:
            signature = self.graph_signature(context, tokens)
            if signature is None:
                raise RuntimeError(
                    validation_error(
                        "TND CSA捕获输入未满足可重放合同",
                        layer_name=self.layer_name,
                        hidden=hidden,
                        positions=positions,
                        output=output,
                        capturing=capturing,
                        metadata=metadata,
                    )
                )
            context.additional_kwargs.setdefault("pto_csa_tnd_graph_contracts", {})[self.layer_name] = TNDGraphContract(
                self, tokens, signature
            )
        record_attention_compute_start()
        prepared()
        self.submissions += 1
        self.last_binding = prepared
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
