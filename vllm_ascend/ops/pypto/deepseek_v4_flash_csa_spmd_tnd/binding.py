# SPDX-License-Identifier: Apache-2.0
"""Native TND共享零拷贝绑定；执行可选Python PTO或预编译ACLNN。

本层共享Native输入/输出、描述符和策略，不读回metadata值。
临时工作区与信号由所选后端管理。显式提供Top-K时复用其输出；
ACLNN固定57参数ABI，业务调用所需的两份辅助Top-K输出在准备期分配。
"""

from collections.abc import Mapping

import torch

from vllm_ascend import envs

from ..deepseek_v4_flash_csa.validation import validation_error
from .policy import CSASPMDPolicy, DeviceSelectedPolicy, select_policy

REQUIRED_CUBE_CORES = 24
REQUIRED_VECTOR_CORES = 2 * REQUIRED_CUBE_CORES


def validate_core_geometry(cube_core_num: int, vector_core_num: int) -> None:
    """固定CSA核角色映射要求整卡24C48V；硬件不匹配时在提交前拒绝。"""
    if (cube_core_num, vector_core_num) != (REQUIRED_CUBE_CORES, REQUIRED_VECTOR_CORES):
        raise NotImplementedError(
            validation_error(
                "CSA单SPMD的核内清零/硬同步要求24个AIC和48个AIV；"
                f"当前为{cube_core_num}个AIC和{vector_core_num}个AIV，不启动其他task路径",
                cube_core_num=cube_core_num,
                vector_core_num=vector_core_num,
            )
        )


class PreparedDynamicCSA:
    """准备Native根参数；无外部scratch，调用内由Simpler管理临时内存。"""

    def __init__(
        self,
        native: Mapping[str, torch.Tensor],
        *,
        max_query_tokens: int,
        max_seq_len: int,
        policy: CSASPMDPolicy | DeviceSelectedPolicy | None = None,
    ):
        if envs.VLLM_ASCEND_ENABLE_NZ != 2:
            raise NotImplementedError(
                validation_error(
                    "完整动态SPMD当前先接Native NZ2，其他布局尚未完成整链验证", nz_mode=envs.VLLM_ASCEND_ENABLE_NZ
                )
            )
        self.native = dict(native)
        hidden = native["x_hc"]
        self.tokens = t = hidden.shape[0]
        self.requests = b = native["kv_seq_lens"].numel()
        if hidden.shape != (t, 4, 4096) or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            raise ValueError(validation_error("要求Native BF16连续hidden[T,4,4096]", hidden=hidden))
        if native["cmp_block_table"].ndim != 2:
            raise ValueError(
                validation_error("cmp_block_table must be two-dimensional", cmp_block_table=native["cmp_block_table"])
            )
        from ..variant import selected_csa_backend

        self.execution_backend = selected_csa_backend()
        device_policy = False
        if self.execution_backend == "aclnn":
            from .backend import uses_device_policy

            device_policy = uses_device_policy()
        self.policy = DeviceSelectedPolicy() if device_policy else policy or select_policy(
            tokens=t,
            requests=b,
            max_query_tokens=max_query_tokens,
            max_seq_len=max_seq_len,
            compressed_table_columns=native["cmp_block_table"].shape[1],
        )
        device = hidden.device
        if device.type == "npu":
            import torch_npu

            # 只查询硬件属性，不读取metadata或Tensor值；绑定准备时执行一次。
            properties = torch_npu.npu.get_device_properties(device)
            validate_core_geometry(properties.cube_core_num, properties.vector_core_num)
        self.buffers: dict[str, torch.Tensor] = {}
        # 名称变更只对应接口职责，Tensor保持原对象；同一组内Q/L分别保留。
        direct = {
            "hc_fn": "hc_attn_fn",
            "hc_scale": "hc_attn_scale",
            "hc_base": "hc_attn_base",
            "norm_w": "attn_norm_w",
            "wq_a": "wq_a",
            "gamma_cq": "gamma_cq",
            "wq_b": "wq_b",
            "wq_b_scale": "wq_b_scale",
            "rope_cos": "freqs_cos",
            "rope_sin": "freqs_sin",
            "wkv": "wkv",
            "gamma_ckv": "gamma_ckv",
            "native_slots": "ori_slot_mapping",
            "main_wkv": "cmp_wkv",
            "main_wgate": "cmp_wgate",
            "main_state": "compress_state",
            "main_state_table": "state_block_table",
            "main_ape": "cmp_ape",
            "positions": "position_ids",
            "main_bounds": "cmp_query_start_loc",
            "main_lengths": "cmp_seq_lens",
            "main_state_slots": "state_slot_mapping",
            "main_gamma": "cmp_norm_w",
            "main_cos": "cmp_freqs_cos",
            "main_sin": "cmp_freqs_sin",
            "main_slots": "cmp_slot_mapping",
            "idx_wkv": "inner_wkv",
            "idx_wgate": "inner_wgate",
            "idx_state": "inner_compress_state",
            "idx_state_table": "inner_state_block_table",
            "idx_ape": "inner_ape",
            "idx_bounds": "idx_query_start_loc",
            "idx_lengths": "kv_seq_lens",
            "idx_state_slots": "inner_state_slot_mapping",
            "idx_gamma": "inner_norm_w",
            "idx_cos": "inner_freqs_cos",
            "idx_sin": "inner_freqs_sin",
            "idx_slots": "idx_slot_mapping",
            "hadamard": "hadamard_idx",
            "idx_cache": "idx_native_kv_cache",
            "original_table": "ori_block_table",
            "compressed_table": "cmp_block_table",
            "indexer_table": "idx_block_table",
            "idx_wq_b": "idx_wq_b",
            "idx_wq_b_scale": "idx_wq_b_scale",
            "weights_proj": "weights_proj",
            "topk_scores": "idx_topk_scores",
            "topk_indices": "idx_topk",
            "wo_a": "wo_a",
            "wo_b": "wo_b",
            "wo_b_scale": "wo_b_scale",
        }
        self._diagnostic = "idx_topk" in native
        if device_policy and ("idx_topk" in native or "idx_topk_scores" in native):
            raise NotImplementedError("94档设备选档ACLNN目前提供55 Tensor业务ABI，不暴露诊断Top-K")
        if self._diagnostic != ("idx_topk_scores" in native):
            raise ValueError(
                validation_error(
                    "诊断Top-K索引与分数必须同时提供",
                    has_topk="idx_topk" in native,
                    has_topk_scores="idx_topk_scores" in native,
                )
            )
        if not self._diagnostic:
            direct.pop("topk_indices")
            direct.pop("topk_scores")
        self._direct_names = tuple(direct.items())
        self.buffers.update({target: native[source] for target, source in direct.items()})
        for name in ("main_bounds", "idx_bounds"):
            if self.buffers[name].shape != (b + 1,) or self.buffers[name].dtype != torch.int32:
                raise ValueError(
                    validation_error(
                        f"{name}须为Native INT32[B+1]", parameter=name, tensor=self.buffers[name], requests=b
                    )
                )
        for name in ("main_lengths", "idx_lengths"):
            if self.buffers[name].shape != (b,) or self.buffers[name].dtype != torch.int32:
                raise ValueError(
                    validation_error(
                        f"{name}须为Native INT32[B]", parameter=name, tensor=self.buffers[name], requests=b
                    )
                )
        for name, width in (
            ("norm_w", 4096),
            ("gamma_cq", 1024),
            ("gamma_ckv", 512),
            ("main_gamma", 512),
            ("idx_gamma", 128),
        ):
            value = self.buffers[name]
            if value.shape != (width,) or value.dtype != torch.bfloat16:
                raise ValueError(
                    validation_error(f"{name}必须保持Native BF16[{width}]", parameter=name, tensor=value, width=width)
                )
        for name in ("wq_a", "wq_b", "wo_a", "wo_b", "wkv", "idx_wq_b", "weights_proj"):
            if device.type == "npu":
                import torch_npu

                if torch_npu.get_npu_format(self.buffers[name]) != 29:
                    raise ValueError(
                        validation_error(
                            f"{name}须由Native已有加载流程提供NZ格式",
                            parameter=name,
                            tensor=self.buffers[name],
                            expected_format=29,
                            actual_format=torch_npu.get_npu_format(self.buffers[name]),
                        )
                    )
        for name, value in self.buffers.items():
            if value.device != device or not value.is_contiguous():
                raise ValueError(
                    validation_error(
                        f"{name}须在同一设备并满足连续GM描述符",
                        parameter=name,
                        tensor=value,
                        expected_device=str(device),
                    )
                )

        self.buffers.update(
            x=hidden.view(t, 16384),
            output=native["x_out"].view(t, 16384),
            hc_base_2d=native["hc_attn_base"].view(1, 24),
            cache_rows=native["kv_cache"].view(-1, 512),
            main_cache=native["cmp_kv"].view(-1, 512),
            attn_sink=native["attn_sink"].view(64, 1),
        )
        # Python无外部scratch；ACLNN仅记录原57参数ABI所需的辅助Top-K输出。
        self.allocations: dict[str, torch.Tensor] = {}
        cache = self.buffers["idx_cache"]
        if cache.dtype != torch.int8 or cache.ndim != 2 or cache.shape[1] < 4160 or cache.shape[1] % 64:
            raise ValueError(
                validation_error("Indexer cache须为Native对齐INT8页，页内至少4096B key+64B scale", cache=cache)
            )
        self._executor = None
        if self.execution_backend == "aclnn":
            from .backend import AclnnExecutor

            self._executor = AclnnExecutor(self)

    def rebind(self, native: Mapping[str, torch.Tensor], *, validate=True) -> None:
        """替换同描述符的Native根参数；临时工作区由所选后端管理。"""
        roots = {target: native[source] for target, source in self._direct_names}
        roots.update(
            x=native["x_hc"].view(self.tokens, 16384),
            output=native["x_out"].view(self.tokens, 16384),
            hc_base_2d=native["hc_attn_base"].view(1, 24),
            cache_rows=native["kv_cache"].view(-1, 512),
            main_cache=native["cmp_kv"].view(-1, 512),
            attn_sink=native["attn_sink"].view(64, 1),
        )
        for name, value in (roots.items() if validate else ()):
            previous = self.buffers[name]
            if (
                value.shape != previous.shape
                or value.stride() != previous.stride()
                or value.dtype != previous.dtype
                or value.device != previous.device
            ):
                raise ValueError(
                    validation_error(
                        f"{name}的Native描述符变化，需要重新准备TND绑定",
                        parameter=name,
                        expected=previous,
                        actual=value,
                    )
                )
        if self._executor is not None:
            updated = {**self.buffers, **roots}
            if validate:
                self._executor.rebind(updated)
            else:
                self._executor.rebind(updated, validate=False)
        self.native = dict(native)
        self.buffers.update(roots)

    def __call__(self, *, config=None):
        """复用统一输入合同；按准备时冻结的后端执行。"""
        if self._executor is not None:
            if config is not None:
                raise ValueError("ACLNN 后端不接受 PyPTO RunConfig；请使用 CANN/Torch profiler 采集")
            self._executor()
            return self.native["x_out"]
        from .entry import decode_csa_spmd, decode_csa_spmd_debug

        options = {} if config is None else {"config": config}
        entry = decode_csa_spmd_debug if self._diagnostic else decode_csa_spmd
        entry(**self.buffers, **self.policy.constexprs(), **options)
        return self.native["x_out"]
