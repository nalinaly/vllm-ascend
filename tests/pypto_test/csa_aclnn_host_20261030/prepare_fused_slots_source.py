# SPDX-License-Identifier: Apache-2.0
"""冻结二维slot输出候选；只修改私有测试源码，不切换公共运行路径。"""

import difflib
import json
import shutil
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[2]
base = here / "results/model_decode_source_unused_slots_v1"
target = here / "results/model_decode_source_fused_slots_v1"
shutil.copytree(base, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
changed_files = set()


def edit(name, before, after):
    path = target / name
    text = path.read_text()
    if text.count(before) != 1:
        raise RuntimeError(f"{name}: 预期唯一修改位置，实际为{text.count(before)}")
    updated = text.replace(before, after)
    path.write_text(updated)
    changed_files.add(name)


block = "vllm_ascend/worker/block_table.py"
edit(block, "        self.kernel_sizes = kernel_sizes\n",
     "        self.dsa_slot_mappings: tuple[torch.Tensor, ...] = ()\n\n        self.kernel_sizes = kernel_sizes\n")
signature = '''    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
'''
edit(block, signature, '''    def bind_dsa_slot_mappings(self, outputs: tuple[torch.Tensor, ...]) -> bool:
        """Reuse builders' buffers; validate the supported layout once at init."""
        if self.dcp_world_size != 1 or self.use_hybrid_blocks or not 1 <= len(outputs) <= 2:
            return False
        if any(x.dtype != torch.int32 or x.ndim != 2 or x.shape[1] != 2
               or x.shape[0] < self.max_num_batched_tokens or not x.is_contiguous() for x in outputs):
            return False
        self.dsa_slot_mappings = outputs
        return True

''' + signature)
edit(block, "        num_tokens = positions.shape[0]\n"
     "        total_cp_world_size = self.dcp_world_size\n", '''        num_tokens = positions.shape[0]
        if self.dsa_slot_mappings:
            from vllm_ascend.ops.triton.dsa_slot_mapping import compute_dsa_slot_mapping_kernel

            first = self.dsa_slot_mappings[0]
            second = self.dsa_slot_mappings[-1]
            compute_dsa_slot_mapping_kernel[(num_reqs + 1,)](
                num_tokens, self.max_num_batched_tokens, query_start_loc, positions,
                self.block_table.gpu, self.block_table.gpu.stride(0), self.slot_mapping.gpu,
                first, second, CACHE_BLOCK_SIZE=self.block_size, BLOCK_SIZE=512,
                WRITE_SECOND=len(self.dsa_slot_mappings) == 2,
            )
            return
        total_cp_world_size = self.dcp_world_size
''')

dsa = "vllm_ascend/attention/dsa_v1.py"
edit(dsa, "        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)\n",
     "        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)\n"
     "        self.slot_mapping_precomputed = False\n")
edit(dsa, "        if self.num_prefills or self.compressor_ratio <= 1:\n",
     "        if (self.num_prefills or self.compressor_ratio <= 1) and not kwargs.get(\n"
     "            \"slot_mapping_precomputed\", False\n"
     "        ):\n")

runner = "vllm_ascend/worker/model_runner_v1.py"
edit(runner, "                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,\n",
     "                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,\n"
     "                    slot_mapping_precomputed=True,\n")
edit(runner, "        cascade_attn_prefix_lens: list[list[int]] | None = None,\n",
     "        cascade_attn_prefix_lens: list[list[int]] | None = None,\n"
     "        slot_mapping_precomputed: bool = False,\n")
defer_marker = (
    "                    defer_native_csa_metadata=is_csa_model(self.vllm_config) and (\n"
    "                        not for_cudagraph_capture "
    "or self.vllm_config.additional_config.get(\"_pto_csa_device_policy\", False)\n"
    "                    ),\n"
)
edit(runner, defer_marker, defer_marker + '''                    slot_mapping_precomputed=(
                        slot_mapping_precomputed and not for_cudagraph_capture and ubid is None
                        and type(builder) is AscendDSAMetadataBuilder and builder.slot_mapping_precomputed
                    ),
''')
edit(runner, "        self.may_reinitialize_input_batch(kv_cache_config)\n",
     "        self.may_reinitialize_input_batch(kv_cache_config)\n"
     "        self._bind_dsa_slot_mappings()\n")
edit(runner, "    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:\n",
     '''    def _bind_dsa_slot_mappings(self) -> None:
        # Bind existing metadata buffers once. No allocation or parameter scan
        # is added to a replay or an individual attention layer.
        if not self.use_compress or self.parallel_config.enable_dbo:
            return
        for group_id, groups in enumerate(self.attn_groups):
            builders = [group.get_metadata_builder(0) for group in groups]
            builders = [builder for builder in builders
                        if type(builder) is AscendDSAMetadataBuilder and builder.compressor_ratio <= 1]
            table = self.input_batch.block_table[group_id]
            if table.bind_dsa_slot_mappings(tuple(builder.slot_mapping for builder in builders)):
                for builder in builders:
                    builder.slot_mapping_precomputed = True

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
''')
kernel = "vllm_ascend/ops/triton/dsa_slot_mapping.py"
shutil.copy2(repo / kernel, target / kernel)
changed_files.add(kernel)
edit("tests/pypto_test/offline_pd/worker.py", '            "task_queue_enable": get_task_queue_enable(),\n',
     '            "task_queue_enable": get_task_queue_enable(),\n'
     '            "dsa_fused_slot_outputs_per_group": [\n'
     '                len(table.dsa_slot_mappings) for table in '
     'self.model_runner.input_batch.block_table.block_tables\n'
     '            ],\n')
changes = []
for name in sorted(changed_files):
    original = (base / name).read_text() if (base / name).exists() else ""
    changes.extend(difflib.unified_diff(original.splitlines(True), (target / name).read_text().splitlines(True),
                                      fromfile="a/" + name if original else "/dev/null", tofile="b/" + name))
(here / "fused_slots_integration.patch").write_text("".join(changes))
(target / "source.json").write_text(json.dumps({
    "base": str(base), "source": str(target), "kernel": kernel,
    "patch": str(here / "fused_slots_integration.patch"),
    "scope": "仅私有候选：CP=1、无hybrid block、DSV4的SWA/state槽；原buffer及一维输出保留。"
             "真实execute才声明已计算；dummy/capture/draft及不支持的后端使用既有格式化。",
}, ensure_ascii=False, indent=2) + "\n")
print(target)
