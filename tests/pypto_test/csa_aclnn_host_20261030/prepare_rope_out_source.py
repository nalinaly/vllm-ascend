# SPDX-License-Identifier: Apache-2.0
"""冻结RoPE直接写入已分配图缓冲的私有候选；不改公共路径。"""

import difflib
import json
import shutil
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[2]
base = here / "results/model_decode_source_draft_events_v1"
target = here / "results/model_decode_source_rope_out_v1"
shutil.copytree(base, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
# Bring the already accepted PTO-only activation rules into the new snapshot.
for name in ("vllm_ascend/attention/dsa_v1.py", "vllm_ascend/worker/model_runner_v1.py",
             "tests/pypto_test/offline_pd/worker.py"):
    shutil.copy2(repo / name, target / name)
originals = {}


def edit(name, before, after):
    path = target / name
    text = path.read_text()
    originals.setdefault(name, text)
    if text.count(before) != 1:
        raise RuntimeError(f"{name}: expected one anchor, got {text.count(before)}")
    path.write_text(text.replace(before, after))


rope = "vllm_ascend/ops/rope_dsv4.py"
edit(rope, "    draft_index: int | None = None,\n", "    draft_index: int | None = None,\n"
     "    write_cache_directly: bool = False,\n")
edit(rope, '''            curr_cos = full_rope_cos[pos_tensor]
            curr_sin = full_rope_sin[pos_tensor]

            if use_cache:
''', '''            if use_cache and write_cache_directly:
                group_buffers = (
                    _ROPE_STATE.runtime_buffer.get(config_key, {}).get(group_name)
                    if draft_index is None
                    else _ROPE_STATE.spec_runtime_buffer.get(config_key, {}).get(group_name)
                )
                if group_buffers is None:
                    continue
                buf_cos, buf_sin = group_buffers
                if draft_index is not None:
                    buf_cos, buf_sin = buf_cos[draft_index - 1], buf_sin[draft_index - 1]
                num_tokens = pos_tensor.size(0)
                curr_cos, curr_sin = buf_cos[:num_tokens], buf_sin[:num_tokens]
                # vLLM position IDs (including zero-filled padding) are nonnegative.
                # Standard index_select writes directly into the existing graph inputs.
                torch.index_select(full_rope_cos, 0, pos_tensor, out=curr_cos)
                torch.index_select(full_rope_sin, 0, pos_tensor, out=curr_sin)
                batch_result[config_key][group_name] = (curr_cos, curr_sin)
                continue

            curr_cos = full_rope_cos[pos_tensor]
            curr_sin = full_rope_sin[pos_tensor]

            if use_cache:
''')
dsa = "vllm_ascend/attention/dsa_v1.py"
edit(dsa, "        self.slot_mapping_precomputed = False\n",
     "        self.slot_mapping_precomputed = False\n"
     "        self.write_rope_cache_directly = vllm_config.additional_config.get(\n"
     '            "_pto_csa_device_policy", False\n'
     "        )\n")
edit(dsa, "                cos, sin = get_cos_and_sin_dsa(input_positions, True)\n",
     "                cos, sin = get_cos_and_sin_dsa(\n"
     "                    input_positions, True, write_cache_directly=self.write_rope_cache_directly\n"
     "                )\n")
edit(dsa, "            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True, draft_index=draft_index)\n",
     "            cos, sin = get_cos_and_sin_dsa(\n"
     "                input_positions, use_cache=True, draft_index=draft_index,\n"
     "                write_cache_directly=self.write_rope_cache_directly,\n"
     "            )\n")
edit("tests/pypto_test/offline_pd/worker.py", '            "task_queue_enable": get_task_queue_enable(),\n',
     '            "task_queue_enable": get_task_queue_enable(),\n' + '''            "rope_direct_output": {
                "target": [
                    [getattr(group.get_metadata_builder(0), "write_rope_cache_directly", False) for group in groups]
                    for groups in self.model_runner.attn_groups
                ],
                "draft": [
                    getattr(group.get_metadata_builder(0), "write_rope_cache_directly", False)
                    for group in self.model_runner.drafter.draft_attn_groups
                ],
            },
''')
patch = here / "rope_out_integration.patch"
patch.write_text("".join(
    line for name, original in originals.items()
    for line in difflib.unified_diff(original.splitlines(True), (target / name).read_text().splitlines(True),
                                     fromfile="a/" + name, tofile="b/" + name)
))
(target / "source.json").write_text(json.dumps({
    "base": str(base), "accepted_revision": "74a598843", "source": str(target), "patch": str(patch),
    "scope": "后置PTO目标/草稿metadata直接index_select(out=原RoPE缓冲)，不增加工作区或校验；"
             "输入为vLLM实际非负position IDs，padding为0；Native与非缓存RoPE路径保持原实现。",
}, ensure_ascii=False, indent=2) + "\n")
print(target)
