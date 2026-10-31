# SPDX-License-Identifier: Apache-2.0
"""在私有源码中复用eager草稿prolog的六个Event；不改变主图或流依赖。"""

import difflib
import json
import shutil
from pathlib import Path

here = Path(__file__).resolve().parent
base = here / "results/model_decode_source_fused_slots_v1"
target = here / "results/model_decode_source_draft_events_v1"
shutil.copytree(base, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
relative = "vllm_ascend/attention/dsa_v1.py"
path = target / relative
original = path.read_text()
text = original
anchor = "        self.vllm_config = get_current_vllm_config()\n"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + "        self._eager_draft_prolog_events = None\n")
begin = text.index("    def _mla_prolog_multistream(")
finish = text.index("\n    def ", begin + 1)
method = text[begin:finish]
anchor = "        is_w8a8 = _is_w8a8_dynamic(self.wq_b)\n"
assert method.count(anchor) == 1
method = method.replace(anchor, '''        # Only eager drafting reuses events. Captured target graphs and prefill
        # keep independent event lifetimes. The record/wait order is unchanged.
        events = (None,) * 6
        speculative = self.vllm_config.speculative_config
        if (not is_prefill and speculative is not None and speculative.enforce_eager
                and getattr(get_forward_context(), "is_draft_model", False)):
            if self._eager_draft_prolog_events is None:
                self._eager_draft_prolog_events = tuple(torch.npu.Event() for _ in range(6))
            events = self._eager_draft_prolog_events

''' + anchor)
for index, name in enumerate(("e_q_quant_done", "e_part2_start", "e_part3_start")):
    old = f"        {name} = main_stream.record_event()"
    assert method.count(old) == 1
    method = method.replace(old, f"        {name} = main_stream.record_event(events[{index * 2}])")
for index in range(3):
    method = method.replace("        main_stream.wait_stream(aux_stream)",
                            f"        main_stream.wait_event(aux_stream.record_event(events[{index * 2 + 1}]))", 1)
assert "main_stream.wait_stream(aux_stream)" not in method
text = text[:begin] + method + text[finish:]
path.write_text(text)
worker_relative = "tests/pypto_test/offline_pd/worker.py"
worker = target / worker_relative
worker_original = worker.read_text()
worker_anchor = '            "task_queue_enable": get_task_queue_enable(),\n'
assert worker_original.count(worker_anchor) == 1
worker_updated = worker_original.replace(worker_anchor, worker_anchor + '''
            "draft_prolog_event_cache": [
                {"module": name, "events": len(events)}
                for name, module in self.model_runner.drafter.model.named_modules()
                if (events := getattr(getattr(module, "impl", None), "_eager_draft_prolog_events", None)) is not None
            ],
'''.lstrip('\n'))
worker.write_text(worker_updated)
patch = here / "draft_events_integration.patch"
patch.write_text("".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                           fromfile="a/" + relative, tofile="b/" + relative)))
with patch.open("a") as handle:
    handle.writelines(difflib.unified_diff(worker_original.splitlines(True), worker_updated.splitlines(True),
                                         fromfile="a/" + worker_relative, tofile="b/" + worker_relative))
(target / "source.json").write_text(json.dumps({
    "base": str(base), "source": str(target), "patch": str(patch),
    "scope": "仅eager MTP三段prolog复用六个Event；主图、prefill保持原Event生命周期。"
             "没有改stream、record/wait顺序、Host队列、MoE或数学计算。",
    "reference": "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_0083.html",
}, ensure_ascii=False, indent=2) + "\n")
print(target)
