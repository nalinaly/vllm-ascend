"""可选的forward入场诊断；仅主机时钟和GC回调，不同步或修改设备工作。"""

import gc
import threading
import time
from contextlib import contextmanager


class ForwardHostDiagnostics:
    def __init__(self):
        self.events = []
        self.gc_enabled = gc.isenabled()
        self.gc_threshold = gc.get_threshold()
        self.gc_freeze_count = gc.get_freeze_count()
        self.runner = None
        self.original_methods = []
        self.metadata_entry = None
        self.dp_sync_entry = None
        self.metadata_builders = []
        self.callback = self._gc_event
        gc.callbacks.append(self.callback)

    def _gc_event(self, phase, info):
        # Do not scan objects or log in the hot path: either could cause stalls.
        self.events.append({"phase": phase, "generation": info["generation"],
                            "monotonic_ns": time.monotonic_ns(), "thread_id": threading.get_native_id(),
                            "collected": info.get("collected", 0)})

    @staticmethod
    def mark(entry, name):
        entry.setdefault("host", {})[name] = {
            "monotonic_ns": time.monotonic_ns(), "thread_cpu_ns": time.thread_time_ns()}

    def attach_runner(self, runner, active_entry):
        """Separate existing waits from preparation without adding any sync."""
        self.runner = runner

        def install(name, wrapped, original, owner=runner):
            self.original_methods.append((owner, name, name in vars(owner), original))
            setattr(owner, name, wrapped)

        def timed(original, label, metadata_only=False):
            def call(*args, **kwargs):
                entry = self.metadata_entry if metadata_only else active_entry()
                if entry is None:
                    return original(*args, **kwargs)
                previous_metadata_entry = self.metadata_entry
                if label == "attention_metadata":
                    self.metadata_entry = entry
                call_label = label
                if metadata_only:
                    # Keep repeated calls rather than overwriting the slow one.
                    occurrence = 1
                    while f"{call_label}_begin" in entry.get("host", {}):
                        occurrence += 1
                        call_label = f"{label}_call{occurrence}"
                self.mark(entry, f"{call_label}_begin")
                try:
                    return original(*args, **kwargs)
                finally:
                    self.mark(entry, f"{call_label}_end")
                    if label == "attention_metadata":
                        self.metadata_entry = previous_metadata_entry
            return call

        for name, label in (
            ("_prepare_inputs", "inputs"),
            ("_determine_batch_execution_and_padding", "batch_coordination"),
            ("_build_attention_metadata", "attention_metadata"),
            ("_preprocess", "preprocess"),
        ):
            original = getattr(runner, name, None)
            if callable(original):
                install(name, timed(original, label), original)

        original_dp_sync = getattr(runner, "_sync_metadata_across_dp", None)
        if callable(original_dp_sync):
            import torch.distributed as dist

            original_all_reduce = dist.all_reduce

            def all_reduce(*args, **kwargs):
                entry = self.dp_sync_entry
                if entry is None:
                    return original_all_reduce(*args, **kwargs)
                self.mark(entry, "dp_all_reduce_begin")
                try:
                    return original_all_reduce(*args, **kwargs)
                finally:
                    self.mark(entry, "dp_all_reduce_end")

            def dp_sync(*args, **kwargs):
                draft = kwargs.get("is_draft_model", args[1] if len(args) > 1 else False)
                entry = None if draft else active_entry()
                if entry is None:
                    return original_dp_sync(*args, **kwargs)
                previous = self.dp_sync_entry
                self.dp_sync_entry = entry
                self.mark(entry, "dp_sync_begin")
                try:
                    return original_dp_sync(*args, **kwargs)
                finally:
                    self.mark(entry, "dp_sync_end")
                    self.dp_sync_entry = previous

            install("all_reduce", all_reduce, original_all_reduce, dist)
            install("_sync_metadata_across_dp", dp_sync, original_dp_sync)

        # The decode harness uses builder 0 (no microbatching). Resolve builders
        # once when attaching, outside measured steps; never inspect tensors here.
        seen_builders = {}
        for cache_group, groups in enumerate(getattr(runner, "attn_groups", ())):
            for attention_group, group in enumerate(groups):
                builder = group.get_metadata_builder(0)
                location = {"cache_group": cache_group, "attention_group": attention_group,
                            "layer_names": list(group.layer_names)}
                if id(builder) in seen_builders:
                    seen_builders[id(builder)]["groups"].append(location)
                    continue
                label = f"metadata_builder_g{cache_group}_a{attention_group}"
                description = {"label": label, "type": type(builder).__name__, "groups": [location]}
                seen_builders[id(builder)] = description
                self.metadata_builders.append(description)
                for name in ("build", "build_decode_metadata", "build_prefill_metadata"):
                    original = getattr(builder, name, None)
                    if callable(original):
                        install(name, timed(original, f"{label}_{name}", metadata_only=True), original, builder)

        original_sync = getattr(runner, "synchronize_input_prep", None)
        if callable(original_sync):
            @contextmanager
            def input_prep(*args, **kwargs):
                entry = active_entry()
                if entry is not None:
                    self.mark(entry, "input_sync_begin")
                with original_sync(*args, **kwargs):
                    if entry is not None:
                        self.mark(entry, "input_sync_end")
                    try:
                        yield
                    finally:
                        if entry is not None:
                            self.mark(entry, "input_prep_end")
            install("synchronize_input_prep", input_prep, original_sync)

    def finish(self, entries):
        for owner, name, instance_attribute, original in reversed(self.original_methods):
            if instance_attribute:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.original_methods.clear()
        self.runner = None
        self.metadata_entry = None
        self.dp_sync_entry = None
        gc.callbacks.remove(self.callback)
        return {"scope": "主机观测单独列出，不从设备forward耗时中扣除；未改变GC配置、未新增设备同步。",
                "gc_enabled": self.gc_enabled, "gc_threshold": self.gc_threshold,
                "metadata_builders": self.metadata_builders,
                "gc_freeze_count": self.gc_freeze_count, "gc_events": self.events,
                "steps": [{"step": e["index"], **e.get("host", {})} for e in entries]}
