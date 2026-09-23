# SPDX-License-Identifier: Apache-2.0
"""Control-only worker extension for D integration evidence, not timing."""

import os
from collections import Counter


def _enable_swimlane_init():
    """按环境变量给本进程的 pypto.torch.init 补上 DFX 泳道参数。

    pypto 只在第一次 init 时绑定诊断配置，而生产代码在 process_weights_after_loading
    里调用它。worker extension 在 load_model 之前就被解析导入，所以在模块导入时打补丁
    即可采集泳道，不需要为了诊断去改生产实现。只有显式设置环境变量的进程会被改写。
    """
    directory = os.environ.get("OFFLINE_PTO_SWIMLANE_DIR")
    if not directory:
        return
    import pypto.torch

    original = pypto.torch.init

    def init(**kwargs):
        return original(**kwargs, enable_chip_swimlane=4, enable_dep_gen=True, output_dir=directory)

    pypto.torch.init = init


_enable_swimlane_init()


class OfflineCSAObserver:
    def offline_cache_layout(self):
        """只记录真实缓存描述符，用于定位共享存储边界，不读取设备数据。"""
        from types import SimpleNamespace

        from vllm_ascend.ops.dsa import _build_kv_cache
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_storage import indexer_storage, physical_pages

        def describe(tensor):
            return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                    "dtype": str(tensor.dtype), "pointer": tensor.data_ptr(),
                    "storage_pointer": tensor.untyped_storage().data_ptr(),
                    "storage_offset": tensor.storage_offset(),
                    "storage_bytes": tensor.untyped_storage().nbytes(),
                    "view_bytes": tensor.numel() * tensor.element_size()}

        layers = {}
        for index, layer in enumerate(self.model_runner.get_model().model.layers):
            attention = layer.self_attn
            if attention.compress_ratio != 4:
                continue
            cmp_kv, swa, state, inner_state, key, scale = _build_kv_cache(
                attention.dsa_attn, SimpleNamespace(virtual_engine=None))
            tensors = {"cmp_kv": cmp_kv, "kv_cache": swa,
                       "compress_state": physical_pages(state),
                       "inner_compress_state": physical_pages(inner_state),
                       "idx_kv_cache": indexer_storage(key, scale)}
            views = {name: describe(tensor) for name, tensor in tensors.items()}
            overlaps = []
            names = list(views)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    a, b = views[left], views[right]
                    begin = max(a["pointer"], b["pointer"])
                    end = min(a["pointer"] + a["view_bytes"], b["pointer"] + b["view_bytes"])
                    if begin < end:
                        overlaps.append({"left": left, "right": right, "overlap_bytes": end - begin,
                                         "same_byte_range": (a["pointer"], a["view_bytes"]) ==
                                                            (b["pointer"], b["view_bytes"]),
                                         "same_descriptor": all(a[k] == b[k] for k in
                                                                ("pointer", "shape", "stride", "dtype"))})
            layers[str(index)] = {"views": views, "overlaps": overlaps}
        return {"scope": "仅采集描述符，无PTO计算或精度结论", "layers": layers}

    def offline_begin_observation(self):
        from vllm.forward_context import get_forward_context

        if getattr(self, "_offline_csa_handles", None):
            raise RuntimeError("CSA observation is already active")
        self._offline_csa_counts = {}
        self._offline_csa_handles = []
        model = self.model_runner.get_model()
        for index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            if attention.compress_ratio != 4:
                continue
            counts = Counter()
            self._offline_csa_counts[str(index)] = counts

            def completed(module, args, kwargs, output, counts=counts):
                hidden = kwargs["hidden_states"]
                positions = kwargs["positions"]
                context = get_forward_context()
                runtime = getattr(module.dsa_attn, "_pto_csa_runtime", None)
                selected = runtime is not None and runtime.eligible(context, hidden, positions)
                counts[f"{'pto' if selected else 'native'}_tokens{hidden.shape[0]}"] += 1

            self._offline_csa_handles.append(attention.register_forward_hook(completed, with_kwargs=True))
        if len(self._offline_csa_counts) != 21:
            raise RuntimeError(f"Expected 21 target CSA layers, got {len(self._offline_csa_counts)}")
        return {"target_csa_layers": list(self._offline_csa_counts),
                "dp_rank": self.vllm_config.parallel_config.data_parallel_rank}

    def offline_end_observation(self):
        for handle in self._offline_csa_handles:
            handle.remove()
        self._offline_csa_handles = []
        return {layer: dict(counts) for layer, counts in self._offline_csa_counts.items()}

    def offline_begin_profile(self, directory, start_step, steps, expected_tokens, expected_requests):
        """从指定 step 起采集整步 CPU+NPU 数据：Level1、带 device kernel、不采 Python stack。

        只统计达到稳态构成的 step，窗口两端各做一次同步，确保设备任务完整落在窗口内。
        窗口结束就地关闭，让各 rank 同时进入后处理，避免单 rank 落后拖住集合通信。
        采集本身带同步和落盘开销，这个窗口只用于结构对照，不产出稳态性能结论。
        """
        import torch_npu

        if getattr(self, "_offline_profile", None) is not None:
            raise RuntimeError("Profiling is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        target = os.path.join(directory, f"rank{rank}")
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(target),
        )
        runner = self.model_runner
        original = runner.execute_model
        state = {"dp_rank": rank, "trace_dir": target, "start_step": start_step,
                 "requested_steps": steps, "expected_tokens": expected_tokens,
                 "expected_requests": expected_requests, "seen_steady_steps": 0,
                 "profiled_steps": 0, "closed": False, "window": []}

        def profiled(scheduler_output, *args, **kwargs):
            import torch

            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
            steady = tokens == expected_tokens and requests == expected_requests
            index = state["seen_steady_steps"]
            if steady:
                state["seen_steady_steps"] += 1
            active = steady and index >= start_step and state["profiled_steps"] < steps
            if active and state["profiled_steps"] == 0:
                torch.npu.synchronize()
                profiler.start()
            try:
                return original(scheduler_output, *args, **kwargs)
            finally:
                if active:
                    state["profiled_steps"] += 1
                    state["window"].append({"steady_step_index": index, "scheduled_tokens": tokens,
                                            "requests": requests})
                    if state["profiled_steps"] == steps:
                        torch.npu.synchronize()
                        profiler.stop()
                        state["closed"] = True

        runner.execute_model = profiled
        self._offline_profile = (state, profiler, original)
        return {"dp_rank": rank, "trace_dir": target}

    def offline_end_profile(self):
        import torch

        state, profiler, original = self._offline_profile
        self.model_runner.execute_model = original
        self._offline_profile = None
        if state["profiled_steps"] and not state["closed"]:
            torch.npu.synchronize()
            profiler.stop()
            state["closed"] = True
        if state["profiled_steps"] != state["requested_steps"]:
            raise RuntimeError(f"Profiled {state['profiled_steps']} steady steps, "
                               f"requested {state['requested_steps']}; see window")
        return state

    def offline_begin_host_profile(self, directory, start_step, steps, expected_tokens, expected_requests):
        """在稳态 step 上开 cProfile，定位 CSA 调用里的主机侧耗时函数。

        cProfile 会放大 Python 调用开销，得到的是相对归因，不能与设备侧耗时直接相加，
        也不能当作稳态性能结论。窗口构成判定与 NPU 采集一致，只统计满批稳态 step。
        """
        import cProfile

        if getattr(self, "_offline_host_profile", None) is not None:
            raise RuntimeError("Host profiling is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        profiler = cProfile.Profile()
        runner = self.model_runner
        original = runner.execute_model
        state = {"dp_rank": rank, "start_step": start_step, "requested_steps": steps,
                 "expected_tokens": expected_tokens, "expected_requests": expected_requests,
                 "seen_steady_steps": 0, "profiled_steps": 0, "closed": False,
                 "path": os.path.join(directory, f"rank{rank}.prof")}

        def profiled(scheduler_output, *args, **kwargs):
            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
            steady = tokens == expected_tokens and requests == expected_requests
            index = state["seen_steady_steps"]
            if steady:
                state["seen_steady_steps"] += 1
            active = steady and index >= start_step and state["profiled_steps"] < steps
            if active and state["profiled_steps"] == 0:
                profiler.enable()
            try:
                return original(scheduler_output, *args, **kwargs)
            finally:
                if active:
                    state["profiled_steps"] += 1
                    if state["profiled_steps"] == steps:
                        profiler.disable()
                        state["closed"] = True

        runner.execute_model = profiled
        self._offline_host_profile = (state, profiler, original)
        return {"dp_rank": rank, "path": state["path"]}

    def offline_end_host_profile(self):
        import pstats

        state, profiler, original = self._offline_host_profile
        self.model_runner.execute_model = original
        self._offline_host_profile = None
        if state["profiled_steps"] and not state["closed"]:
            profiler.disable()
            state["closed"] = True
        if state["profiled_steps"] != state["requested_steps"]:
            raise RuntimeError(f"Host-profiled {state['profiled_steps']} steady steps, "
                               f"requested {state['requested_steps']}")
        os.makedirs(os.path.dirname(state["path"]), exist_ok=True)
        profiler.dump_stats(state["path"])
        rows = []
        for func, (calls, primitive, total, cumulative, _) in pstats.Stats(profiler).stats.items():
            rows.append({"function": f"{func[0]}:{func[1]}({func[2]})", "calls": calls,
                         "primitive_calls": primitive, "tottime_seconds": round(total, 4),
                         "cumtime_seconds": round(cumulative, 4)})
        state["top_by_tottime"] = sorted(rows, key=lambda r: r["tottime_seconds"], reverse=True)[:30]
        state["top_by_cumtime"] = sorted(rows, key=lambda r: r["cumtime_seconds"], reverse=True)[:30]
        return state

    def offline_begin_padding_capture(self, directory, layer_index):
        """在真实离线 D 上捕获一次带补位请求的 decode 步，落盘 Native metadata 与 compact 形状。

        挂在 eligible 而不是 __call__：补位时 PTO 会回退 Native，__call__ 根本不会进，
        而 eligible 每步都被调用，且看到的 metadata 与生产路径完全一致。

        只读取并落盘小张量，额外只多调一次 Native 自己的 compressor_metadata
        生产器来量其真实形状；不改变本步的计算路径与结果。
        """
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import CSAServiceRuntime

        if getattr(self, "_offline_padding", None) is not None:
            raise RuntimeError("Padding capture is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        attention = self.model_runner.get_model().model.layers[layer_index].self_attn
        wanted = getattr(attention.dsa_attn, "_pto_csa_runtime", None)
        if wanted is None:
            raise ValueError(f"Layer {layer_index} has no PTO CSA runtime")
        original = CSAServiceRuntime.eligible
        state = {"dp_rank": rank, "layer_index": layer_index,
                 "layer_name": wanted.layer_name, "captured": 0, "steps_seen": 0}

        def probed(runtime, context, hidden, positions):
            verdict = original(runtime, context, hidden, positions)
            if runtime is not wanted:
                return verdict
            state["steps_seen"] += 1
            if not state["captured"]:
                record = self._offline_padding_record(runtime, context, hidden, positions, verdict)
                if record is not None:
                    state["captured"] += 1
                    self._offline_padding_write(directory, rank, record)
            return verdict

        CSAServiceRuntime.eligible = probed
        self._offline_padding = (state, original)
        return dict(state)

    def _offline_padding_record(self, runtime, context, hidden, positions, verdict):
        """只在本步确实存在补位请求时返回记录，否则返回 None。"""
        import torch

        metadata = context.attn_metadata
        if metadata is None:
            return None
        tokens = hidden.shape[0]
        groups = {}
        padded = False
        for name, prefix in runtime.prefixes.items():
            item = metadata.get(prefix)
            if item is None or item.decode is None:
                return None
            decode = item.decode
            seq_lens = decode.seq_lens
            # Native 的补位判据：seq_lens 清零、start_pos 清零、页表行清零。
            zero_len = int(torch.count_nonzero(seq_lens == 0).item())
            entry = {
                "num_actual_tokens": int(item.num_actual_tokens),
                "num_decodes": int(item.num_decodes),
                "num_reqs_actual": None if decode.num_reqs_actual is None else int(decode.num_reqs_actual),
                "seq_lens": seq_lens.tolist(),
                "seq_lens_zero_count": zero_len,
                "block_table_shape": list(decode.block_table.shape),
                "query_start_loc": decode.query_start_loc.tolist(),
            }
            start_pos = getattr(decode, "start_pos", None)
            if start_pos is not None:
                entry["start_pos"] = start_pos.tolist()
            # 每行页表的非零列数，用来区分被清零的补位行。
            entry["block_table_nonzero_per_row"] = torch.count_nonzero(
                decode.block_table, dim=-1).flatten().tolist()
            groups[name] = entry
            if entry["num_reqs_actual"] is not None and entry["num_reqs_actual"] < entry["num_decodes"]:
                padded = True
            if entry["num_actual_tokens"] < tokens or zero_len:
                padded = True
        if not padded:
            return None

        # 量 Native 自己的 compact 生产器在这份补位 metadata 上的真实输出。
        impl = runtime.wrapper.dsa_attn.impl
        compact = {}
        for name in ("compressed", "indexer"):
            decode = metadata[runtime.prefixes[name]].decode
            cos, sin, slots = impl._compute_compressor_metadata(decode)
            slots_cpu = slots.cpu()
            # slot 的列布局由 DeviceOperator.get_dsa_compressor_slot_mapping_format()
            # 决定，这里不假定它是 [rows, 2]；按实际维度取第一列用于判负。
            first = slots_cpu if slots_cpu.dim() == 1 else slots_cpu.reshape(slots_cpu.shape[0], -1)[:, 0]
            compact[name] = {
                "cos_shape": list(cos.shape), "sin_shape": list(sin.shape),
                "slot_shape": list(slots_cpu.shape), "slot_dtype": str(slots_cpu.dtype),
                "num_compressed_tokens": int(decode.num_compressed_tokens),
                "slot_negative_rows": int((first < 0).sum().item()),
                "slot_first_column": first.tolist(),
            }
        return {"tokens": int(tokens), "eligible": bool(verdict),
                "positions": positions.tolist(), "groups": groups, "compact": compact}

    def _offline_padding_write(self, directory, rank, record):
        import json
        import pathlib

        target = pathlib.Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"padding_capture_rank{rank}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def offline_end_padding_capture(self):
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import CSAServiceRuntime

        if getattr(self, "_offline_padding", None) is None:
            return {"dp_rank": self.vllm_config.parallel_config.data_parallel_rank, "captured": 0}
        state, original = self._offline_padding
        CSAServiceRuntime.eligible = original
        self._offline_padding = None
        return dict(state)

    def offline_begin_swimlane(self, layer_index, expected_tokens):
        """只给一层、一次达到稳态构成的 CSA 调用开 DFX 窗口，其余调用保持原路径。"""
        import pypto.torch

        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import CSAServiceRuntime

        if getattr(self, "_offline_swimlane", None) is not None:
            raise RuntimeError("Swimlane capture is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        if not os.environ.get("OFFLINE_PTO_SWIMLANE_DIR"):
            self._offline_swimlane = None
            return {"dp_rank": rank, "enabled": False}
        attention = self.model_runner.get_model().model.layers[layer_index].self_attn
        wanted = getattr(attention.dsa_attn, "_pto_csa_runtime", None)
        if wanted is None:
            raise ValueError(f"Layer {layer_index} has no PTO CSA runtime")
        original = CSAServiceRuntime.__call__
        state = {"dp_rank": rank, "enabled": True, "layer_index": layer_index,
                 "layer_name": wanted.layer_name, "expected_tokens": expected_tokens, "captured": 0}

        def traced(runtime, context, hidden, *args, **kwargs):
            if runtime is not wanted or state["captured"] or hidden.shape[0] != expected_tokens:
                return original(runtime, context, hidden, *args, **kwargs)
            pypto.torch.begin_dfx()
            try:
                return original(runtime, context, hidden, *args, **kwargs)
            finally:
                pypto.torch.end_dfx()
                state["captured"] += 1

        CSAServiceRuntime.__call__ = traced
        self._offline_swimlane = (state, original)
        return dict(state)

    def offline_end_swimlane(self):
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import CSAServiceRuntime

        if self._offline_swimlane is None:
            return {"dp_rank": self.vllm_config.parallel_config.data_parallel_rank,
                    "enabled": False, "captured": 0}
        state, original = self._offline_swimlane
        CSAServiceRuntime.__call__ = original
        self._offline_swimlane = None
        if state["captured"] != 1:
            raise RuntimeError(f"Expected exactly one DFX window, captured {state['captured']}")
        return dict(state)
