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
    # CSA 的五个 cache group：swa、compressed、state、indexer、indexer_state。
    _OFFLINE_PADDING_GROUPS = 5

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

        挂在 Native 的 AscendDSAMetadataBuilder.build 上，而不是 CSAServiceRuntime：
        后者只在 PTO 后端存在。builder 两个后端都会走，看到的就是生产路径的 metadata。
        （T1.4 之前 can_replay_csa_graph 还会在补位时主动回退 eager，把 padding 自己
        消掉，这也是必须挂 builder 的原因之一；该闸门已于 T1.4 放开。）

        只读取并落盘小张量，额外只多调一次 Native 自己的 compressor_metadata
        生产器来量其真实形状；不改变本步的计算路径与结果。

        同时统计 can_replay_csa_graph 的判定结果，用来证明补位档位确实进入了图重放，
        而不是静默回退 Native/eager（T1.4 的完成判据之一）。
        """
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

        if getattr(self, "_offline_padding", None) is not None:
            raise RuntimeError("Padding capture is already active")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        original = AscendDSAMetadataBuilder.build
        state = {"dp_rank": rank, "layer_index": layer_index,
                 "captured": 0, "builds_seen": 0, "padded_builds": 0,
                 "replay_calls": 0, "replay_padded": 0,
                 "replay_padded_allowed": 0, "replay_rejected": 0,
                 "records": [], "directory": str(directory)}
        self._offline_replay_probe(state)

        def probed(builder, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
            result = original(builder, common_prefix_len, common_attn_metadata, fast_build, **kwargs)
            state["builds_seen"] += 1
            actual = kwargs.get("num_reqs_actual")
            padded = actual is not None and actual < common_attn_metadata.num_reqs
            if not padded:
                return result
            state["padded_builds"] += 1
            # 一步里每个 cache group 各 build 一次，采满一轮即停，不逐步累积。
            if state["captured"] < self._OFFLINE_PADDING_GROUPS:
                record = self._offline_padding_record(builder, common_attn_metadata, actual, result)
                if record is not None:
                    state["captured"] += 1
                    state["records"].append(record)
            return result

        AscendDSAMetadataBuilder.build = probed
        self._offline_padding = (state, original)
        return dict(state)

    def _offline_replay_probe(self, state):
        """统计 can_replay_csa_graph 的判定，只计数不改判定结果。

        Native 后端不注入 PyptoCSADeepseekV4ForCausalLM，is_csa_model 为假，
        该函数根本不会被调用，计数保持为 0——这本身就是两个后端的区分证据。
        """
        from vllm_ascend.worker import model_runner_v1

        origin = getattr(model_runner_v1, "can_replay_csa_graph", None)
        if origin is None:
            state["replay_probe"] = "absent"
            return

        def counted(*, num_tokens, num_reqs, uniform_decode, padded_tokens):
            verdict = origin(num_tokens=num_tokens, num_reqs=num_reqs,
                             uniform_decode=uniform_decode, padded_tokens=padded_tokens)
            state["replay_calls"] += 1
            if num_tokens < padded_tokens:
                state["replay_padded"] += 1
                if verdict:
                    state["replay_padded_allowed"] += 1
            if not verdict:
                state["replay_rejected"] += 1
            return verdict

        model_runner_v1.can_replay_csa_graph = counted
        self._offline_replay_origin = origin

    def _offline_padding_record(self, builder, common, num_reqs_actual, result):
        """落盘这一份补位 metadata 的关键量；拿不到 decode 段时返回 None。"""
        import torch

        decode = getattr(result, "decode", None)
        if decode is None:
            return None
        seq_lens = decode.seq_lens
        entry = {
            "prefix": list(getattr(builder, "layer_names", []) or [])[:1],
            "compressor_ratio": int(getattr(builder, "compressor_ratio", 0)),
            "num_reqs_padded": int(common.num_reqs),
            "num_reqs_actual": int(num_reqs_actual),
            "num_actual_tokens": int(common.num_actual_tokens),
            "num_input_tokens": int(getattr(common, "num_input_tokens", 0)),
            "num_decodes": int(result.num_decodes),
            # Native 归一化的三个量，用于在计划第 4 节的方案 C 与 D 之间定夺。
            "seq_lens": seq_lens.tolist(),
            "seq_lens_zero_count": int((seq_lens == 0).sum().item()),
            "query_start_loc": decode.query_start_loc.tolist(),
            "block_table_shape": list(decode.block_table.shape),
            "block_table_nonzero_per_row": torch.count_nonzero(
                decode.block_table.reshape(decode.block_table.shape[0], -1), dim=-1).tolist(),
        }
        start_pos = getattr(decode, "start_pos", None)
        if start_pos is not None:
            entry["start_pos"] = start_pos.tolist()
        positions = getattr(common, "positions", None)
        if positions is not None:
            entry["positions"] = positions[: int(getattr(common, "num_input_tokens", 0)) or None].tolist()

        # compact 行数就是 Native 自己算好的 num_compressed_tokens
        # （dsa_v1.py:606 _num_compressor_metadata_rows 的结果），直接读取即可，
        # 不额外调用 compressor_metadata 算子：那需要与本 builder 同一层的 impl，
        # 取错层会因 compress_ratio 不匹配而报错，也会扰动本步。
        rows = getattr(decode, "num_compressed_tokens", None)
        if rows is not None:
            entry["num_compressed_tokens"] = int(rows)
        return entry


    def _offline_padding_write(self, directory, rank, records):
        import json
        import pathlib

        target = pathlib.Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"padding_capture_rank{rank}.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def offline_end_padding_capture(self):
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

        if getattr(self, "_offline_padding", None) is None:
            return {"dp_rank": self.vllm_config.parallel_config.data_parallel_rank, "captured": 0}
        state, original = self._offline_padding
        AscendDSAMetadataBuilder.build = original
        origin = getattr(self, "_offline_replay_origin", None)
        if origin is not None:
            from vllm_ascend.worker import model_runner_v1
            model_runner_v1.can_replay_csa_graph = origin
            self._offline_replay_origin = None
        self._offline_padding = None
        directory, rank = state.pop("directory"), state["dp_rank"]
        records = state.pop("records")
        if records:
            self._offline_padding_write(directory, rank, records)
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
