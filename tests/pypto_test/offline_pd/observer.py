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
    # dummy 步里取几次 slot mapping 样本。
    _OFFLINE_SLOT_SAMPLES = 4
    # dummy 步里复算几次 compact slot mapping。
    _OFFLINE_COMPACT_SAMPLES = 3

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
                 "profiled_steps": 0, "closed": False, "window": [], "observed": {}}

        def profiled(scheduler_output, *args, **kwargs):
            import torch

            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
            # 把看到的 (tokens, requests) 分布记下来：判定失败时不必重跑就能诊断。
            key = f"{tokens}/{requests}"
            state["observed"][key] = state["observed"].get(key, 0) + 1
            # 稳态判据只看"所有请求都还在跑"，不再要求 tokens 恰好等于 batch*6。
            # 投机解码下每步的 token 数不恒定：首个 decode 步还没有 draft token，
            # 之后每步取决于上一步接受了几个，所以那个等式本来就不该指望
            # （batch 32 时实测 0 次命中）。expected_tokens 仍记录备查。
            steady = requests == expected_requests
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
                 "dummy_runs": 0, "dummy_tokens": [],
                 "step_probes": 0, "step_probe_results": [],
                 "in_dummy": False, "slot_samples": 0, "slot_probe_results": [],
                 "recaptures": 0,
                 "compact_samples": 0, "compact_probe_results": [],
                 "replay_calls": 0, "replay_padded": 0,
                 "replay_padded_allowed": 0, "replay_rejected": 0,
                 "records": [], "directory": str(directory)}
        self._offline_replay_probe(state)
        self._offline_dummy_probe(state)
        self._offline_slot_probe(state)
        self._offline_recapture_probe(state)

        def probed(builder, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
            result = original(builder, common_prefix_len, common_attn_metadata, fast_build, **kwargs)
            state["builds_seen"] += 1
            actual = kwargs.get("num_reqs_actual")
            padded = actual is not None and actual < common_attn_metadata.num_reqs
            if not padded:
                return result
            state["padded_builds"] += 1
            # dummy 步里把 decode metadata 暂存，供跑完之后复算 compact slot。
            # 不在这里算：算子调用虽然只产出 metadata、不碰 KV cache，
            # 但放在 build 内仍有扰动本步的风险，等 dummy 返回后再算更稳。
            if state["in_dummy"] and state["compact_samples"] < self._OFFLINE_COMPACT_SAMPLES:
                ratio = int(getattr(builder, "compressor_ratio", 0))
                if ratio == 4 and getattr(result, "decode", None) is not None:
                    state["pending_compact"] = (ratio, result.decode)
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

    def _offline_compact_sample(self, state, runner, num_tokens):
        """复算 dummy 步的 compact slot mapping，判断它会不会写进 cache。

        compact slot（cmp_slot_mapping／idx_slot_mapping）由 compressor_metadata
        算子在图内从 start_pos 与 block_table 现算，**不来自**被 fill 成 -1 的那个
        slot_mapping 缓冲，所以主 slot 那轮测量覆盖不到它。图内产出的张量 Python
        侧读不到，但算子是纯 metadata 生产者（只产出 cos/sin/slots，不碰 KV cache），
        用同一份 metadata 复算一次即可得到与图内一致的结果。

        必须取**同一层**的 impl：compress_ratio 不匹配会直接报错
        （task_20260924_000518 即因此失败）。这里按 builder 的 compressor_ratio
        找到对应层。
        """
        pending = state.pop("pending_compact", None)
        if pending is None or state["compact_samples"] >= self._OFFLINE_COMPACT_SAMPLES:
            return
        ratio, decode = pending
        try:
            import torch

            impl = None
            for layer in runner.get_model().model.layers:
                attention = layer.self_attn
                if getattr(attention, "compress_ratio", 0) == ratio:
                    impl = attention.dsa_attn.dsa_attn.impl
                    break
            if impl is None:
                state.setdefault("compact_probe_error", f"no layer with compress_ratio={ratio}")
                return
            torch.npu.synchronize()
            _cos, _sin, slots = impl._compute_compressor_metadata(decode)
            flat = slots.reshape(-1, slots.shape[-1]) if slots.dim() > 1 else slots.reshape(-1, 1)
            pages = flat[:, 0]
            state["compact_samples"] += 1
            state["compact_probe_results"].append({
                "num_tokens": int(num_tokens),
                "rows": int(flat.shape[0]),
                "non_negative_pages": int((pages >= 0).sum().item()),
                "max_page": int(pages.max().item()),
                "min_page": int(pages.min().item()),
            })
        except Exception as error:
            state.setdefault("compact_probe_error", repr(error))

    def _offline_recapture_probe(self, state):
        """统计采集窗口内新建了多少个 NPUGraph，用来判定有没有 replay 期重新捕获。

        `ACLGraphWrapper.__call__` 里 `entry.aclgraph is None` 走捕获分支、
        否则重放，捕获时会 `torch.npu.NPUGraph()`。采集窗口开在预热与捕获之后，
        窗口内再出现新的 NPUGraph 就意味着重新捕获——这正是 T1.5 那条
        "无 replay 期重新编译"要排除的情形。计数为 0 即判定通过。
        """
        try:
            import torch

            origin = torch.npu.NPUGraph
        except Exception as error:
            state["recapture_probe"] = f"unavailable: {error!r}"
            return

        def counted(*args, **kwargs):
            state["recaptures"] += 1
            return origin(*args, **kwargs)

        torch.npu.NPUGraph = counted
        self._offline_recapture_origin = origin

    def _offline_slot_probe(self, state):
        """记录 dummy 步里 PTO 实际拿到的 slot mapping。

        整份缓存视图对比在这套布局下不可能成立——`decode_cache_layout_v1` 实测
        `cmp_kv` 与 `compress_state` 共用同一块 678MB 分配、重叠 676MB，写一处
        会让多个视图一起"变化"。改为直接看**输入**：若某个 slot mapping 全为
        -1，kernel 的 `page >= 0` 守卫就必然挡住，写不进去，与存储布局无关。
        """
        try:
            from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import CSAServiceRuntime
        except Exception as error:
            state["slot_probe"] = f"unavailable: {error!r}"
            return

        origin = CSAServiceRuntime.__call__

        def probed(runtime, context, hidden, positions, output, kv_cache):
            if state["in_dummy"] and state["slot_samples"] < self._OFFLINE_SLOT_SAMPLES:
                try:
                    metadata = context.attn_metadata
                    record = {"dummy_index": state["dummy_runs"], "tokens": int(hidden.shape[0])}
                    for name, prefix in runtime.prefixes.items():
                        decode = metadata[prefix].decode
                        slots = getattr(decode, "slot_mapping", None)
                        if slots is None:
                            continue
                        flat = slots.reshape(-1)
                        record[name] = {"count": int(flat.numel()),
                                        "non_negative": int((flat >= 0).sum().item()),
                                        "max": int(flat.max().item())}
                    state["slot_samples"] += 1
                    state["slot_probe_results"].append(record)
                except Exception as error:
                    state.setdefault("slot_probe_error", repr(error))
            return origin(runtime, context, hidden, positions, output, kv_cache)

        CSAServiceRuntime.__call__ = probed
        self._offline_slot_origin = origin

    def _offline_dummy_body(self, state, origin, runner, num_tokens, args, kwargs):
        """dummy 的实际执行体；顺便读一次常驻 slot mapping 缓冲。

        图重放不跑 Python 前向闸门，所以 hook CSAServiceRuntime.__call__ 在
        dummy 步上一次都不会触发（实测 dummy_runs=26 而 slot_samples=0）。
        但重放读的就是这些固定地址的常驻缓冲，从 Python 侧可以直接读到。
        `model_runner_v1.py` 在非图捕获时会把它们填成 -1，这里核实是否属实：
        若确为全 -1，kernel 的 `page >= 0` 守卫必然挡住主 slot 那条写入路径。
        """
        result = origin(runner, num_tokens, *args, **kwargs)
        self._offline_compact_sample(state, runner, num_tokens)
        if state["slot_samples"] >= self._OFFLINE_SLOT_SAMPLES:
            return result
        try:
            import torch

            torch.npu.synchronize()
            record = {"dummy_index": state["dummy_runs"], "num_tokens": int(num_tokens),
                      "groups": {}}
            for gid in range(len(runner.kv_cache_config.kv_cache_groups)):
                table = runner.input_batch.block_table[gid]
                flat = table.slot_mapping.gpu.reshape(-1)
                entry = {
                    "count": int(flat.numel()),
                    "non_negative": int((flat >= 0).sum().item()),
                    "max": int(flat.max().item()),
                }
                # compact slot mapping 由算子在图内从 start_pos 与 block_table 现算，
                # 不来自上面那个被 fill 成 -1 的缓冲。但它的**输入**同样是常驻的，
                # 这里一并读：若 block_table 全为 0，算出的 compact slot 必然指向
                # 0 号页，而 0 号页是 vLLM 保留的 null block（永不分配给任何请求），
                # 那条写入路径就是无害的。
                blocks = getattr(table, "block_table", None)
                gpu = getattr(blocks, "gpu", blocks)
                if gpu is not None and hasattr(gpu, "reshape"):
                    bt = gpu.reshape(-1)
                    entry["block_table"] = {
                        "count": int(bt.numel()),
                        "non_zero": int((bt != 0).sum().item()),
                        "max": int(bt.max().item()),
                    }
                record["groups"][str(gid)] = entry
            state["slot_samples"] += 1
            state["slot_probe_results"].append(record)
        except Exception as error:
            state.setdefault("slot_probe_error", repr(error))
        return result

    def _offline_dummy_probe(self, state):
        """统计本 rank 跑了多少次 dummy batch，以及其中有多少次是空调度。

        空 rank（D04 / T1.6）不能靠耗时推断：rank0 与 rank1 测量耗时相同只能说明
        两者被 DP 锁步，说明不了引擎到底有没有真的空转。这里直接数
        NPUModelRunner._dummy_run 的调用次数，并按 with_prefill / num_reqs 分类。
        """
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        origin = getattr(NPUModelRunner, "_dummy_run", None)
        if origin is None:
            state["dummy_probe"] = "absent"
            return

        def counted(runner, num_tokens, *args, **kwargs):
            state["dummy_runs"] += 1
            state["dummy_tokens"].append(int(num_tokens))
            state["in_dummy"] = True
            try:
                return self._offline_dummy_body(state, origin, runner, num_tokens, args, kwargs)
            finally:
                state["in_dummy"] = False

        NPUModelRunner._dummy_run = counted
        self._offline_dummy_origin = origin

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
        recapture_origin = getattr(self, "_offline_recapture_origin", None)
        if recapture_origin is not None:
            import torch
            torch.npu.NPUGraph = recapture_origin
            self._offline_recapture_origin = None
        slot_origin = getattr(self, "_offline_slot_origin", None)
        if slot_origin is not None:
            from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import CSAServiceRuntime
            CSAServiceRuntime.__call__ = slot_origin
            self._offline_slot_origin = None
        dummy_origin = getattr(self, "_offline_dummy_origin", None)
        if dummy_origin is not None:
            from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
            NPUModelRunner._dummy_run = dummy_origin
            self._offline_dummy_origin = None
        self._offline_padding = None
        tokens = state.pop("dummy_tokens", [])
        if tokens:
            histogram = {}
            for value in tokens:
                histogram[value] = histogram.get(value, 0) + 1
            state["dummy_token_histogram"] = dict(sorted(histogram.items()))
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
