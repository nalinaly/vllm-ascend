# SPDX-License-Identifier: Apache-2.0
"""Control-only worker extension for D integration evidence, not timing."""

from collections import Counter


class OfflineCSAObserver:
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
