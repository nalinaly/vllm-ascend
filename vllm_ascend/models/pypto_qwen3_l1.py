#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""L1 enqueue path for Qwen3-14B fused prefill/decode.

PyPTO L1 borrows the current torch_npu device/stream and enqueues one
operator. It does not ``stream.synchronize`` / ``npu.synchronize``. Later
torch ops on the same stream observe the writes.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.models.pypto_qwen3_adapter import materialize_npu_args, prefer_grok_pypto

prefer_grok_pypto()


def l1_execution_enabled() -> bool:
    """Single-card fused host uses L1 unless the caller forces ChipWorker."""
    mode = os.environ.get("PYPTO_QWEN3_EXECUTION", "l1").strip().lower()
    return mode not in {"l2", "chip", "chipworker"}


class PyptoL1Session:
    """Compile fused hosts once, then enqueue them through ``pypto.runtime.l1``."""

    def __init__(self, device_id: int | None = None) -> None:
        from pypto.runtime import RunConfig

        if device_id is None:
            device_id = int(os.environ.get("LOCAL_RANK", "0"))
        self.device_id = int(device_id)
        # Single-card trial: bake HBG so L1 can restore a graph package per
        # enqueue instead of TRB's AICPU-built task stream.
        self.runtime = os.environ.get("PYPTO_QWEN3_L1_RUNTIME", "host_build_graph").strip()
        self.config = RunConfig(platform="a2a3", device_id=self.device_id, runtime=self.runtime)
        self._compiled: dict[int, Any] = {}
        self._context: Any = None
        self._ops: dict[int, Any] = {}

    def compile(self, kernel: Any, sample_args: Sequence[torch.Tensor]) -> Any:
        key = id(kernel)
        cached = self._compiled.get(key)
        if cached is not None:
            return cached
        cpu_args = [tensor.detach().contiguous().cpu() for tensor in sample_args]
        previous_dir = self.config.save_kernels_dir
        kernel_name = getattr(kernel, "__name__", "pypto_kernel")
        rank = os.environ.get("LOCAL_RANK", "0")
        base = Path(os.environ.get("PYPTO_QWEN3_BUILD_DIR", "build_output"))
        self.config.save_kernels_dir = str(base / f"l1_{kernel_name}_rank{rank}_pid{os.getpid()}")
        try:
            compiled = kernel.compile(*cpu_args, config=self.config)
        finally:
            self.config.save_kernels_dir = previous_dir
        self._compiled[key] = compiled
        return compiled

    def _ensure_op(self, compiled: Any) -> Any:
        from pypto.runtime.l1 import L1Config, L1InitializationError, pypto_init

        key = id(compiled)
        existing = self._ops.get(key)
        if existing is not None:
            return existing
        if self._context is None:
            use_queue = os.environ.get("PYPTO_QWEN3_L1_TASK_QUEUE", "1") != "0"
            try:
                self._context = pypto_init(
                    programs=[compiled],
                    device=self.device_id,
                    config=L1Config(use_task_queue=use_queue),
                )
            except (RuntimeError, L1InitializationError):
                if not use_queue:
                    raise
                self._context = pypto_init(
                    programs=[compiled],
                    device=self.device_id,
                    config=L1Config(use_task_queue=False),
                )
            import pypto

            print(
                f"PYPTO_QWEN3_L1 ready device={self.device_id} "
                f"runtime={self.runtime} "
                f"task_queue={self._context._config.use_task_queue} "
                f"pypto={getattr(pypto, '__file__', None)}",
                flush=True,
            )
            op = self._context.operator(compiled)
        else:
            op = self._context.add_program(compiled)
        self._ops[key] = op
        return op

    def invoke(self, kernel: Any, args: Sequence[torch.Tensor]) -> None:
        """Enqueue ``kernel`` on the current torch_npu stream. No host sync."""
        live = materialize_npu_args(args)
        compiled = self.compile(kernel, live)
        op = self._ensure_op(compiled)
        param_infos, output_indices, _ = compiled._get_metadata()
        output_set = set(output_indices)
        positional: list[torch.Tensor] = []
        outs: list[torch.Tensor] = []
        for index, (info, tensor) in enumerate(zip(param_infos, live, strict=True)):
            del info
            if index in output_set:
                outs.append(tensor)
            else:
                positional.append(tensor)
        if outs:
            out_arg: torch.Tensor | tuple[torch.Tensor, ...] = outs[0] if len(outs) == 1 else tuple(outs)
            op(*positional, out=out_arg)
        else:
            op(*positional)


def invoke_pypto_kernel_l1(
    kernel: Any,
    args: Sequence[torch.Tensor],
    *,
    session: PyptoL1Session,
) -> None:
    if torch.npu.is_available():
        torch.npu.set_device(session.device_id)
    session.invoke(kernel, args)
