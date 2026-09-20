# Qwen3-14B PyPTO profiling examples

These scripts are reference entry points for single-card, BF16, ND-format
Qwen3-14B validation. Run them after loading the same `pto_eager` and
vLLM-Ascend environment used by the model process.

The local scripts call `pto_eager_env.activate_pto_eager()` before importing
vLLM. This only repairs conflicting editable installs in a development
workspace (including spawned workers); production model code uses the public
installed `pypto` package and contains no source-tree selection logic.

## ACL Graph Native/PyPTO profiling comparison

The two variants must run in separate processes. Decode uses one full-model ACL
Graph (`FULL_DECODE_ONLY`), not a piecewise graph. Stack, shape, and memory
profiling are disabled to keep the JSON small.

```bash
ROOT=/tmp/qwen3_aclgraph_profile

python tests/pypto_test/qwen3_aclgraph_profile.py \
  --variant native --output-dir "$ROOT/native"

python tests/pypto_test/qwen3_aclgraph_profile.py \
  --variant pypto --output-dir "$ROOT/pypto"

python tests/pypto_test/compare_qwen3_aclgraph_profiles.py --root "$ROOT"
```

The default `--decode-steps 3` requests four output tokens: the first token is
sampled by prefill, followed by exactly three decode ACL Graph replays. The
capture script writes an easy-to-open `<variant>_profiling.json`; the comparison
script verifies the `aclmdlRIExecuteAsync` count and writes `comparison.json`.

Do not set `VLLM_ASCEND_QWEN3_PYPTO_DFX_DIR` for this workflow. DFX boundaries
synchronize the stream and therefore must not be mixed with ACL Graph profiling.

## One-layer PyPTO swimlane in an end-to-end model run

```bash
python tests/pypto_test/qwen3_single_layer_swimlane.py \
  --output-dir /tmp/qwen3_pypto_layer0_swimlane
```

The script sets `VLLM_ASCEND_QWEN3_PYPTO_DFX_DIR`, disables ACL Graph, runs the
full model, and captures only transformer layer 0's first PyPTO decode attention
call. Qwen3-14B's transformer layers use the same attention implementation, so
capturing all 40 layers would only repeat the same task graph.

Outputs include:

- `chip_swimlane_records.json`: raw level-4 PyPTO/Simpler timing records;
- `deps.json`: dependency and kernel-name mapping;
- `merged_swimlane.json`: Perfetto-compatible swimlane;
- `capture_summary.json`: the exact model/run conditions.

`begin_dfx()` and `end_dfx()` both drain the current torch_npu stream and host
task queue. Consequently, DFX artifacts are diagnostic views, not unbiased
latency measurements. Use the ACL Graph profiling workflow above for performance
comparison.
