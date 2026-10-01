#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit分配16卡}"
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$experiment_dir/../../.." && pwd)"
workspace="$(dirname "$repo")"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
output="$(realpath -m "${1:?新结果目录}")"
snapshot="$(realpath "${2:?冻结源码}")"
bank="$(realpath "${3:?混合bank}")"
batch="${4:?batch}"
configuration="${5:?指定组合}"
backend=pto
attention=both
variant=performance
case "$configuration" in
    performance|csa_performance_pto_hca) ;;
    precision) variant=precision ;;
    native) backend=native ;;
    csa_precision_native_hca) attention=csa; variant=precision ;;
    csa_performance_native_hca) attention=csa ;;
    native_csa_pto_hca) attention=hca ;;
    *) echo "未知组合：$configuration" >&2; exit 2 ;;
esac
mkdir "$output"
python - "$configuration" "$backend" "$attention" "$variant" "$snapshot" "$bank" "$batch" \
    > "$output/configuration.json" <<'PY'
import json
import sys
print(json.dumps(dict(zip(('configuration', 'backend', 'attention', 'variant', 'snapshot', 'bank', 'batch'),
                          sys.argv[1:])), ensure_ascii=False, indent=2))
PY
compile_cwd="$(mktemp -d "$workspace/.cache/csa-mix-XXXXXX")"
printf '%s\n' "$compile_cwd" > "$output/compile_workdir.txt"
cd "$compile_cwd"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$workspace/.cache/migration-v0.25.1rc1/vllm:$snapshot:$snapshot/tests/pypto_test:${PYTHONPATH:-}"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export HCCL_DETERMINISTIC=true PTO_CSA_VARIANT="$variant"
export ASCEND_PROCESS_LOG_PATH="$output/ascend" PYPTO_PROG_BUILD_DIR="$output/build"
opp_workspace="$(mktemp -d "$workspace/.cache/csa-mix-opp-XXXXXX")"
export ASCEND_OPP_PATH="$(python "$snapshot/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$opp_workspace")"
printf '%s\n' "$ASCEND_OPP_PATH" > "$output/opp_path.txt"
# B24长档的既有容量口径；短档沿用0.95。
util=0.95
capture_sizes=(6 $((batch * 6)))
if [[ "$batch" == 24 ]]; then
    util=0.97
    # 长B24沿用已有的单144图容量口径；收尾由该图补位，不额外占用一份T6运行时。
    capture_sizes=(144)
fi
python "$snapshot/tests/pypto_test/offline_pd/run.py" decode \
    --bank "$bank" --output "$output" --backend "$backend" --pto-attention "$attention" --mixed-requests \
    --decode-dp 16 --gpu-memory-utilization "$util" --batch "$batch" --deterministic \
    --decode-tokens 192 --max-num-batched-tokens 400 --weight-nz-mode 2 \
    --graph-mode full_decode_only --capture-sizes "${capture_sizes[@]}" --port 30851 \
    > "$output/launch.log" 2>&1
python "$snapshot/tests/pypto_test/low_acceptance_20261001/mixed.py" \
    --root "$output" --bank "$bank" --batch "$batch" > "$output/acceptance_summary.log"
