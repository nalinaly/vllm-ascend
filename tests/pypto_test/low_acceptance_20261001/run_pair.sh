#!/usr/bin/env bash
# 固定混合bank的正式两版回归入口。输出目录必须全新。
set -eo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="$(realpath -m "${1:?新结果根目录}")"
snapshot="$(realpath "${2:?冻结源码}")"
bank="$(realpath "${3:?固定bank}")"
batch="${4:?batch}"
mean_min=3
mean_max=4
# 队列会在命令尾追加--device，不能把它误当成可选的均值范围。
if [[ -n "${5:-}" && "$5" != --device ]]; then
    mean_min="$5"
    mean_max="${6:?给出范围上界}"
fi
mkdir "$output"
for variant in performance precision; do
    bash "$script_dir/run_mixed.sh" "$output/$variant" "$snapshot" "$bank" "$batch" "$variant"
    PYTHONPATH="$snapshot/tests/pypto_test" python "$snapshot/tests/pypto_test/low_acceptance_20261001/mixed.py" \
        --root "$output/$variant" --bank "$bank" --batch "$batch" --mean-range "$mean_min" "$mean_max" \
        > "$output/${variant}_acceptance_summary.log"
done
PYTHONPATH="$snapshot/tests/pypto_test" python "$snapshot/tests/pypto_test/low_acceptance_20261001/compare_mixed.py" \
    --performance "$output/performance" --precision "$output/precision" --bank "$bank" --batch "$batch" \
    --mean-range "$mean_min" "$mean_max" --output "$output/comparison.json" > "$output/comparison.log"
