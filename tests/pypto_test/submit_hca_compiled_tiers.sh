#!/usr/bin/env bash
# 按上线编译口径（npugraph_ex + static_kernel + SuperKernel + inplace_pass）刷新
# 七档 Native vs PTO 对照。每个（档位, 侧）一个任务、各自一份干净的私有 OPP 目录，
# 用 --device auto 成批并行。
#
# 档位表（2026-09-29 定稿）：128K×B4/B8/B16/B24 + 8K×B16/B24/B32。
# 8K/B40 已退役：上线模板 --max-num-seqs 32 限制一步最多 32 个请求，B40 超容量。
#
# 两侧各一个进程、无法互相扣漂移（实测进程间 p50 stdev 6～13 μs），所以每个
# （档位, 侧）重复 REPEATS 轮，取各轮 p50 的中位数。
#
# 用法: REPEATS=3 submit_hca_compiled_tiers.sh <结果目录> [额外参数透传给 dsv4_hca_compiled_case.py]
set -eo pipefail
tests_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="$(realpath -m "${1:?指定结果目录}")"
shift
repeats="${REPEATS:-3}"
mkdir -p "$output"
tasks="$output/tasks.json"
printf '[\n' >"$tasks"
first=1
for tier in 131072:4 131072:8 131072:16 131072:24 8192:16 8192:24 8192:32; do
  history="${tier%%:*}"
  batch="${tier##*:}"
  for side in native pto; do
    # 验收口径：Native 开 SuperKernel（它最好的形态），PTO 结构上开不了，恒为 0。
    # 关掉 Native 的 SuperKernel 只用于 incore task 细分对比，靠 SK_NATIVE=0 覆盖。
    if [ "$side" = native ]; then sk="${SK_NATIVE:-1}"; else sk=0; fi
    for round in $(seq 0 $((repeats - 1))); do
      target="$output/h${history}_b${batch}/${side}_r${round}"
      mkdir -p "$target"
      id="$(task-submit --device auto --max-time 7200 \
        "bash $tests_dir/run_hca_compiled_case.sh $target $side --history $history --batch $batch --super-kernel $sk $*" \
        2>&1 | tail -n 1)"
      [ "$first" -eq 1 ] || printf ',\n' >>"$tasks"
      first=0
      printf '  {"history": %s, "batch": %s, "side": "%s", "round": %s, "task": "%s", "output": "%s"}' \
        "$history" "$batch" "$side" "$round" "$id" "$target" >>"$tasks"
      printf '%s h%s/b%s r%s %s\n' "$side" "$history" "$batch" "$round" "$id"
    done
  done
done
printf '\n]\n' >>"$tasks"
printf '任务清单: %s\n' "$tasks"
