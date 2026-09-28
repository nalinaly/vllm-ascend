#!/usr/bin/env bash
# 把七个单卡档位（128K×B4/B8/B16/B24、8K×B24/B32/B40）各提交一张卡，成批并行。
# 档位表于 2026-09-29 调整：新增 128K/B24，去除 8K/B16。
# 每档跑 run_hca_reference_pair.sh：先基线、再候选，候选对基线快照做逐 bit 门禁并计时。
# 本机 task-submit 只接受“一整条命令字符串”，不能按 argv 传，否则参数会被它当成自己的选项。
# 用法：submit_hca_seven_tier.sh <结果目录> <基线源码> <候选源码>
set -eo pipefail
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hca_repo="$(cd "$hca_tests/../.." && pwd)"
output="$(realpath -m "${1:?指定结果目录}")"
baseline="$(realpath "${2:?指定基线源码}")"
candidate="$(realpath "${3:?指定候选源码}")"
mkdir -p "$output"
tasks="$output/tasks.json"
printf '[\n' >"$tasks"
first=1
for tier in 131072:4 131072:8 131072:16 131072:24 8192:24 8192:32 8192:40; do
  history="${tier%%:*}"
  batch="${tier##*:}"
  target="$output/h${history}_b${batch}"
  handle="$(task-submit --device auto --max-time 5400 \
    "cd $hca_repo && TASK_DEVICE=\"\$TASK_DEVICE\" bash $hca_tests/run_hca_reference_pair.sh $target $history $batch $baseline $candidate timing" \
    | tail -n 1)"
  [[ $first -eq 1 ]] || printf ',\n' >>"$tasks"
  first=0
  printf '  {"history": %s, "batch": %s, "task": "%s", "output": "%s"}' \
    "$history" "$batch" "$handle" "$target" >>"$tasks"
  echo "h${history}_b${batch} -> $handle"
done
printf '\n]\n' >>"$tasks"
echo "句柄清单：$tasks"
