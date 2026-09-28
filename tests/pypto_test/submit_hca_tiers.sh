#!/usr/bin/env bash
# 把六个单卡验收档位（128K×B4/B8/B16/B24、8K×B24/B32）各提交一张卡，成批并行。
# 档位表于 2026-09-29 两次调整后定稿：新增 128K/B24；去除 8K/B16；去除 8K/B40——
# 上线模板 --max-num-seqs 32 是一步最多调度的请求数，B40 是 40 个请求（240 token），
# 超出该上限，这种形状生产里不会出现（--max-num-batched-tokens 400 不构成约束）。
# 因此 8K/B32（192 token）就是上线能达到的最大一步。
# 每档跑 run_hca_reference_pair.sh：先基线、再候选，候选对基线快照做逐 bit 门禁并计时。
# 本机 task-submit 只接受“一整条命令字符串”，不能按 argv 传，否则参数会被它当成自己的选项。
# 用法：submit_hca_tiers.sh <结果目录> <基线源码> <候选源码>
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
for tier in 131072:4 131072:8 131072:16 131072:24 8192:24 8192:32; do
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
