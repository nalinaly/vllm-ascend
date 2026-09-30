#!/usr/bin/env bash
# 验收口径：在**同一张卡**上交替跑 Native 与 PTO，按 ABBA 顺序，算出可信的比值。
#
# 为什么必须同卡：`task-submit --device auto` 每个任务拿到的卡不同，同配置换卡实测
# 差 45 μs（2026-09-29）。此前七档基线里 Native 与 PTO 分属不同任务、不同卡，
# 比值带着跨卡差异，量级足以盖过 20 μs 的改动。
#
# Native 开 SuperKernel（它最好的形态），PTO 结构上开不了、恒为 0。
# kernel 侧的 NZ 档由 NZ_MODE 给出，同时写进 --weight-nz-mode，两侧一致。
#
# 用法: [CYCLES=n] [NZ_MODE=2] [PTO_SOURCE=<目录>] run_hca_sides_same_card.sh <结果目录> <history> <batch>
set -eo pipefail
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sd_output="$(realpath -m "${1:?指定结果目录}")"
sd_history="${2:?指定 history}"
sd_batch="${3:?指定 batch}"
shift 3
sd_cycles="${CYCLES:-1}"
sd_nz="${NZ_MODE:-2}"
mkdir -p "$sd_output"
printf '卡: %s  档位: %s/%s  NZ档: %s  轮数: %s  PTO源: %s\n' \
  "${TASK_DEVICE:-?}" "$sd_history" "$sd_batch" "$sd_nz" "$sd_cycles" "${PTO_SOURCE:-生产版}" \
  | tee "$sd_output/plan.txt"

sd_order=()
for ((c = 0; c < sd_cycles; c++)); do sd_order+=(native pto pto native); done

pass=0
for side in "${sd_order[@]}"; do
  pass=$((pass + 1))
  target="$sd_output/p${pass}_${side}"
  # Native 开 SK，PTO 恒 0（见 hca-gap-uses-native-with-superkernel）。
  if [ "$side" = native ]; then sk=1; src=(); else sk=0
    src=(); [ -n "${PTO_SOURCE:-}" ] && src=(--operator-source "$PTO_SOURCE")
  fi
  printf '=== pass %s: %s ===\n' "$pass" "$side"
  env "VLLM_ASCEND_ENABLE_NZ=$sd_nz" bash "$hca_tests/run_hca_compiled_case.sh" "$target" "$side" \
    --history "$sd_history" --batch "$sd_batch" --super-kernel "$sk" \
    --weight-nz-mode "$sd_nz" "${src[@]}" "$@" \
    || printf 'pass %s (%s) 失败\n' "$pass" "$side" >&2
done
printf '完成，结果在 %s\n' "$sd_output"
