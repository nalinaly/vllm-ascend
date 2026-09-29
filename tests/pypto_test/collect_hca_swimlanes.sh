#!/usr/bin/env bash
# 把七档 PTO 泳道整理成一个可下载的文件夹，文件名带档位与算子版本。
#
# 用法: collect_hca_swimlanes.sh <泳道结果根目录> <输出文件夹> [版本标签]
# 期望结构: <根>/h{history}_b{batch}/dfx/{merged_swimlane.json,chip_swimlane_records.json,...}
set -eo pipefail
src="$(realpath "${1:?指定泳道结果根目录}")"
dst="$(realpath -m "${2:?指定输出文件夹}")"
tag="${3:-pto}"
mkdir -p "$dst"
printf '%s\n' "档位,算子版本,merged_swimlane,critical_path_report,makespan_ms" > "$dst/index.csv"
for tier in 131072_b4 131072_b8 131072_b16 131072_b24 8192_b16 8192_b24 8192_b32; do
  d="$src/h$tier/dfx"
  [ -d "$d" ] || { printf '缺 %s\n' "$d" >&2; continue; }
  hist="${tier%%_b*}"; batch="${tier##*_b}"
  # 档位名统一成 128K / 8K，便于阅读
  case "$hist" in 131072) label="128K";; 8192) label="8K";; *) label="${hist}";; esac
  base="hca_${tag}_${label}_B${batch}"
  for f in merged_swimlane.json chip_swimlane_records.json deps.json critical_path_report.md name_map.json; do
    [ -f "$d/$f" ] && cp "$d/$f" "$dst/${base}__${f}"
  done
  ms=""
  if [ -f "$d/critical_path_report.md" ]; then
    ms="$(grep -o 'makespan\*\*: [0-9.]*' "$d/critical_path_report.md" | head -n1 | awk '{print $2}')"
  fi
  printf '%s,%s,%s,%s,%s\n' "$label/B$batch" "$tag" "${base}__merged_swimlane.json" \
    "${base}__critical_path_report.md" "${ms:-}" >> "$dst/index.csv"
  printf '已收 %s\n' "$label/B$batch"
done
cat > "$dst/README.md" <<'MD'
# HCA PTO 七档泳道

- `hca_pto_<档位>_B<batch>__merged_swimlane.json`：Perfetto 可直接打开
  （ui.perfetto.dev → Open trace file）。**只看 pid 4 的 Worker View**，
  那是 incore task；pid 2 是 AICPU Scheduler、pid 1 是 Orchestrator。
- `hca_pto_..._chip_swimlane_records.json`：原始记录，含每块的
  `local_setup_us`（= start − receive，核收到任务到开始执行的等待）与 `kernel-duration-us`。
- `hca_pto_..._critical_path_report.md`：`simpler_setup.tools.critical_path` 重建的
  官方关键路径，含静态 CPM 下限、观测路径的 compute/stall 拆分。
- `index.csv`：档位与 makespan 对照。

采集口径：PTO 侧、`--super-kernel 0`（PTO 结构上用不了 SuperKernel）、
level-4 chip swimlane、`begin_dfx`/`end_dfx` 包一次图重放后导出。
泳道为诊断用，采样数很小，**其计时数字不能当性能数据**；
性能数字一律看 `report.json` 的 `device_span_us`（纯设备耗时）。
MD
printf '输出: %s\n' "$dst"
