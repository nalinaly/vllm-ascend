#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
side=mix_m4_box8_d512_v3
bash "$here/run.sh" "$side" 8192 24
# B3/S6共18行：最后worker只有2行；同址补位3→2→1→3单独检查，非性能数据。
bash "$here/run.sh" "$side" 16507 3 --cycles 1 --padding-graph
