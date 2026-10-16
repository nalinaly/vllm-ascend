#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
build="$(realpath "${1:?Usage: run_superkernel.sh BUILD_DIR NEW_OUTPUT_DIR [graph|superkernel] [EARLY_START=0|1]}")"
output="$(realpath -m "${2:?Pass a new output directory}")"
mode="${3:-superkernel}"
early_start="${4:-0}"
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
[[ "$mode" == graph || "$mode" == superkernel ]] || exit 2
[[ "$early_start" == 0 || ( "$early_start" == 1 && "$mode" == superkernel ) ]] || exit 2
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
[[ -f "$root/build/torch/hcpre_pto_torch.so" ]] || { echo "Build the Torch binding first" >&2; exit 2; }
packages=("$build"/custom_opp_*.run)
[[ ${#packages[@]} -eq 1 && -f "${packages[0]}" ]] || exit 2
mkdir -p "$output/environment/opp/built-in" "$output/environment/opp/static_kernel"

# Both the package installer and runtime use this private OPP root. CANN's
# layout detector requires a real built-in directory; its contents can be links.
original_opp="$(realpath "$ASCEND_HOME_PATH/opp")"
for item in "$original_opp"/*; do
    name="$(basename "$item")"
    [[ "$name" == built-in || "$name" == static_kernel ]] && continue
    ln -s "$item" "$output/environment/opp/$name"
done
for item in "$original_opp/built-in"/*; do
    ln -s "$item" "$output/environment/opp/built-in/$(basename "$item")"
done
ln -s "$(realpath "$ASCEND_HOME_PATH/$(uname -m)-linux")" "$output/environment/$(uname -m)-linux"
bash "${packages[0]}" --noexec --extract="$output/custom_opp" > "$output/extract.log" 2>&1
export ASCEND_OPP_PATH="$output/environment/opp"
export ASCEND_CUSTOM_OPP_PATH="$output/custom_opp/packages/vendors/hcpre_pto"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1
export ASCEND_PROCESS_LOG_PATH="$output/cann_logs"

cleanup() {
    # npugraph_ex searches ASCEND_HOME_PATH for uninstall scripts and cannot
    # discover our private root. Clean only packages installed by this run.
    local script
    for script in "$ASCEND_OPP_PATH"/static_kernel/ai_core/static_kernel_*/uninstall.sh; do
        [[ -f "$script" ]] || continue
        bash "$script" >> "$output/uninstall.log" 2>&1 || return 1
    done
}
trap cleanup EXIT
cd "$output"
python "$root/tests/test_superkernel.py" --mode "$mode" \
    --binding "$root/build/torch/hcpre_pto_torch.so" --output result --profile --early-start "$early_start"
