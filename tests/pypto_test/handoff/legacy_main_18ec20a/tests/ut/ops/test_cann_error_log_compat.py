# SPDX-License-Identifier: Apache-2.0
"""Compile the CANN diagnostic backport without requiring NPU hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("sdk_has_helpers", [False, True])
def test_diagnostics_survive_opdev_macro_redefinition(tmp_path, sdk_has_helpers):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the diagnostic header regression")
    include = tmp_path / "log"
    include.mkdir()
    # Model the op-common logging API and opdev's incompatible OP_LOGE signature.
    # The real CANN headers are additionally exercised by the native package build.
    (include / "log.h").write_text(
        r"""
#pragma once
#include <cstdarg>
#include <cstdio>
#include <string>
#include <type_traits>
constexpr int OP = 0;
constexpr int DLOG_ERROR = 3;
inline int CheckLogLevel(int, int) { return 1; }
inline void DlogRecord(int, int, const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    std::putchar('\n');
    va_end(args);
}
namespace Ops::Base {
inline std::string GetOpInfo(const std::string& name) { return name; }
}
#define REPORT_INNER_ERR_MSG(code, ...) DlogRecord(OP, DLOG_ERROR, __VA_ARGS__)
#define OP_LOGE(opName, ...) DlogRecord(OP, DLOG_ERROR, __VA_ARGS__)
"""
    )
    source = tmp_path / "diagnostics.cpp"
    sdk_helper = ""
    if sdk_has_helpers:
        sdk_helper = """
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, argument, value, reason) \
    DlogRecord(OP, DLOG_ERROR, "SDK helper preserved")
"""
    source.write_text(
        sdk_helper
        + """
#include "cann_error_log_compat.h"
#undef OP_LOGE
#define OP_LOGE(errorCode, ...) \
    static_assert(std::is_integral_v<decltype(errorCode)>, "opdev expects an error code")
int main() {
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("Indexer", "layout", "must be TND");
    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("Indexer", "batch", "0", "must be positive");
    OP_LOGE_FOR_INVALID_SHAPEDIM("Indexer", "head_dim", "64", "128");
}
"""
    )
    repo = Path(__file__).resolve().parents[3]
    executable = tmp_path / "diagnostics"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Werror",
            f"-I{tmp_path}",
            f"-I{repo / 'csrc/common/include'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout
    assert output.count("OpName:[Indexer] Invalid layout=: must be TND") == 2
    assert output.count("OpName:[Indexer] Invalid head_dim=64: expected 128") == 2
    if sdk_has_helpers:
        assert output.count("SDK helper preserved") == 1
        assert "Invalid batch" not in output
    else:
        assert output.count("OpName:[Indexer] Invalid batch=0: must be positive") == 2
