"""Build isolated CANN 9.2 RMSNorm/rotary ABI corrections using prior native objects.

The checkout and original native-install artifact are unchanged. This is an
environment compatibility experiment, not a production source patch.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import regex as re


def main():
    repo = Path(__file__).resolve().parents[2]
    cache = repo / ".cache/csa"
    old_build = cache / "native-build"
    directory = cache / "native-cann92-build"
    install = cache / "native-cann92-install"
    directory.mkdir(exist_ok=True)
    install.mkdir(exist_ok=True)
    original = (repo / "csrc/torch_binding.cpp").read_text()
    before = (
        "    EXEC_NPU_CMD(aclnnRmsNormDynamicQuant, x, gamma, smooth_scale, smooth_scale2, beta, epsilon, "
        "output_mask, dst_type,\n                 y_out, y2_out, scale_out, scale2_out);"
    )
    after = """    // Local CANN 9.2 built-in ABI from include/aclnnop/aclnn_rms_norm_dynamic_quant.h.
    // The repository's custom-op ABI has two scales/outputs and is incompatible.
    int64_t cann92_dst_type = static_cast<int64_t>(ACL_INT8);
    EXEC_NPU_CMD(aclnnRmsNormDynamicQuant, x, gamma, smooth_scale, beta, epsilon,
                 cann92_dst_type, y_out, scale_out);"""
    if original.count(before) != 1:
        raise RuntimeError("source signature changed; inspect before applying the CANN ABI correction")
    modified = original.replace(before, after)
    rotary_before = "    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, it->second, partial_slice, negate_sin);"
    rotary_after = (
        '    TORCH_CHECK(!negate_sin, "CANN 9.2 compatibility path requires explicit negated sin tensor");\n'
        "    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, it->second, partial_slice);"
    )
    if modified.count(rotary_before) != 1:
        raise RuntimeError("rotary source signature changed")
    modified = modified.replace(rotary_before, rotary_after)
    source = directory / "torch_binding.cpp"
    source.write_text(modified)
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(True),
            modified.splitlines(True),
            fromfile="csrc/torch_binding.cpp",
            tofile="local-cann92/torch_binding.cpp",
        )
    )
    (directory / "compatibility.patch").write_text(patch)
    cmake_files = old_build / "CMakeFiles/vllm_ascend_C.dir"
    flag_text = (cmake_files / "flags.make").read_text()
    flags = []
    for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        flags.extend(shlex.split(re.search(rf"^{name} = (.*)$", flag_text, re.MULTILINE)[1]))
    obj = directory / "torch_binding.cpp.o"
    compile_cmd = ["/usr/bin/c++", *flags, f"-I{repo / 'csrc'}", "-c", str(source), "-o", str(obj)]
    link_cmd = shlex.split((cmake_files / "link.txt").read_text())
    output_position = link_cmd.index("-o") + 1
    artifact = install / Path(link_cmd[output_position]).name
    link_cmd[output_position] = str(artifact)
    old_obj = "CMakeFiles/vllm_ascend_C.dir/csrc/torch_binding.cpp.o"
    link_cmd[link_cmd.index(old_obj)] = str(obj)
    link_cmd = [argument.replace("\\$ORIGIN", "$ORIGIN") for argument in link_cmd]
    for name in ("libvllm_ascend_kernels.so", "lib"):
        src = cache / "native-install" / name
        dst = install / name
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=src.is_dir())
    commands = {"compile": compile_cmd, "link": link_cmd, "cwd": str(old_build)}
    (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    subprocess.run(compile_cmd, cwd=old_build, check=True)
    subprocess.run(link_cmd, cwd=old_build, check=True)
    report = {
        "status": "BUILD_PASS_RUNTIME_UNVERIFIED",
        "artifact": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "original_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "modified_source_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "patch": str(directory / "compatibility.patch"),
        "scope": "RmsNormDynamicQuant_and_InplacePartialRotaryMul_builtin_ABI_on_CANN_9.2",
    }
    (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
