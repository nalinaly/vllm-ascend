# SPDX-License-Identifier: Apache-2.0
"""为已验收的94分支生成独立ACLNN工程；Host固定一份容量布局，设备选择策略。"""

import argparse
import importlib.util
import json
import re
import shutil
from pathlib import Path


def simplify_torch_run(source: str) -> str:
    """运行时保留TensorList参数数目检查；完整描述符验证仅用于Meta推导。"""
    old = " Check(a,variant); const c10_npu::OptionalNPUGuard guard(a[0].device());"
    new = (
        ' TORCH_CHECK(a.size()==55,"Expected 55 CSA entry buffers");\n'
        " const c10_npu::OptionalNPUGuard guard(a[0].device());"
    )
    assert source.count(old) == 1, "ACLNN Torch生成模板的Run入口已变化"
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src, ref, out = args.artifact.resolve(), args.reference.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for directory in (
        "include",
        "generated",
        "op_host",
        "op_kernel",
        "torch",
        "scripts",
        "aclnn_test",
        "facade/kernels/aic",
        "facade/kernels/aiv",
        "facade/orchestration",
        "objects",
        "logs",
    ):
        (out / directory).mkdir(parents=True, exist_ok=True)
    for name in (
        "CMakeLists.txt",
        "op_host/CMakeLists.txt",
        "op_kernel/CMakeLists.txt",
        "include/workspace.h",
        "include/pto_adapter.h",
        "include/contract.h",
    ):
        shutil.copy2(ref / name, out / name)
    repo = Path(__file__).resolve().parents[3]
    records = json.loads((repo / "tests/pypto_test/csa_policy_compile_20261027/compile_results.json").read_text())[
        "results"
    ]
    records = sorted((r for r in records if r["mode"] == "spmd"), key=lambda r: r["index"])
    envelope = Path(records[78]["artifact"])
    facade = out / "facade"
    shutil.copy2(src / "compiled_meta.json", facade / "compiled_meta.json")
    shutil.copy2(src / "orchestration/decode_csa_spmd.cpp", facade / "orchestration/decode_csa_spmd.cpp")
    orchestration = facade / "orchestration/decode_csa_spmd.cpp"
    orchestration.write_text(orchestration.read_text().replace("policy_state_shape", "policy_state_ci_shapes"))
    for role, symbol in (("aic", "cube_schedule"), ("aiv", "vector_schedule")):
        shutil.copy2(envelope / f"kernels/{role}/{symbol}.cpp", facade / f"kernels/{role}/{symbol}.cpp")
    (out / "generated/manifest.json").write_text(
        json.dumps({"variants": [dict(tiling_key=0, source_build=str(facade), cpp="unused.cpp")]})
    )
    # 复用既有ACLNN生成器的Host描述符/torch注册部分；设备计算入口在下方替换。
    adapter = (ref / "scripts/generate_adapter.py").read_text()
    for old, new in (
        ("decode_csa_spmd_debug.cpp", "decode_csa_spmd.cpp"),
        ("assert len(payload) == 133", "assert len(payload) == 134"),
        ("assert len(names) == len(payload)", "assert len(names) + 1 == len(payload)"),
        ('re.match(r"const Tensor& \\w+ = alloc_", t)', 're.match(r"const Tensor& \\w+ = .*get_ref", t)'),
        ("len(outputs) == 8", "len(outputs) == 6"),
        ("ext[57]", "ext[55]"),
        ("a.size()==57", "a.size()==55"),
        ("Expected 57 diagnostic entry buffers", "Expected 55 CSA entry buffers"),
        ("133, offsets", "134, offsets"),
        ("133, sources", "134, sources"),
    ):
        assert old in adapter, old
        adapter = adapter.replace(old, new)
    script = out / "scripts/generate_host.py"
    script.write_text(adapter)
    spec = importlib.util.spec_from_file_location("csa_device_host_generator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    workspace = out / "include/workspace.h"
    workspace.write_text(
        workspace.read_text().replace("offsets[133]", "offsets[134]").replace("sources[133]", "sources[134]")
    )
    contract = out / "include/contract.h"
    contract.write_text(
        contract.read_text().replace("{10, 11, 14, 21, 24, 37, 50, 51, 56}", "{10, 11, 14, 21, 24, 37, 54}")
    )

    payload = json.loads((src / "cpp_manifest.json").read_text())["payload_names"]
    tasks = []
    for row in records:
        i = row["index"]
        shutil.copy2(src / f"generated/p{i:03d}.cpp", out / f"generated/p{i:03d}.cpp")
        for role, symbol in (("aic", "cube_schedule"), ("aiv", "vector_schedule")):
            wrapper = (Path(row["artifact"]) / f"kernels/{role}/{symbol}.cpp").read_text().split('extern "C"')[-1]
            names = re.findall(r"// Unpack tensor: (\w+)", wrapper)
            assert names == payload[role]
            types = {name: dtype for dtype, name in re.findall(r"__gm__ (\w+)\* (\w+) = reinterpret_cast", wrapper)}
            scalars = re.findall(r"^    int64_t (\w+) = (.*);$", wrapper, re.M)
            assert len(scalars) == 28
            lines = [
                "#include <cstdint>",
                "#ifndef __aicore__",
                "#define __aicore__ [aicore]",
                "#endif",
                "#include <pto/pto-inst.hpp>",
                "using namespace pto;",
                f"namespace csa_math_{role}_{i:03d} {{",
                f'#include "p{i:03d}.cpp"',
                "}",
                f'extern "C" __aicore__ __attribute__((noinline)) void csa_{role}_{i:03d}(',
                "__gm__ uint8_t** ptrs, const int64_t* scalars, int32_t block, int32_t subblock) {",
            ]
            lines += [
                f"auto* {name} = reinterpret_cast<__gm__ {types[name]}*>(ptrs[{n}]);" for n, name in enumerate(names)
            ]
            for n, (name, _) in enumerate(scalars):
                adjustment = (
                    " - scalars[0]"
                    if name.startswith("group_offset_words") and not row["policy"]["FUSED_INDEXER_COEFF"]
                    else ""
                )
                lines.append(f"int64_t {name} = scalars[{n}]{adjustment};")
            call = re.findall(r"\b" + symbol + r"\(([^;]+)\);", wrapper)[-1]
            call = (
                call.replace("__pypto_spmd_block_idx", "block")
                .replace("__pypto_spmd_block_num", "24")
                .replace("__pypto_spmd_subblock_idx", "subblock")
            )
            lines += [f"csa_math_{role}_{i:03d}::{symbol}({call});", "}"]
            path = out / f"generated/{role}_{i:03d}.cpp"
            path.write_text("\n".join(lines) + "\n")
            tasks.append(dict(role=role, index=i, source=str(path), output=str(out / f"objects/{role}_{i:03d}.o")))

    # CANN kernel保留生成器给出的55参数ABI；只换为统一设备选择与134指针布局。
    kernel = (out / "op_kernel/csa_spmd_tnd_pto.cpp").read_text()
    start = kernel.index('extern "C" __global__')
    body = kernel[start : kernel.index("set_atomic_none();", start)]
    body = body.replace("ptrs[133]", "ptrs[134]").replace("i<133", "i<134")
    ids = {name.removesuffix("__ssa_v0"): i for i, name in enumerate(payload["aiv"])}
    prefix = [
        '#include "kernel_operator.h"',
        '#include "lib/matmul_intf.h"',
        '#include "../include/pto_adapter.h"',
        "#define CSA_DEVICE __aicore__ __attribute__((always_inline))",
        '#include "../include/policy_select.h"',
        "using namespace pto;",
    ]
    for role, macro in (("aic", "__DAV_CUBE__"), ("aiv", "__DAV_VEC__")):
        prefix.append(f"#if defined({macro})")
        prefix += [
            f'extern "C" __aicore__ void csa_{role}_{i:03d}(__gm__ uint8_t**, const int64_t*, int32_t, int32_t);'
            for i in range(94)
        ]
        prefix.append("#endif")
    logic = f"""set_atomic_none();
const int block = CsaGroupIdx();
auto* state = reinterpret_cast<__gm__ int32_t*>(ptrs[133]);
#if defined(__DAV_VEC__)
set_mask_norm(); set_vector_mask(-1, -1);
const int lane = block * 2 + get_subblockid();
Tile<TileType::Vec, int32_t, 1, 11520> zero;
TASSIGN(zero, 0); TEXPANDS(zero, int32_t(0));
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
GlobalTensor<int32_t, Shape<1,1,1,1,11520>, Stride<11520,11520,11520,11520,1>> target(
reinterpret_cast<__gm__ int32_t*>(ptrs[{ids["signal_init_matrix"]}]) + lane * 11520);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); TSTORE(target, zero);
if (lane == 0) {{
auto* lengths = reinterpret_cast<__gm__ int32_t*>(ptrs[{ids["idx_lengths"]}]);
auto* bounds = reinterpret_cast<__gm__ int32_t*>(ptrs[{ids["idx_bounds"]}]);
int64_t q=1, l=8192;
for (int r=0;r<data.scalars[5] && r+1<data.scalars[4];++r) {{
const int64_t begin=bounds[r], end=bounds[r+1], length=lengths[r];
if (length>0 && begin>=0 && end>begin && end<=data.scalars[0] && end-begin<=length) {{
q=q>end-begin?q:end-begin; l=l>length?l:length;
}}
}}
const int selected=csa_selector::select(data.scalars[0],data.scalars[5],q,l,data.scalars[27]);
dcci((__gm__ void*)state, cache_line_t::SINGLE_CACHE_LINE);
*reinterpret_cast<volatile __gm__ int32_t*>(state)=selected;
dcci((__gm__ void*)state, cache_line_t::SINGLE_CACHE_LINE);
}}
#endif
pipe_barrier(PIPE_ALL); dsb(DSB_DDR); SYNCALL<SyncCoreType::Mix>();
dcci((__gm__ void*)state, cache_line_t::SINGLE_CACHE_LINE);
const int selected=*reinterpret_cast<volatile __gm__ int32_t*>(state);
"""
    for role, macro in (("aic", "__DAV_CUBE__"), ("aiv", "__DAV_VEC__")):
        logic += f"#if defined({macro})\n"
        lane = "get_subblockid()" if role == "aiv" else "0"
        logic += (
            "\n".join(
                f"{'if' if i == 0 else 'else if'} (selected=={i}) csa_{role}_{i:03d}(ptrs,data.scalars,block,{lane});"
                for i in range(94)
            )
            + "\n#endif\n"
        )
    (out / "op_kernel/csa_spmd_tnd_pto.cpp").write_text("\n".join(prefix) + "\n" + body + logic + "}\n")
    shutil.copy2(src / "policy_select.h", out / "include/policy_select.h")
    # 独立op/vendor/torch名字，避免替换其它session的旧62档安装。
    for path in [
        out / "CMakeLists.txt",
        out / "op_host/CMakeLists.txt",
        out / "op_host/csa_spmd_tnd_pto.cpp",
        out / "op_kernel/csa_spmd_tnd_pto.cpp",
        out / "torch/binding.cpp",
    ]:
        text = (
            path.read_text()
            .replace("CsaSpmdTndPto", "CsaSpmdTndDevicePto")
            .replace("csa_spmd_tnd_pto", "csa_spmd_tnd_device_pto")
            .replace("TORCH_LIBRARY(csa_pto,", "TORCH_LIBRARY(csa_pto_device,")
            .replace("TORCH_LIBRARY_IMPL(csa_pto,", "TORCH_LIBRARY_IMPL(csa_pto_device,")
        )
        path.write_text(text)
    (out / "op_host/csa_spmd_tnd_pto.cpp").rename(out / "op_host/csa_spmd_tnd_device_pto.cpp")
    (out / "op_kernel/csa_spmd_tnd_pto.cpp").rename(out / "op_kernel/csa_spmd_tnd_device_pto.cpp")
    torch_binding = out / "torch/binding.cpp"
    torch_binding.write_text(simplify_torch_run(torch_binding.read_text()))
    (out / "build.json").write_text(json.dumps(dict(completed=0, failed=0, results=[])))
    (out / "cpp_manifest.json").write_text(
        json.dumps(dict(tasks=tasks, policy_mode="device", policies=94, external_tensors=55), indent=2)
    )
    shutil.copy2(src / "compile_flags.json", out / "compile_flags.json")
    print(out)


if __name__ == "__main__":
    main()
