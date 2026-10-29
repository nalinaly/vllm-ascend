# SPDX-License-Identifier: Apache-2.0
"""原62策略ACLNN与当前TND binding的CPU合同、传参和失败原子性检查。"""

import copy
import hashlib
import json
import platform
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch_npu

from vllm_ascend import envs
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd import backend
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy

pytestmark = pytest.mark.skipif(envs.VLLM_ASCEND_ENABLE_NZ != 2, reason="原始CSA ACLNN包固定为NZ2")
_FIXTURES = Path(__file__).with_name("fixtures") / "csa_aclnn_caed2c32"
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}
_NATIVE_NAMES = {
    "hc_fn": "hc_attn_fn",
    "hc_scale": "hc_attn_scale",
    "hc_base": "hc_attn_base",
    "norm_w": "attn_norm_w",
    "wq_a": "wq_a",
    "gamma_cq": "gamma_cq",
    "wq_b": "wq_b",
    "wq_b_scale": "wq_b_scale",
    "rope_cos": "freqs_cos",
    "rope_sin": "freqs_sin",
    "wkv": "wkv",
    "gamma_ckv": "gamma_ckv",
    "native_slots": "ori_slot_mapping",
    "main_wkv": "cmp_wkv",
    "main_wgate": "cmp_wgate",
    "main_state": "compress_state",
    "main_state_table": "state_block_table",
    "main_ape": "cmp_ape",
    "positions": "position_ids",
    "main_bounds": "cmp_query_start_loc",
    "main_lengths": "cmp_seq_lens",
    "main_state_slots": "state_slot_mapping",
    "main_gamma": "cmp_norm_w",
    "main_cos": "cmp_freqs_cos",
    "main_sin": "cmp_freqs_sin",
    "main_slots": "cmp_slot_mapping",
    "idx_wkv": "inner_wkv",
    "idx_wgate": "inner_wgate",
    "idx_state": "inner_compress_state",
    "idx_state_table": "inner_state_block_table",
    "idx_ape": "inner_ape",
    "idx_bounds": "idx_query_start_loc",
    "idx_lengths": "kv_seq_lens",
    "idx_state_slots": "inner_state_slot_mapping",
    "idx_gamma": "inner_norm_w",
    "idx_cos": "inner_freqs_cos",
    "idx_sin": "inner_freqs_sin",
    "idx_slots": "idx_slot_mapping",
    "hadamard": "hadamard_idx",
    "idx_cache": "idx_native_kv_cache",
    "original_table": "ori_block_table",
    "compressed_table": "cmp_block_table",
    "indexer_table": "idx_block_table",
    "idx_wq_b": "idx_wq_b",
    "idx_wq_b_scale": "idx_wq_b_scale",
    "weights_proj": "weights_proj",
    "wo_a": "wo_a",
    "wo_b": "wo_b",
    "wo_b_scale": "wo_b_scale",
}


def reference_interface():
    return json.loads((_FIXTURES / "interface.json").read_text())["params"]


def reference_variants():
    return json.loads((_FIXTURES / "manifest.json").read_text())["variants"]


def tensor_buffers(*, tokens=2, requests=1):
    """保持真实静态权重形状；CPU empty只提供描述符，测试不读取权重内容。"""
    dynamic_shapes = {
        "x": (tokens, 16384),
        "output": (tokens, 16384),
        "positions": (tokens,),
        "rope_cos": (tokens, 64),
        "rope_sin": (tokens, 64),
        "native_slots": (tokens, 2),
        "main_bounds": (requests + 1,),
        "idx_bounds": (requests + 1,),
        "main_lengths": (requests,),
        "idx_lengths": (requests,),
        "main_state_slots": (tokens, 2),
        "idx_state_slots": (tokens, 2),
        "main_state_table": (requests, 2),
        "idx_state_table": (requests, 2),
        "original_table": (requests, 32),
        "compressed_table": (requests, 32),
        "indexer_table": (requests, 32),
        "main_state": (1, 8192),
        "idx_state": (1, 1040),
        "idx_cache": (1, 4160),
        "cache_rows": (32, 512),
        "main_cache": (32, 512),
        "topk_scores": (tokens, 512),
        "topk_indices": (tokens, 512),
    }
    return {
        parameter["name"]: torch.empty(
            dynamic_shapes.get(parameter["name"], tuple(1 if dim == -1 else dim for dim in parameter["shape"])),
            dtype=_DTYPES[parameter["dtype"]],
        )
        for parameter in reference_interface()
    }


def native_inputs(*, tokens=2, requests=1, diagnostic=False):
    buffers = tensor_buffers(tokens=tokens, requests=requests)
    native = {native_name: buffers[root_name] for root_name, native_name in _NATIVE_NAMES.items()}
    native.update(
        x_hc=buffers["x"].view(tokens, 4, 4096),
        x_out=buffers["output"].view(tokens, 4, 4096),
        kv_cache=buffers["cache_rows"].view(1, 32, 1, 512),
        cmp_kv=buffers["main_cache"].view(1, 32, 1, 512),
        attn_sink=buffers["attn_sink"].view(64),
    )
    if diagnostic:
        native.update(idx_topk=buffers["topk_indices"], idx_topk_scores=buffers["topk_scores"])
    return native


@pytest.fixture
def package_root(tmp_path):
    root = tmp_path / "csa_pto_ops"
    (root / "generated").mkdir(parents=True)
    (root / "lib").mkdir()
    for name in ("interface.json", "manifest.json"):
        shutil.copyfile(_FIXTURES / name, root / "generated" / name)
    library = root / "lib/csa_pto_npu.so"
    archive = root / "csa_spmd_tnd_pto.run"
    library.write_bytes(b"CPU unit-test library placeholder; never loaded")
    archive.write_bytes(b"CPU unit-test OPP placeholder; never executed")
    metadata = {
        "upstream_commit": "1685da6914ae6196537d7f1a66e84e09a9c48ae5",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "python": sys.version,
        "architecture": platform.machine(),
        "torch_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "run_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (root / "package_manifest.json").write_text(json.dumps(metadata))
    return root


@pytest.fixture
def kernel_package(package_root, monkeypatch):
    package = backend._read_package(package_root)
    monkeypatch.setattr(backend, "_load_kernel_package", lambda: package)
    monkeypatch.setenv("PTO_CSA_VARIANT", "spmd_tnd")
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", "aclnn")
    return package


def test_current_root_abi_matches_unmodified_reference_interface():
    expected = reference_interface()
    actual = backend._root_contract()
    assert len(actual) == 57
    assert actual == expected
    backend._check_root_contract(expected)
    assert {entry["name"] for entry in actual if entry["direction"] == "InOut"} == {
        "cache_rows",
        "main_state",
        "main_cache",
        "idx_state",
        "idx_cache",
        "topk_scores",
        "topk_indices",
        "output",
    }


def test_device_package_uses_normal_abi_and_ignores_host_policy(kernel_package, monkeypatch):
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd import binding

    parameters = tuple(p for p in kernel_package.parameters if p["name"] not in ("topk_scores", "topk_indices"))
    package = replace(kernel_package, parameters=parameters, policies={}, device_policy=True)
    monkeypatch.setattr(backend, "_load_kernel_package", lambda: package)

    def forbidden_host_policy(**kwargs):
        pytest.fail("设备选档后端不应调用Host策略选择")

    monkeypatch.setattr(binding, "select_policy", forbidden_host_policy)
    calls = []
    monkeypatch.setattr(torch.ops.csa_pto_device, "run", lambda *args: calls.append(args), raising=False)
    native = native_inputs()
    short = PreparedDynamicCSA(native, max_query_tokens=2, max_seq_len=8192)
    long = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=131072)
    assert short.policy == long.policy and short.policy.constexprs() == {}
    assert not short.allocations and not long.allocations
    assert short() is native["x_out"]
    assert len(calls[0][0]) == 55 and calls[0][1] == 0
    backend._check_root_contract(list(parameters), diagnostic=False)


def test_device_package_rejects_unwritten_diagnostic_outputs(kernel_package, monkeypatch):
    package = replace(kernel_package, device_policy=True)
    monkeypatch.setattr(backend, "_load_kernel_package", lambda: package)
    with pytest.raises(NotImplementedError, match="不暴露诊断Top-K"):
        PreparedDynamicCSA(native_inputs(diagnostic=True), max_query_tokens=2, max_seq_len=8192)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "renamed_weight"),
        ("direction", "InOut"),
        ("dtype", "fp32"),
        ("layout", "ND"),
        ("shape", [4096, 512]),
    ],
)
def test_abi_drift_is_rejected(field, value):
    expected = copy.deepcopy(reference_interface())
    next(entry for entry in expected if entry["name"] == "wkv")[field] = value
    with pytest.raises(RuntimeError, match="根 ABI 不匹配"):
        backend._check_root_contract(expected)


def test_parameter_order_is_part_of_abi():
    expected = reference_interface()
    expected[0], expected[1] = expected[1], expected[0]
    with pytest.raises(RuntimeError, match="根 ABI 不匹配"):
        backend._check_root_contract(expected)


def test_original_manifest_has_all_62_exact_policy_keys(package_root):
    package = backend._read_package(package_root)
    variants = reference_variants()
    assert len(package.policies) == 62
    assert set(package.policies.values()) == set(range(62))
    for item in variants:
        assert package.policies[backend._policy_key(item["policy"])] == item["tiling_key"]
    current = select_policy(tokens=96, requests=16, max_query_tokens=6, max_seq_len=131078)
    assert package.policies[backend._policy_key(current.constexprs())] == 3


@pytest.mark.parametrize(
    "field,bad",
    [
        ("upstream_commit", "0" * 40),
        ("torch", "unmatched-torch"),
        ("torch_npu", "unmatched-torch-npu"),
        ("python", "0.0.0"),
        ("architecture", "unmatched-architecture"),
    ],
)
def test_package_build_identity_mismatch_fails(package_root, field, bad):
    path = package_root / "package_manifest.json"
    metadata = json.loads(path.read_text())
    metadata[field] = bad
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="来源|ABI|架构"):
        backend._read_package(package_root)


@pytest.mark.parametrize(
    "relative",
    [
        "generated/interface.json",
        "generated/manifest.json",
        "lib/csa_pto_npu.so",
        "csa_spmd_tnd_pto.run",
    ],
)
def test_payload_change_is_not_accepted_by_reusing_metadata(package_root, relative):
    path = package_root / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="原始 62 档包|清单不一致"):
        backend._read_package(package_root)


def test_missing_package_file_reports_incomplete_wheel(package_root):
    (package_root / "lib/csa_pto_npu.so").unlink()
    with pytest.raises(RuntimeError, match="内容不完整"):
        backend._read_package(package_root)


def test_policy_keys_preserve_types_and_all_fields():
    original = reference_variants()[3]["policy"]
    changed = dict(original, PREFETCH=True)
    assert original["PREFETCH"] == 1
    assert backend._policy_key(original) != backend._policy_key(changed)
    assert backend._policy_key(original) != backend._policy_key(dict(original, NEW_CONSTEXPR=False))


def test_new_padding_policy_is_rejected_as_missing_strategy_not_as_illegal_tnd(kernel_package):
    # 该容量在当前TND允许；原62策略无法覆盖B1的大容量组合。
    policy = select_policy(tokens=32, requests=1, max_query_tokens=6, max_seq_len=131078)
    assert backend._policy_key(policy.constexprs()) not in kernel_package.policies
    prepared = NS(policy=policy, buffers={}, allocations={})
    with pytest.raises(NotImplementedError, match="没有匹配的编译策略"):
        backend.AclnnExecutor(prepared)
    assert prepared.allocations == {}


def test_supported_padding_keeps_current_tokens_capacity_contract(kernel_package):
    native = native_inputs(tokens=32, requests=4)
    prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=131078)
    assert prepared.tokens > prepared.requests * 6
    assert prepared.execution_backend == "aclnn"
    assert prepared._executor.variant in range(62)


@pytest.mark.parametrize("diagnostic", [False, True])
def test_executor_uses_reference_order_and_preserves_topk_ownership(kernel_package, diagnostic):
    native = native_inputs(diagnostic=diagnostic)
    prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=131078)
    executor = prepared._executor
    assert len(executor.arguments) == 57
    for parameter, argument in zip(reference_interface(), executor.arguments):
        name = parameter["name"]
        source = executor.auxiliary[name] if name in executor.auxiliary else prepared.buffers[name]
        assert argument is source
    if diagnostic:
        assert executor.auxiliary == {} and prepared.allocations == {}
        assert executor.arguments[50] is native["idx_topk_scores"]
        assert executor.arguments[51] is native["idx_topk"]
    else:
        assert set(prepared.allocations) == {"aclnn_topk_scores", "aclnn_topk_indices"}
        assert executor.auxiliary["topk_scores"].shape == (2, 512)
        assert executor.auxiliary["topk_scores"].dtype == torch.float32
        assert executor.auxiliary["topk_indices"].dtype == torch.int32
        assert "topk_scores" not in prepared.buffers and "topk_indices" not in prepared.buffers


def test_acl_dispatch_mutates_shared_outputs_without_executing_pypto(kernel_package, monkeypatch):
    import pypto.torch

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd import entry

    def forbidden(*args, **kwargs):
        raise AssertionError("ACLNN路径不得初始化Simpler或执行PyPTO根函数")

    monkeypatch.setattr(pypto.torch, "init", forbidden)
    monkeypatch.setattr(entry, "decode_csa_spmd", forbidden)
    monkeypatch.setattr(entry, "decode_csa_spmd_debug", forbidden)
    native = native_inputs()
    native["x_hc"].fill_(3)
    native["x_out"].fill_(-1)
    prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=131078)

    def simulate_run(arguments, variant):
        values = dict(zip([entry["name"] for entry in reference_interface()], arguments))
        assert variant == prepared._executor.variant
        values["output"].copy_(values["x"] + 1)
        values["main_state"].fill_(7)
        values["topk_scores"].fill_(0.25)
        values["topk_indices"].fill_(12)

    monkeypatch.setattr(torch.ops.csa_pto, "run", simulate_run, raising=False)
    # 运行期环境变化不能将已准备的ACLNN对象偷偷切回Python。
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_BACKEND", "python")
    output = prepared()
    assert output is native["x_out"] and torch.all(output == 4)
    assert torch.all(native["compress_state"] == 7)
    assert torch.all(prepared.allocations["aclnn_topk_scores"] == 0.25)
    assert torch.all(prepared.allocations["aclnn_topk_indices"] == 12)


def test_auxiliary_storage_survives_successful_rebind(kernel_package):
    native = native_inputs()
    prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=131078)
    auxiliary = dict(prepared._executor.auxiliary)
    replaced = dict(native, x_hc=torch.empty_like(native["x_hc"]), x_out=torch.empty_like(native["x_out"]))
    prepared.rebind(replaced)
    assert prepared.native["x_out"] is replaced["x_out"]
    assert prepared._executor.arguments[0].data_ptr() == replaced["x_hc"].data_ptr()
    assert prepared._executor.arguments[-1].data_ptr() == replaced["x_out"].data_ptr()
    for name, value in auxiliary.items():
        assert prepared._executor.auxiliary[name] is value
        assert prepared.allocations["aclnn_" + name] is value


def test_bad_rebind_does_not_change_native_buffers_or_arguments(kernel_package):
    native = native_inputs()
    prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=131078)
    old_native, old_arguments = prepared.native, prepared._executor.arguments
    old_buffers = dict(prepared.buffers)
    replaced = dict(native, freqs_cos=native["freqs_cos"].to(torch.float16))
    with pytest.raises(ValueError, match="描述符变化"):
        prepared.rebind(replaced)
    assert prepared.native is old_native and prepared._executor.arguments is old_arguments
    assert all(prepared.buffers[name] is value for name, value in old_buffers.items())


def test_backend_rebind_failure_does_not_commit_shared_binding(kernel_package, monkeypatch):
    native = native_inputs()
    prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=131078)
    old_native, old_arguments = prepared.native, prepared._executor.arguments
    old_buffers = dict(prepared.buffers)
    replacement = dict(native, x_out=torch.empty_like(native["x_out"]))

    def reject_backend_contract(buffers):
        assert buffers["output"].data_ptr() == replacement["x_out"].data_ptr()
        raise ValueError("backend format mismatch after shared descriptor validation")

    monkeypatch.setattr(prepared._executor, "rebind", reject_backend_contract)
    with pytest.raises(ValueError, match="backend format mismatch"):
        prepared.rebind(replacement)
    assert prepared.native is old_native and prepared._executor.arguments is old_arguments
    assert all(prepared.buffers[name] is value for name, value in old_buffers.items())


def test_failed_executor_preparation_does_not_publish_auxiliary_allocations(kernel_package):
    buffers = tensor_buffers()
    buffers.pop("topk_scores")
    buffers.pop("topk_indices")
    buffers["norm_w"] = buffers["norm_w"].float()
    existing = torch.empty(1)
    prepared = NS(
        tokens=2,
        buffers=buffers,
        allocations={"existing": existing},
        policy=NS(constexprs=lambda: reference_variants()[3]["policy"]),
    )
    with pytest.raises(ValueError, match="norm_w"):
        backend.AclnnExecutor(prepared)
    assert list(prepared.allocations) == ["existing"]
    assert prepared.allocations["existing"] is existing


@pytest.mark.parametrize("parameter,bad_format", [("wkv", 2), ("idx_wq_b", 2), ("weights_proj", 2), ("rope_cos", 29)])
def test_executor_format_validation_preserves_previous_arguments(kernel_package, monkeypatch, parameter, bad_format):
    prepared = PreparedDynamicCSA(native_inputs(), max_query_tokens=6, max_seq_len=131078)
    executor = prepared._executor
    old_arguments = executor.arguments
    device = NS(type="npu")
    all_values = {**prepared.buffers, **executor.auxiliary}
    descriptors = {
        value["name"]: NS(
            shape=all_values[value["name"]].shape,
            ndim=all_values[value["name"]].ndim,
            dtype=all_values[value["name"]].dtype,
            device=device,
            is_contiguous=lambda: True,
            format=29 if value["layout"] == "NZ" else 2,
        )
        for value in reference_interface()
    }
    descriptors[parameter].format = bad_format
    monkeypatch.setattr(torch_npu, "get_npu_format", lambda value: value.format)
    # 给该格式检查使用完整57参数视图，避免CPU辅助输出的device与模拟NPU冲突。
    monkeypatch.setattr(executor, "auxiliary", {})
    with pytest.raises(ValueError, match=parameter):
        executor.rebind(descriptors)
    assert executor.arguments is old_arguments


def test_pypto_run_config_is_rejected_before_aclnn_call(kernel_package, monkeypatch):
    native = native_inputs()
    native["x_out"].fill_(-3)
    prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=131078)

    def forbidden(*args):
        raise AssertionError("非法RunConfig不得触发执行")

    monkeypatch.setattr(torch.ops.csa_pto, "run", forbidden, raising=False)
    with pytest.raises(ValueError, match="不接受 PyPTO RunConfig"):
        prepared(config=object())
    assert torch.all(native["x_out"] == -3)
