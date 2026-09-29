"""为有完整区间收益的主wgate补旧快照和两版根契约，不修改已排队私有源码。"""
import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / '.cache/csa-cmp-wgate-nz-369ad2c1-v1-candidate'
TARGET = WORKSPACE / '.cache/csa-cmp-wgate-nz-369ad2c1-v2-compat'
PACKAGE = 'dsv4_csa_cmp_wgate_nz_369ad2c1_v1'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def replay(text):
    old = '''    converted = []
    for name, target in target_layouts.items():
        source = meta["weight_layouts"].get(name)
'''
    new = '''    source_layouts = dict(meta["weight_layouts"])
    # 旧schema=2没有把Compressor权重列入根布局名单，但张量级记录已明确ND。
    # 限定已知几何、dtype和基础物理格式，不能从元素数推测来源或覆盖未知布局。
    if "cmp_wgate" in target_layouts and "cmp_wgate" not in source_layouts:
        spec, value = meta["tensors"]["cmp_wgate"], tensors["cmp_wgate"]
        if (meta.get("schema_version") == SCHEMA_VERSION and spec.get("layout") == "ND"
                and spec.get("source_format") in (0, 2) and spec["shape"] == [1024, 4096]
                and spec["dtype"] == "torch.bfloat16" and str(value.dtype) == "torch.bfloat16"
                and tuple(value.shape) == (1024, 4096) and value.is_contiguous()):
            source_layouts["cmp_wgate"] = "ND"
    converted = []
    for name, target in target_layouts.items():
        source = source_layouts.get(name)
'''
    text = once(text, old, new)
    return once(text, 'repack_weights(tensors, meta["weight_layouts"], target_layouts, target_shapes)',
                'repack_weights(tensors, source_layouts, target_layouts, target_shapes)')


def root_test(text):
    text = once(text, '    if mode >= 1:\n        expected_nz.update(("wq_b", "wo_b"))\n',
                '    if mode >= 1:\n        expected_nz.update(("wq_b", "wo_b"))\n'
                '    if suffix == "_perf" and mode == 2:\n        expected_nz.add("cmp_wgate")\n')
    return once(text, '        "wo_a": (8, 4096, 1024), "wo_b": (8192, 4096),\n',
                '        "wo_a": (8, 4096, 1024), "wo_b": (8192, 4096), "cmp_wgate": (1024, 4096),\n')


TESTS = '''

@pytest.mark.parametrize("target_layout", ["ND", "NZ"])
def test_legacy_compressor_gate_snapshot_migrates_both_ways(target_layout):
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_adapter import _unpack_nz

    gate = (torch.arange(1024 * 4096).reshape(1024, 4096) % 97).to(torch.bfloat16)
    roles, source = {"cmp_wgate": "in"}, {"state_timing": "before_call"}
    meta, payload = capture_tensors({"cmp_wgate": gate}, roles, {}, source)
    values, _ = materialize(meta, payload)
    shapes = {"cmp_wgate": (1024, 4096)}
    converted, names = convert_weight_layouts(values, meta, {"cmp_wgate": target_layout}, shapes)
    assert names == (["cmp_wgate"] if target_layout == "NZ" else [])
    actual = _unpack_nz(converted["cmp_wgate"]) if target_layout == "NZ" else converted["cmp_wgate"]
    assert torch.equal(actual, gate)
    if target_layout == "ND":
        assert converted["cmp_wgate"] is values["cmp_wgate"]
    new_meta, new_payload = capture_tensors(converted, roles, {"cmp_wgate": target_layout}, source)
    new_values, _ = materialize(new_meta, new_payload)
    unchanged, names = convert_weight_layouts(new_values, new_meta, {"cmp_wgate": target_layout}, shapes)
    assert names == [] and unchanged["cmp_wgate"] is new_values["cmp_wgate"]
    precision, names = convert_weight_layouts(new_values, new_meta, {"cmp_wgate": "ND"}, shapes)
    assert torch.equal(precision["cmp_wgate"], gate)


def test_legacy_compressor_gate_requires_explicit_nd_and_exclusive_input():
    gate = torch.zeros((1024, 4096), dtype=torch.bfloat16)
    source, shapes = {"state_timing": "before_call"}, {"cmp_wgate": (1024, 4096)}
    for field, value in (("source_format", 29), ("layout", "NZ")):
        meta, _ = capture_tensors({"cmp_wgate": gate}, {"cmp_wgate": "in"}, {}, source)
        meta["tensors"]["cmp_wgate"][field] = value
        with pytest.raises(ValueError, match="未知权重布局"):
            convert_weight_layouts({"cmp_wgate": gate}, meta, {"cmp_wgate": "NZ"}, shapes)
    meta, _ = capture_tensors({"cmp_wgate": gate}, {"cmp_wgate": "in"}, {"cmp_wgate": "unknown"}, source)
    with pytest.raises(ValueError, match="未知权重布局"):
        convert_weight_layouts({"cmp_wgate": gate}, meta, {"cmp_wgate": "NZ"}, shapes)
    for aliased in (False, True):
        tensors = {"cmp_wgate": gate, "alias": gate.view(-1)} if aliased else {"cmp_wgate": gate}
        roles = {name: "in" for name in tensors} if aliased else {"cmp_wgate": "inout"}
        meta, _ = capture_tensors(tensors, roles, {}, source)
        with pytest.raises(ValueError, match="共享存储" if aliased else "只读权重"):
            convert_weight_layouts(tensors, meta, {"cmp_wgate": "NZ"}, shapes)
'''


def main():
    assert not TARGET.exists(), TARGET
    shutil.copytree(BASE, TARGET, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'build_output'))
    changes = {
        'tests/pypto_test/dsv4_csa_replay.py': replay,
        'tests/pypto_test/test_csa_nz_config.py': root_test,
        'tests/pypto_test/test_csa_replay.py': lambda text: text + TESTS,
    }
    patch = []
    for relative, change in changes.items():
        p = TARGET / relative
        before = p.read_text()
        after = change(before)
        ast.parse(after)
        p.chmod(0o644)
        p.write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile='a/' + relative, tofile='b/' + relative))
    # CPU真实根测试用标准性能入口，复制同一份已测kernel；不动旧冻结包或生产源码。
    packages = TARGET / 'vllm_ascend/ops/pypto'
    for name in ('decode_csa.py', 'decode_compressor_ratio4.py'):
        p = packages / 'deepseek_v4_flash_dspark_perf' / name
        p.chmod(0o644)
        p.write_text((packages / PACKAGE / name).read_text())
    (ROOT / 'compat.patch').write_text(''.join(patch))
    (ROOT / 'compat_source.json').write_text(json.dumps({
        'source': str(TARGET), 'base': str(BASE), 'kernel_variant': 'pkg:' + PACKAGE,
        'change': '只补旧ND快照到新NZ/ND、NZ回精度版ND及实际两版根契约；已测kernel与数学顺序保持',
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
