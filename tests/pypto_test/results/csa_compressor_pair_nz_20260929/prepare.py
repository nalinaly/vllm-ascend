"""组合两项有完整CSA收益且逐元素通过的Compressor NZ；其余权重保持生产。"""
import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / '.cache/csa-kv-native-nz-369ad2c1-v1-baseline'
GATE = WORKSPACE / '.cache/csa-cmp-wgate-nz-369ad2c1-v2-compat'
INNER = WORKSPACE / '.cache/csa-inner-wkv-nz-369ad2c1-v1-candidate'
PREFIX = WORKSPACE / '.cache/csa-compressor-pair-nz-369ad2c1-v1'
PACKAGE = 'dsv4_csa_compressor_pair_nz_369ad2c1_v1'
OLD = 'dsv4_csa_kv_native_nz_369ad2c1_v1'
GATE_PACKAGE = 'dsv4_csa_cmp_wgate_nz_369ad2c1_v1'
INNER_PACKAGE = 'dsv4_csa_inner_wkv_nz_369ad2c1_v1'
TEMPLATE = ROOT.parent / 'csa_cmp_wgate_nz_20260929'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    packages = Path('vllm_ascend/ops/pypto')
    for side, origin, old_package in (('baseline', BASE, OLD), ('candidate', GATE, GATE_PACKAGE)):
        dest = Path(str(PREFIX) + '-' + side)
        assert not dest.exists(), dest
        shutil.copytree(origin, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'build_output'))
        (dest / packages / old_package).rename(dest / packages / PACKAGE)
    dest = Path(str(PREFIX) + '-candidate')
    root_path = packages / PACKAGE / 'decode_csa.py'
    changes = {
        root_path: once((dest / root_path).read_text(),
                       'inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16]',
                       'inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16, BF16_WEIGHT_LAYOUT]'),
        packages / PACKAGE / 'decode_indexer_compressor.py': (
            INNER / packages / INNER_PACKAGE / 'decode_indexer_compressor.py').read_text(),
    }
    p = packages / 'deepseek_v4_flash_dspark/nz_mode.py'
    changes[p] = once((dest / p).read_text(), '"wo_b", "cmp_wgate"):', '"wo_b", "cmp_wgate", "inner_wkv"):')
    p = packages / 'deepseek_v4_flash_dspark/native_adapter.py'
    changes[p] = once((dest / p).read_text(), '"inner_wkv": weight(inner.wkv, (256, 4096), bf16),',
                      '"inner_wkv": root_weight("inner_wkv", (256, 4096), bf16, inner.wkv),')
    p = Path('tests/pypto_test/dsv4_csa_replay.py')
    text = (dest / p).read_text()
    start, end = text.index('    # 旧schema=2没有'), text.index('    converted = []')
    text = text[:start] + '''    # 旧schema=2张量级已有ND记录；只迁移已知的两个Compressor投影权重。
    # 保留显式未知布局、形状/dtype/物理格式和只读独占存储的拒绝检查。
    for name, shape in (("cmp_wgate", (1024, 4096)), ("inner_wkv", (256, 4096))):
        if name in target_layouts and name not in source_layouts:
            spec, value = meta["tensors"][name], tensors[name]
            if (meta.get("schema_version") == SCHEMA_VERSION and spec.get("layout") == "ND"
                    and spec.get("source_format") in (0, 2) and spec["shape"] == list(shape)
                    and spec["dtype"] == "torch.bfloat16" and str(value.dtype) == "torch.bfloat16"
                    and tuple(value.shape) == shape and value.is_contiguous()):
                source_layouts[name] = "ND"
''' + text[end:]
    changes[p] = text
    p = Path('tests/pypto_test/test_csa_nz_config.py')
    text = once((dest / p).read_text(), 'expected_nz.add("cmp_wgate")',
                'expected_nz.update(("cmp_wgate", "inner_wkv"))')
    changes[p] = once(text, '"cmp_wgate": (1024, 4096),', '"cmp_wgate": (1024, 4096), "inner_wkv": (256, 4096),')
    p = Path('tests/pypto_test/test_csa_replay.py')
    text = (dest / p).read_text()
    start = text.index('@pytest.mark.parametrize("target_layout", ["ND", "NZ"])')
    tests = text[start:]
    tests = tests.replace('@pytest.mark.parametrize("target_layout", ["ND", "NZ"])',
                          '@pytest.mark.parametrize("name,width", [("cmp_wgate", 1024), ("inner_wkv", 256)])\n'
                          '@pytest.mark.parametrize("target_layout", ["ND", "NZ"])')
    tests = tests.replace('def test_legacy_compressor_gate_snapshot_migrates_both_ways(target_layout):',
                          'def test_legacy_compressor_gate_snapshot_migrates_both_ways(name, width, target_layout):')
    tests = tests.replace('def test_legacy_compressor_gate_requires_explicit_nd_and_exclusive_input():',
                          '@pytest.mark.parametrize("name,width", [("cmp_wgate", 1024), ("inner_wkv", 256)])\n'
                          'def test_legacy_compressor_gate_requires_explicit_nd_and_exclusive_input(name, width):')
    # 只替换测试主体中的键/几何，参数表保留明确的两组名字和形状。
    lines = tests.splitlines(True)
    for i, line in enumerate(lines):
        if not line.startswith('@pytest.mark.parametrize'):
            lines[i] = line.replace('"cmp_wgate"', 'name').replace('1024 * 4096', 'width * 4096').replace(
                '1024, 4096', 'width, 4096')
    changes[p] = text[:start] + ''.join(lines)
    for relative, text in changes.items():
        ast.parse(text)
        p = dest / relative
        p.chmod(0o644)
        p.write_text(text)
    # CPU根测试用标准包；设备仍由冻结的pkg入口选择相同kernel。
    for name in ('decode_csa.py', 'decode_compressor_ratio4.py', 'decode_indexer_compressor.py'):
        p = dest / packages / 'deepseek_v4_flash_dspark_perf' / name
        p.chmod(0o644)
        p.write_text((dest / packages / PACKAGE / name).read_text())
    # 两侧统一诊断映射，不改变模型或计时。
    for side in ('baseline', 'candidate'):
        p = Path(str(PREFIX) + '-' + side) / 'tests/pypto_test/dsv4_csa_single_layer.py'
        before = (BASE / 'tests/pypto_test/dsv4_csa_single_layer.py').read_text()
        text = once(before, 'original = getattr(layer.self_attn, name).weight',
                    'original = (layer.self_attn.compressor.wgate.weight if name == "cmp_wgate"\n'
                    '                        else layer.self_attn.indexer.compressor.wkv.weight '
                    'if name == "inner_wkv"\n'
                    '                        else getattr(layer.self_attn, name).weight)')
        p.chmod(0o644)
        p.write_text(text)
    patch = []
    for relative in (*changes, packages / PACKAGE / 'decode_compressor_ratio4.py'):
        before = (Path(str(PREFIX) + '-baseline') / relative).read_text()
        after = (dest / relative).read_text()
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile='a/' + str(relative), tofile='b/' + str(relative)))
    (ROOT / 'candidate.patch').write_text(''.join(patch))
    for name in ('compile.py', 'run.sh', 'run_side.sh'):
        text = (TEMPLATE / name).read_text().replace(TEMPLATE.name, ROOT.name)
        text = text.replace('csa-cmp-wgate-nz-369ad2c1-v1', PREFIX.name).replace(GATE_PACKAGE, PACKAGE)
        (ROOT / name).write_text(text)
    (ROOT / 'source.json').write_text(json.dumps({
        'baseline': '369ad2c1', 'source_prefix': str(PREFIX), 'variant': 'pkg:' + PACKAGE,
        'cases': [[131072, 16], [8192, 24]],
        'change': '组合主Compressor wgate与Indexer Compressor wkv两项NZ；初始化准备，Native ND保留',
        'not_included': 'KV/Indexer Q/head/Hadamard NZ均不叠加；不改变K顺序/分工/任务依赖',
        'acceptance': '先确认组合仍缩短完整CSA，再查八类完整状态和P95；单项收益不可直接相加',
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
