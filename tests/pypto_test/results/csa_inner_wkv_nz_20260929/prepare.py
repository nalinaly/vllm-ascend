"""只将inner_wkv的BF16 B权重改为NZ，沿用同工作量和私有包对照。"""
import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / '.cache/csa-kv-native-nz-369ad2c1-v1-baseline'
PREFIX = WORKSPACE / '.cache/csa-inner-wkv-nz-369ad2c1-v1'
OLD = 'dsv4_csa_kv_native_nz_369ad2c1_v1'
PACKAGE = 'dsv4_csa_inner_wkv_nz_369ad2c1_v1'
TEMPLATE = ROOT.parent / 'csa_kv_native_nz_20260929'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def kernel(text):
    old = 'wkv: pl.Tensor[[OUT_DIM, D], pl.BF16]'
    assert text.count(old) >= 2
    text = text.replace(old, old[:-1] + ', BF16_WEIGHT_LAYOUT]')
    return once(text, 'import pypto.language as pl\n',
                'import pypto.language as pl\n\nfrom .nz_mode import BF16_WEIGHT_LAYOUT\n')


def layout(text):
    return once(text, 'for name in ("wq_a", "wq_b", "wo_a", "wo_b"):',
                'for name in ("wq_a", "wq_b", "wo_a", "wo_b", "inner_wkv"):')


def adapter(text):
    text = once(text, 'def root_weight(name, shape, dtype):', 'def root_weight(name, shape, dtype, module=None):')
    text = once(text, 'value = getattr(attention, name).weight.detach()',
                'value = (getattr(attention, name) if module is None else module).weight.detach()')
    text = once(text, 'return value if current == 29 else _recast(name, value, current, 29)',
                'if current == 29:\n                return value\n'
                '            if module is not None:\n'
                '                logger.info("PTO_CSA_WEIGHT_INIT_NZ %s: PTO专用NZ副本，Native原权重保留", name)\n'
                '                return torch_npu.npu_format_cast(value, 29)\n'
                '            return _recast(name, value, current, 29)')
    return once(text, '"inner_wkv": weight(inner.wkv, (256, 4096), bf16),',
                '"inner_wkv": root_weight("inner_wkv", (256, 4096), bf16, inner.wkv),')


def main():
    patch = []
    for side in ('baseline', 'candidate'):
        dest = Path(str(PREFIX) + '-' + side)
        assert not dest.exists(), dest
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'build_output'))
        folder = dest / 'vllm_ascend/ops/pypto'
        (folder / OLD).rename(folder / PACKAGE)
        # 两侧诊断同样支持嵌套权重映射；不触及计算/计时行为。
        diagnostic = dest / 'tests/pypto_test/dsv4_csa_single_layer.py'
        text = once(diagnostic.read_text(), 'original = getattr(layer.self_attn, name).weight',
                    'original = (layer.self_attn.indexer.compressor.wkv.weight if name == "inner_wkv"\n'
                    '                        else getattr(layer.self_attn, name).weight)')
        diagnostic.chmod(0o644)
        diagnostic.write_text(text)
    mutations = {
        f'vllm_ascend/ops/pypto/{PACKAGE}/decode_indexer_compressor.py': kernel,
        f'vllm_ascend/ops/pypto/{PACKAGE}/decode_csa.py': lambda text: once(
            text, 'inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16]',
            'inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16, BF16_WEIGHT_LAYOUT]'),
        'vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/nz_mode.py': layout,
        'vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/native_adapter.py': adapter,
    }
    for relative, change in mutations.items():
        path = Path(str(PREFIX) + '-candidate') / relative
        before = path.read_text()
        after = change(before)
        ast.parse(after)
        path.chmod(0o644)
        path.write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile='a/' + relative, tofile='b/' + relative))
    (ROOT / 'candidate.patch').write_text(''.join(patch))
    for name in ('compile.py', 'run.sh', 'run_side.sh'):
        text = (TEMPLATE / name).read_text().replace(TEMPLATE.name, ROOT.name)
        text = text.replace('csa-kv-native-nz-369ad2c1-v1', PREFIX.name).replace(OLD, PACKAGE)
        (ROOT / name).write_text(text)
    (ROOT / 'source.json').write_text(json.dumps({
        'baseline': '369ad2c1', 'source_prefix': str(PREFIX), 'base_source': str(BASE),
        'variant': 'pkg:' + PACKAGE, 'cases': [[131072, 16], [8192, 24]],
        'change': '仅inner_wkv的BF16 B按NZ读取，初始化准备PTO专用NZ，Native原权重不变；矩阵方向/K顺序/任务数/依赖保持',
        'acceptance': '先看完整CSA绝对时间与P95，8:2；性能后核对全部状态，未证实整体收益不接生产',
        'not_included': '独立对照，不叠加KV候选；旧快照/ND/atomic/尾行按收益决定后续受影响检查',
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
