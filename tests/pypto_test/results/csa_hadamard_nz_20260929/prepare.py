"""只将Q/K共享Hadamard的BF16 B权重改为NZ，沿用同工作量和私有包对照。"""
import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / '.cache/csa-kv-native-nz-369ad2c1-v1-baseline'
PREFIX = WORKSPACE / '.cache/csa-hadamard-nz-369ad2c1-v1'
OLD = 'dsv4_csa_kv_native_nz_369ad2c1_v1'
PACKAGE = 'dsv4_csa_hadamard_nz_369ad2c1_v1'
TEMPLATE = ROOT.parent / 'csa_kv_native_nz_20260929'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def kernel(text):
    found = 0
    for size in ('IDX_HEAD_DIM', 'HEAD_DIM'):
        old = f'hadamard: pl.Tensor[[{size}, {size}], pl.BF16]'
        found += text.count(old)
        text = text.replace(old, old[:-1] + ', BF16_WEIGHT_LAYOUT]')
    assert found >= 2
    return once(text, 'import pypto.language as pl\n',
                'import pypto.language as pl\n\nfrom .nz_mode import BF16_WEIGHT_LAYOUT\n')


def layout(text):
    return once(text, 'for name in ("wq_a", "wq_b", "wo_a", "wo_b"):',
                'for name in ("wq_a", "wq_b", "wo_a", "wo_b", "hadamard_idx"):')


def adapter(text):
    text = once(text, '    def scale(module, width):',
                '    def hadamard_weight(value):\n'
                '        value = value.detach().T.to(torch.bfloat16).contiguous()\n'
                '        return torch_npu.npu_format_cast(value, 29) if layouts["hadamard_idx"] == "NZ" else value\n\n'
                '    def scale(module, width):')
    return once(text, '"hadamard_idx": hadamard.detach().T.to(bf16).contiguous()',
                '"hadamard_idx": hadamard_weight(hadamard)')


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
                    'original = (hadamard if name == "hadamard_idx"\n'
                    '                        else getattr(layer.self_attn, name).weight)')
        text = once(text, 'if already_matches and not reused:',
                    'if already_matches and not reused and name != "hadamard_idx":')
        diagnostic.chmod(0o644)
        diagnostic.write_text(text)
    mutations = {
        f'vllm_ascend/ops/pypto/{PACKAGE}/decode_indexer.py': kernel,
        f'vllm_ascend/ops/pypto/{PACKAGE}/decode_indexer_compressor.py': kernel,
        f'vllm_ascend/ops/pypto/{PACKAGE}/decode_csa.py': lambda text: once(
            text, 'hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16]',
            'hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16, BF16_WEIGHT_LAYOUT]'),
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
        'change': (
            '仅Q/K共享Hadamard的BF16 B按NZ读取，'
            '初始化准备一次NZ并供Q/K路径共享，'
            '保留既有矩阵方向与缩放；'
            '矩阵方向/K顺序/任务数/依赖保持'
        ),
        'acceptance': '先看完整CSA绝对时间与P95，8:2；性能后核对全部状态，未证实整体收益不接生产',
        'not_included': '独立对照，不叠加KV候选；旧快照/ND/atomic/尾行按收益决定后续受影响检查',
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
