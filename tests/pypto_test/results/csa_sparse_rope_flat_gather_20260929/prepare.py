"""借鉴AscendC整块Gather，消除Sparse最终RoPE的逐行向量barrier。"""
import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / '.cache/csa-kv-native-nz-369ad2c1-v1-baseline'
PREFIX = WORKSPACE / '.cache/csa-sparse-rope-flat-gather-369ad2c1-v1'
OLD_PACKAGE = 'dsv4_csa_kv_native_nz_369ad2c1_v1'
PACKAGE = 'dsv4_csa_sparse_rope_flat_gather_369ad2c1_v1'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not (ROOT / 'task.txt').exists()
    for side in ('baseline', 'candidate'):
        dest = Path(str(PREFIX) + '-' + side)
        assert not dest.exists(), dest
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'build_output'))
        packages = dest / 'vllm_ascend/ops/pypto'
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    relative = Path('vllm_ascend/ops/pypto') / PACKAGE / 'decode_sparse_attn_csa.py'
    path = Path(str(PREFIX) + '-candidate') / relative
    before = path.read_text()
    after = once(before,
                 'm_swap_idx = pl.row_expand_add(m_swap_source, pl.reshape(m_row_offsets, [FINAL_HEAD_TILE, 1]))',
                 '''m_swap_idx = pl.reshape(
                pl.row_expand_add(m_swap_source, pl.reshape(m_row_offsets, [FINAL_HEAD_TILE, 1])),
                [1, FINAL_HEAD_TILE * ROPE_DIM],
            )''')
    after = once(after, 'm_gather_tmp = pl.create_tile([FINAL_HEAD_TILE, ROPE_DIM], dtype=pl.INT32)',
                 'm_gather_tmp = pl.create_tile([1, FINAL_HEAD_TILE * ROPE_DIM], dtype=pl.INT32)')
    after = once(after, 'm_swapped = pl.tile.gather(n_full, m_swap_idx, m_gather_tmp)',
                 '''# The indices already contain the absolute head-row offset.
                            # One contiguous Gather avoids A3's per-row vector barriers.
                            m_swapped_flat = pl.tile.gather(
                                pl.reshape(n_full, [1, FINAL_HEAD_TILE * HEAD_DIM]),
                                m_swap_idx, m_gather_tmp,
                            )
                            m_swapped = pl.reshape(m_swapped_flat, [FINAL_HEAD_TILE, ROPE_DIM])''')
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    (ROOT / 'candidate.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='a/' + str(relative), tofile='b/' + str(relative))))
    template = ROOT.parent / 'csa_kv_native_nz_20260929'
    for name in ('compile.py', 'run.sh', 'run_side.sh'):
        text = (template / name).read_text().replace(template.name, ROOT.name)
        text = text.replace('csa-kv-native-nz-369ad2c1-v1', PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        (ROOT / name).write_text(text)
    (ROOT / 'source.json').write_text(json.dumps({
        'baseline': '369ad2c1', 'source_prefix': str(PREFIX), 'base_source': str(BASE),
        'variant': 'pkg:' + PACKAGE, 'cases': [[131072, 16], [8192, 24]],
        'change': '仅Sparse最终逆RoPE的16x64 Gather展平1x1024；沿用绝对head偏移，输出reshape还原',
        'native_reference': 'ops-transformer28f40354/rotary_position_embedding/rotate_interleaved_split_bsn_pad.h',
        'isa_reference': 'pto-isa-src/include/pto/npu/a2a3/TGather.hpp:TGather逐validRow vmuls/barrier/vgather',
        'not_changed': 'Sparse QK/PV、softmax与舍入/除法、KV采集、任务依赖/early、Native cache、四张NZ及工具链',
        'acceptance': '同卡两档完整CSA/P95与四窗Sparse核内；性能后八类完整状态；有收益再补H127/B3/padding',
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
