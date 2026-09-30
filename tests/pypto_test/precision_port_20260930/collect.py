# SPDX-License-Identifier: Apache-2.0
"""汇总七档原始样本；检查配对配置、保护区及迁移代表用例。"""
import json
from pathlib import Path

from compare import compare

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'results' / HERE.name
CASES = [(131072, b) for b in (4, 8, 16, 24)] + [(8192, b) for b in (16, 24, 32)]
rows, reports, numerical = [], {}, {}
for history, batch in CASES:
    key = f'h{history}_b{batch}'
    pair = {}
    for variant in ('performance', 'precision'):
        report = json.loads((ROOT / 'matrix' / key / variant / 'report.json').read_text())
        assert report['status'] == 'MEASURED', (key, variant, report['status'])
        assert all(v['status'] == 'PASS' for v in report['guards']['CSA'].values())
        assert report['replay_vs_eager']['CSA']['status'] == 'PASS'
        assert not any(report['nonfinite'].values())
        pair[variant] = report
    for field in ('history', 'batch', 'atomic_add', 'task_device', 'weight_nz_mode', 'runtime',
                  'source', 'versions', 'warmup', 'iterations', 'checkpoint', 'deterministic_level'):
        assert pair['performance'][field] == pair['precision'][field], (key, field)
    assert pair['performance']['atomic_add'] == 0
    reports[key] = pair
    pf, pr = [pair[v]['timing_us']['CSA'] for v in ('performance', 'precision')]
    delta = (pr['mean'] / pf['mean'] - 1) * 100
    label = f'{history // 1024}K/B{batch}'
    rows.append(f"| {label} | {pf['min']:.2f} | {pf['mean']:.2f} | {pf['max']:.2f} "
                f"| {pr['min']:.2f} | {pr['mean']:.2f} | {pr['max']:.2f} | {delta:+.2f}% |")
    numerical[key] = compare(ROOT / 'matrix' / key / 'performance', ROOT / 'matrix' / key / 'precision')

guards = {}
for key in ('h131072_b16', 'h8192_b16', 'h257_b1', 'h131072_b24'):
    baseline = ROOT / key / 'baseline'
    candidate = ROOT / key / 'candidate_v11' if key == 'h257_b1' else ROOT / 'matrix' / key / 'precision'
    result = compare(baseline, candidate)
    assert result['all_exact'], (key, result)
    guards[key] = result

header = ('| 档位 | 性能版 min | 性能版 mean | 性能版 max | 精度版 min | 精度版 mean | '
          '精度版 max | 精度版 mean 增加 |\n'
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
text = ('# 精度版迁移后的七档 CSA 对比\n\n'
        '单位 μs；两版同卡顺序执行。CANN 9.2、TMR、NZ=2、atomic=0、deterministic level=0。'
        '正式第 2 层权重，合成固定历史，compact metadata 已准备。'
        '计时为 HC_pre+norm+CSA+HC_post 完整图重放；每版 5 次预热、20 次事件计时，'
        'D2D 恢复初态在计时外。未采集 profiler，未执行 16 卡。\n\n' + header + '\n' + '\n'.join(rows) + '\n\n')
text += ('四个迁移看护档的输出、Top-K 索引/分数与 cache/state 写入区均与旧精度版逐元素一致，'
         '保护区及只读 metadata 检查通过。七档两版分别通过图重放/eager 精确一致、有限值与写入保护检查。'
         '两版采用不同的算术策略，其逐元素差异单列于 numerical_difference.json，'
         '不能据此声称 Native/token/DSpark 已验收。\n')

text += '\n\n迁移前后的精度版代表档（同一环境，旧版基线不属于七档性能/精度相邻配对计时）：\n\n'
text += '| 档位 | 旧精度版 mean | 新精度版 mean | 耗时减少 |\n| --- | ---: | ---: | ---: |\n'
for key, label in (('h131072_b16', '128K/B16'), ('h131072_b24', '128K/B24'), ('h8192_b16', '8K/B16')):
    old = json.loads((ROOT / key / 'baseline/report.json').read_text())['timing_us']['CSA']['mean']
    new = reports[key]['precision']['timing_us']['CSA']['mean']
    text += f'| {label} | {old:.2f} | {new:.2f} | {(1 - new / old) * 100:.2f}% |\n'
ratio = max(v['timing_us']['CSA']['p95'] / v['timing_us']['CSA']['median']
            for pair in reports.values() for v in pair.values())
text += (f'\n本轮14组中最高 P95/median = {ratio:.4f}，最大值全部保留；'
         '20次采样未见突出的P95抬高，但不足以排除低频长尾。\n'
         '性能版七档CSA输出与Top-K索引和上一轮已有atomic0结果逐元素一致，见 `performance_unchanged.json`。\n')
(HERE / 'RESULTS.md').write_text(text)
(HERE / 'evidence.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
(HERE / 'migration_guards.json').write_text(json.dumps(guards, ensure_ascii=False, indent=2) + '\n')
(HERE / 'numerical_difference.json').write_text(json.dumps(numerical, ensure_ascii=False, indent=2) + '\n')
print(text)
