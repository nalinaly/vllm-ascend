# SPDX-License-Identifier: Apache-2.0
"""Regenerate the old comparison, then append a paired repeat using old / new."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def passed(*groups):
    return '通过' if all(v['passed'] for g in groups for v in g.values()) else '未通过'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_aclnn', type=Path)
    parser.add_argument('new_aclnn', type=Path)
    parser.add_argument('new_original', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    current = json.loads(args.new_aclnn.read_text())
    original = json.loads(args.new_original.read_text())
    rows = {(r['B'], r['S']): r for r in current['cases']}
    original_rows = {(r['batch_size'], r['seqlen']): r for r in original['cases']}
    expected = {(b, s) for b in (2, 4, 8, 16, 20, 24, 28) for s in (5, 6)}
    assert set(rows) == set(original_rows) == expected
    assert original['status'] == 'PASS'
    assert str(current['physical_device']) == str(original['physical_device'])
    for row in rows.values():
        for side in ('native', 'pto'):
            assert len(row['device'][side]['samples_us']) == 3
    subprocess.run([sys.executable, str(root / 'scripts/summarize_all.py'),
                    str(args.old_aclnn)], check=True)
    md = root / '文档' / '开源Native与PTO_Graph重放对比_20261013.md'
    lines = md.read_text().splitlines()
    section = ''
    output = []
    for line in lines:
        if line.startswith('- 本次物理设备：'):
            line = line.replace('- 本次物理设备：', '- 左侧旧 ACLNN/Native 数据的物理设备：') + f" 右侧复测三组均为设备 `{current['physical_device']}`。"
        elif line.startswith('- 本次报告：'):
            line = line.replace('- 本次报告：', '- 左侧旧 ACLNN/Native 报告：')
        if line.startswith('|B|S|T|10-12 版本 Native|'):
            section = 'main'
        elif line.startswith('|B|S|T|Native 图重放|'):
            section = 'replay'
        elif line.startswith('|B/S|普通调用'):
            section = 'checks'
        elif line.startswith('#'):
            section = ''
        if line.startswith('|') and section:
            cells = line.split('|')[1:-1]
            if cells[0].isdigit() or (section == 'checks' and cells[0].split('/')[0].isdigit()):
                key = tuple(map(int, cells[0].split('/'))) if section == 'checks' else (int(cells[0]), int(cells[1]))
                row = rows[key]
                additions = {}
                if section == 'main':
                    n = row['device']['native']['median_us']
                    q = row['device']['pto']['median_us']
                    additions = {
                        5: f"{original_rows[key]['incore_timing']['pto']['span_p50_us']:.2f}",
                        6: f'{n:.2f}', 7: f'{q:.2f}',
                        8: f'{(n / float(cells[3]) - 1) * 100:+.1f}%',
                        9: f'{(q / n - 1) * 100:+.1f}%',
                        10: passed(row['pto_vs_native']),
                    }
                elif section == 'replay':
                    additions = {3: f"{row['replay']['native']['median_us']:.2f}",
                                 4: f"{row['replay']['pto']['median_us']:.2f}"}
                elif section == 'checks':
                    additions = {1: passed(row['cpu']['pto']), 2: passed(row['cpu']['native']),
                                 3: passed(row['pto_graph_vs_eager'], row['native_graph_vs_eager']),
                                 4: passed(row['pto_updated_eager'], row['native_updated_eager']),
                                 5: passed(row['pto_updated_cpu']),
                                 6: passed(row['native_updated_cpu'])}
                for index, value in additions.items():
                    cells[index] += ' / ' + value
                line = '|' + '|'.join(cells) + '|'
        output.append(line)
        if line == '## 3. ACLNN 接入测试结果':
            output += ['', '**表格中的 `原值 / 复测值`：左边保留此前数据，右边为同卡连续复测；10-12 版本历史列不变。**',
                       f"复测物理设备 {current['physical_device']}，先跑原版 PyPTO/Simpler，再跑 ACLNN/Native，仍为相同的 14 组输入。"]
    output += ['', '### 同卡复测原始记录', '', '- 随仓保存的两轮耗时样本及精度结果：[performance_samples.json](../reports/performance_samples.json)。']
    output += ['- 表中原版 PTO 为 L1 核内跨度，ACLNN 为 CANN kernel 时长；同卡复测消除了卡号差异，但计时边界仍不同，差值不能直接当作 Python 或 ACLNN 调用开销。', '']
    output += ['### 复测观察', '']
    for key in [(16, 6), (28, 6)]:
        r = rows[key]
        l1 = original_rows[key]['incore_timing']['pto']['span_p50_us']
        aclnn = r['device']['pto']['median_us']
        output.append(f"- B{key[0]}/S{key[1]}，T{r['T']}：原版 L1 {l1:.2f} μs，ACLNN kernel {aclnn:.2f} μs。")
    output += ['- 两类耗时的计时边界不同，且运行时状态存在差异，不能据此拆出纯包装开销。']
    failures = []
    for key, r in rows.items():
        for name, result in r['native_updated_cpu'].items():
            if not result['passed']:
                failures.append(f"B{key[0]}/S{key[1]} {name}（最大绝对误差 {result['max_abs']:.3e}）")
    if failures:
        output.append('- 修改输入为原值的 -0.5 倍后，Native 对 CPU 未通过当前阈值的项目：' + '、'.join(failures) + '。详见验证表。')
    output.append('')
    md.write_text('\n'.join(output))
    provenance = { 'physical_device': current['physical_device'], 'reports': {}}
    for label, path in [('original_pto', args.new_original), ('aclnn', args.new_aclnn)]:
        provenance['reports'][label] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / 'reports/paired_repeat.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(md)


if __name__ == '__main__':
    main()
