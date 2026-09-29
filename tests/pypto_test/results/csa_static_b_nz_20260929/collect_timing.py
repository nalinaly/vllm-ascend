"""性能采样结束后检查完整状态；无收益项不为DFX另占卡。"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
STATES = {'x_out', 'idx_topk', 'swa.0', 'compressed.0', 'state.0', 'indexer.0', 'indexer.1', 'indexer_state.0'}
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location('nz_timing_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    folder = parser.parse_args().folder.resolve()
    task = (folder / 'task.txt').read_text().strip()
    status = subprocess.check_output(['task-submit', '--status', task], text=True).strip()
    if status != 'completed (exit=0)':
        raise RuntimeError(status)
    source = json.loads((folder / 'source.json').read_text())
    pair = load(ROOT.parent / 'csa_compiled_pair_20260929/analyze.py')
    torch.set_num_threads(4)
    evidence = {'task': task, 'task_status': status, 'source': source, 'cases': []}
    for history, batch in source['cases']:
        sides, states = {}, {}
        for side in ('baseline', 'candidate'):
            path = folder / f'h{history}_b{batch}/timing/{side}'
            report = json.loads((path / 'report.json').read_text())
            if (report['source'], report['variant'], report['batch'], report['history'], report['side']) != (
                    source['source_prefix'] + '-' + side, source['variant'], batch, history, 'pto'):
                raise ValueError('来源/档位错误')
            if not report['backend_options']['inplace_pass']:
                raise ValueError('缺少inplace_pass')
            item = pair.analyze_side(path)
            if item['topk']['structural_errors']:
                raise ValueError('Top-K结构错误')
            item['samples_us'] = report['timing']['samples_us']
            item['p95_over_p50'] = item['us_p95'] / item['us_p50']
            item['over_p50_5pct'] = sum(v > 1.05 * item['us_p50'] for v in item['samples_us'])
            sides[side] = item
            states[side] = torch.load(path / 'states.pt', map_location='cpu', weights_only=True)
            if set(states[side]) != STATES:
                raise ValueError('缺少完整状态')
        for field in ('device', 'cann', 'requested'):
            if sides['baseline'][field] != sides['candidate'][field]:
                raise ValueError(f'两侧配置不一致: {field}')
        checks = {k: compare_tensor(states['candidate'][k], states['baseline'][k], 0, 0) for k in sorted(STATES)}
        del states
        before, after = sides['baseline']['mean_us'], sides['candidate']['mean_us']
        evidence['cases'].append({'history': history, 'batch': batch, 'sides': sides, 'state_checks': checks,
                                  'delta_us': after - before, 'change_pct': 100 * (after / before - 1)})
    evidence['state_status'] = ('PASS' if all(v['status'] == 'PASS' for c in evidence['cases']
                                            for v in c['state_checks'].values()) else 'FAIL')
    evidence['weighted_8_2_delta_us'] = sum(w * c['delta_us'] for w, c in zip((.8, .2), evidence['cases']))
    evidence['weighted_8_2_change_pct'] = sum(w * c['change_pct'] for w, c in zip((.8, .2), evidence['cases']))
    (folder / 'timing_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 静态B逐项NZ：' + folder.name, '', '同卡CANN9.2/mode2/atomic0/det0，5预热20事件；单位μs。', '',
             '| 档位 | ND基线 | NZ候选 | 差值 | 变化 | P95 | max | 八类完整状态 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for case in evidence['cases']:
        a, b = (case['sides'][s] for s in ('baseline', 'candidate'))
        passed = all(v['status'] == 'PASS' for v in case['state_checks'].values())
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {a['mean_us']:.3f} | {b['mean_us']:.3f} "
                     f"| {case['delta_us']:+.3f} | {case['change_pct']:+.3f}% "
                     f"| {a['us_p95']:.3f}→{b['us_p95']:.3f} | {a['us_max']:.3f}→{b['us_max']:.3f} "
                     f"| {'PASS' if passed else 'FAIL'} |")
    lines += ['', f"8:2差值 {evidence['weighted_8_2_delta_us']:+.3f}μs；"
              f"变化 {evidence['weighted_8_2_change_pct']:+.3f}%；精度 {evidence['state_status']}。",
              '', '包括自身eager/图重放、跨版本完整状态、metadata/保护区及Top-K结构；不是整模型token/DSpark验收。',
              '有收益的候选再补DFX和受影响边界；本报告不宣称已有逐核性能或Native/PTO整模型结果。',
              '[全部样本、精度及profile路径](timing_evidence.json)']
    (folder / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if evidence['state_status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
