"""为已通过性能后精度检查的候选补DFX与权重格式证据，不重复比对完整状态。"""
import argparse
import functools
import importlib.util
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--task-name', required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    timing = json.loads((folder / 'timing_evidence.json').read_text())
    if timing['state_status'] != 'PASS':
        raise ValueError('完整状态未通过')
    task = (folder / 'dfx_task.txt').read_text().strip()
    if subprocess.check_output(['task-submit', '--status', task], text=True).strip() != 'completed (exit=0)':
        raise RuntimeError('DFX任务未成功完成')
    task_log = subprocess.check_output(['task-submit', '--log', task], text=True)
    devices = set(re.findall(r'已获取设备 ([0-9]+) 的锁', task_log))
    if len(devices) != 1:
        raise ValueError('无法确认同一任务的单卡分配')
    device = int(next(iter(devices)))
    source = json.loads((folder / 'source.json').read_text())
    metrics = load('nz_pair_metrics', ROOT.parent / 'csa_score_segment_ub_20260929/collect.py')
    worker = load('nz_pair_worker', ROOT.parent / 'csa_scheduling_20260927/upstream_725/compare.py')
    worker.summarize = functools.partial(worker.summarize, publication_task='qk_pv_aiv')
    helper = load('nz_pair_helper', WORKSPACE / 'pypto-lib/.claude/skills/critical-path/scripts/report.py')
    result = {'dfx_task': task, 'timing_task': timing['task'], 'target': args.task_name, 'cases': []}
    for history, batch in source['cases']:
        sides = {}
        for side in ('baseline', 'candidate'):
            p = folder / f'h{history}_b{batch}/swimlane/{side}/report.json'
            report = json.loads(p.read_text())
            if (report['status'], report['variant'], report['history'], report['batch'],
                report['effective_weight_nz_mode'], report['deterministic_level'],
                report['pto_reduction']['atomic_add']) != ('MEASURED', source['variant'], history, batch, 2, 0, 0):
                raise ValueError('DFX来源/配置不一致')
            windows = report['swimlane_windows']
            if len(windows) != 4 or any(not w['exported'] or w['execution'] != 'graph_replay' for w in windows):
                raise ValueError('缺少独立四窗口')
            analyzed = [metrics.schedule_window(Path(w['merged_swimlane']), batch, helper, worker) for w in windows]
            tasks = [w['tasks'][args.task_name] for w in analyzed]
            if any(t['blocks'] != 24 for t in tasks):
                raise ValueError('Compressor工作份数改变')
            sides[side] = {'device': device, 'windows': analyzed,
                           'kernel_us': [t['kernel_mean_us'] for t in tasks],
                           'execution_config': report['execution_config']}
            if side == 'candidate':
                binding = report['weight_storage_binding'][args.weight]
                if (binding['native_format'] not in (0, 2) or binding['pto_format'] != 29
                        or binding['same_data_ptr'] or binding['root_layout'] != 'NZ'):
                    raise ValueError('Native ND/PTO初始化NZ绑定未生效')
                sides[side]['weight_binding'] = binding
        if sides['baseline']['device'] != sides['candidate']['device']:
            raise ValueError('DFX两侧不同卡')
        a, b = (statistics.mean(sides[s]['kernel_us']) for s in ('baseline', 'candidate'))
        result['cases'].append({'history': history, 'batch': batch, 'sides': sides,
                                'kernel_change_pct': 100 * (b / a - 1)})
    (folder / 'dfx_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    summary = {'dfx_task': task, 'timing_task': timing['task'], 'target': args.task_name, 'cases': []}
    for c in result['cases']:
        a, b = (c['sides'][s] for s in ('baseline', 'candidate'))
        summary['cases'].append({'history': c['history'], 'batch': c['batch'],
                                 'baseline_us': a['kernel_us'], 'candidate_us': b['kernel_us'],
                                 'change_pct': c['kernel_change_pct'], 'dfx_device': a['device'],
                                 'weight_binding': b['weight_binding']})
    (folder / 'dfx_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
