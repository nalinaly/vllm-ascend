# SPDX-License-Identifier: Apache-2.0
"""混合请求基准：逐token、逐请求DSpark与实际PTO图覆盖对照。"""
import argparse
import json
import re
from pathlib import Path

from low_acceptance_20261001.mixed import collect


def compare(performance, precision, bank, batch, tokens=192, mean_range=(3.0, 4.0)):
    report = {'status': 'FAIL', 'batch': batch, 'ranks': 16, 'tokens_per_request': tokens,
              'compared_tokens': 0, 'token_mismatches': 0, 'dspark_mismatched_requests': 0,
              'errors': [], 'acceptance': {}, 'cases': [], 'graph_coverage': {}, 'static_kernel': {}}
    targets = {f'model.layers.{i}.self_attn.attn' for i in range(2, 43)}
    for variant, root in [('performance', performance), ('precision', precision)]:
        summary = collect(root, bank, batch, tokens, mean_range)
        report['acceptance'][variant] = {k: v for k, v in summary.items() if k != 'requests'}
        report['errors'].extend(f'{variant}: {e}' for e in summary['errors'])
        if not summary['qualified']:
            report['errors'].append(f'{variant}: 主基准接受分布不符合要求')
        coverage = []
        for rank in range(16):
            data = json.loads((root / f'rank{rank}.mixed.json').read_text())
            for key, expected in {'backend': 'pto', 'variant': variant, 'batch': batch,
                                  'decode_tokens': tokens, 'atomic_add': '0', 'deterministic': True,
                                  'hccl_deterministic': 'true', 'pto_runtime': 'tensormap_and_ringbuffer'}.items():
                if data.get(key) != expected:
                    report['errors'].append(f'{variant}/rank{rank}: {key}配置不符')
            observed = data['observation'][0]
            selection = observed.get('capture_time_selection', {})
            bucket = f'pto_tokens{batch * 6}'
            forwards = observed.get('model_forward_counts', {}).get(f'FULL_tokens{batch * 6}', 0)
            if (set(observed.get('target_layer_names', [])) != targets or set(selection) != targets
                    or any(selection.get(layer, {}).get(bucket, 0) <= 0 for layer in targets) or forwards <= 0):
                report['errors'].append(f'{variant}/rank{rank}: 41层满档PTO图覆盖不足')
            log = (root / f'rank{rank}.log').read_text(errors='replace')
            if any(not re.search(rf'OFFLINE_CACHE_LOADED dp={rank} key={row["key"]}(?:\s|$)', log)
                   for row in data['requests']):
                report['errors'].append(f'{variant}/rank{rank}: cache恢复记录缺失')
            coverage.append({'rank': rank, 'full_graph_forwards': forwards, 'layers': len(selection)})
        report['graph_coverage'][variant] = coverage
        log = (root / 'rank0.log').read_text(errors='replace')
        counts = [int(x) for x in re.findall(r'OFFLINE_STATIC_KERNEL[^\n]*installed_packages=(\d+)', log)]
        errors = log.count('execute op_compiler error')
        report['static_kernel'][variant] = {'installed_packages': counts, 'compiler_errors': errors}
        if not counts or min(counts) <= 0 or errors:
            report['errors'].append(f'{variant}: static kernel实际安装失败')
    for rank in range(16):
        sides = [json.loads((root / f'rank{rank}.mixed.json').read_text()) for root in (performance, precision)]
        if sides[0]['worker_runtime_config'] != sides[1]['worker_runtime_config']:
            report['errors'].append(f'rank{rank}: worker实际配置不同')
        events = [{row['key']: row['events'] for row in map(json.loads,
                   (root / f'request_stats_rank{rank}.jsonl').read_text().splitlines())}
                  for root in (performance, precision)]
        for left, right in zip(sides[0]['requests'], sides[1]['requests']):
            if left['key'] != right['key']:
                raise ValueError('两侧请求顺序不同')
            mismatch = [{'position': i, 'performance': a, 'precision': b}
                        for i, (a, b) in enumerate(zip(left['output_token_ids'], right['output_token_ids'])) if a != b]
            stats = []
            for side in events:
                event = side[left['key']]
                stats.append({'drafts': len(event), 'proposed': sum(e[0] for e in event),
                              'accepted': sum(e[1] for e in event),
                              'histogram': [sum(e[1] == n for e in event) for n in range(6)]})
            changed = stats[0] != stats[1]
            report['cases'].append({'rank': rank, 'key': left['key'], 'kind': left['kind'],
                                    'token_mismatches': len(mismatch), 'first_mismatches': mismatch[:4],
                                    'dspark': dict(zip(('performance', 'precision'), stats)),
                                    'dspark_different': changed,
                                    'event_sequence_equal': events[0][left['key']] == events[1][left['key']]})
            report['compared_tokens'] += tokens
            report['token_mismatches'] += len(mismatch)
            report['dspark_mismatched_requests'] += changed
    if not report['errors'] and not report['token_mismatches'] and not report['dspark_mismatched_requests']:
        report['status'] = 'PASS'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance', type=Path, required=True)
    parser.add_argument('--precision', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--mean-range', nargs=2, type=float, default=(3.0, 4.0))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.performance, args.precision, args.bank, args.batch, mean_range=tuple(args.mean_range))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'graph_coverage')}, ensure_ascii=False))
    raise SystemExit(result['status'] != 'PASS')
