# SPDX-License-Identifier: Apache-2.0
"""五种CSA/HCA组合以Native为参考；可补充同配置Native/PTO跨运行复现。"""
import argparse
import gzip
import itertools
import json
import re
from pathlib import Path

from low_acceptance_20261001.mixed import collect
from low_acceptance_20261001.reference import extract

CONFIGURATIONS = {
    'native': ('Native CSA + Native HCA', 'native', 'both', 'performance', set()),
    'csa_precision_native_hca': ('CSA精度版 + Native HCA', 'pto', 'csa', 'precision', {'csa'}),
    'csa_performance_native_hca': ('CSA性能版 + Native HCA', 'pto', 'csa', 'performance', {'csa'}),
    'csa_performance_pto_hca': ('CSA性能版 + PTO HCA', 'pto', 'both', 'performance', {'csa', 'hca'}),
    'native_csa_pto_hca': ('Native CSA + PTO HCA', 'pto', 'hca', 'performance', {'hca'}),
}


def validate(root, bank, batch, configuration, mean_range=(3.0, 4.0)):
    label, backend, attention, variant, pto_kinds = CONFIGURATIONS[configuration]
    acceptance = collect(root, bank, batch, 192, mean_range)
    errors = list(acceptance['errors'])
    targets = {f'model.layers.{i}.self_attn.attn' for i in range(2, 43)}
    expected_pto_count = sum(21 if kind == 'csa' else 20 for kind in pto_kinds)
    graph_counts, configs, selections = [], [], []
    legacy_both_seen = False
    for rank in range(16):
        data = json.loads((root / f'rank{rank}.mixed.json').read_text())
        expected = {'backend': backend, 'variant': variant, 'batch': batch, 'decode_tokens': 192,
                    'atomic_add': '0', 'deterministic': True, 'hccl_deterministic': 'true',
                    'pto_runtime': 'tensormap_and_ringbuffer'}
        for field, value in expected.items():
            if data.get(field) != value:
                errors.append(f'rank{rank}: {field}={data.get(field)!r}, expected={value!r}')
        # 兼容修复前第4组的历史结果：当时没有描述符，但有41层PTO捕获证据。
        # 当前同步加载矩阵全部重新执行，具有完整描述符，不走此分支。
        legacy_both = configuration == 'csa_performance_pto_hca' and 'attention_implementations' not in data
        legacy_both_seen |= legacy_both
        if not legacy_both:
            if data.get('pto_attention') != attention:
                errors.append(f'rank{rank}: attention选择不符')
            rows = data['attention_implementations'][0]
            if ({r['name'] for r in rows} != targets
                    or sum(r['kind'] == 'csa' for r in rows) != 21
                    or sum(r['kind'] == 'hca' for r in rows) != 20):
                errors.append(f'rank{rank}: 模型目标层缺失')
            for row in rows:
                required = 'pto' if row['kind'] in pto_kinds else 'native'
                if row['implementation'] != required:
                    errors.append(f'rank{rank}/{row["name"]}: runtime实际选择错误')
                if row['kind'] == 'csa' and required == 'pto':
                    package = ('deepseek_v4_flash_dspark_perf' if variant == 'performance'
                               else 'deepseek_v4_flash_dspark')
                    if not row['runtime_class'].endswith(f'.{package}.service'):
                        errors.append(f'rank{rank}/{row["name"]}: CSA算术版本错误')
            if rank == 0:
                selections = rows
        observed = data['observation'][0]
        selection = observed.get('capture_time_selection', {})
        selected = [name for name, counts in selection.items() if counts.get(f'pto_tokens{batch * 6}', 0)]
        if len(selected) != expected_pto_count:
            errors.append(f'rank{rank}: PTO捕获层数{len(selected)} != {expected_pto_count}')
        forwards = observed.get('model_forward_counts', {}).get(f'FULL_tokens{batch * 6}', 0)
        if forwards <= 0:
            errors.append(f'rank{rank}: 未重放B×6档位图')
        graph_counts.append(forwards)
        configs.append(data['worker_runtime_config'])
        for actual in data['worker_runtime_config']:
            if (actual['deterministic_level'] != 1 or not actual['torch_deterministic']
                    or actual['hccl_deterministic'] != 'true' or actual['dynamic_eplb']):
                errors.append(f'rank{rank}: 实际确定性/EPLB配置不符')
            compilation = actual['decode_optimizations']['ascend_compilation_config']
            if not compilation['enable_npugraph_ex'] or not compilation['enable_static_kernel']:
                errors.append(f'rank{rank}: npugraph_ex/static_kernel未实际配置')
        log = (root / f'rank{rank}.log').read_text(errors='replace')
        if any(not re.search(rf'OFFLINE_CACHE_LOADED dp={rank} key={r["key"]}(?:\s|$)', log)
               for r in data['requests']):
            errors.append(f'rank{rank}: 固定cache恢复记录缺失')
        if 'execute op_compiler error' in log:
            errors.append(f'rank{rank}: 静态kernel编译失败')
    log = (root / 'rank0.log').read_text(errors='replace')
    installed = [int(v) for v in re.findall(r'OFFLINE_STATIC_KERNEL[^\n]*installed_packages=(\d+)', log)]
    compile_errors = log.count('execute op_compiler error')
    if not installed or min(installed) <= 0 or compile_errors:
        errors.append('静态kernel实际安装失败')
    records = extract(root)
    for rank in range(16):
        if len([r for r in records if r['rank'] == rank]) != batch:
            errors.append(f'rank{rank}: 请求数不符')
    # 只证明同次执行的跨DP一致性，不冒充同版本跨运行复现性。
    cross_dp = sum(a['tokens'] != b['tokens']
                   for rank in range(1, 16)
                   for a, b in zip(records[:batch], records[rank*batch:(rank+1)*batch]))
    cross_dp_stats = sum(a['events'] != b['events']
                         for rank in range(1, 16)
                         for a, b in zip(records[:batch], records[rank*batch:(rank+1)*batch]))
    return {'label': label, 'source': str(root.resolve()), 'execution_status': 'COMPLETE',
            'validation_errors': errors,
            'acceptance': {k: v for k, v in acceptance.items() if k != 'requests'},
            'cross_dp_different_outputs_within_run': cross_dp,
            'cross_dp_different_acceptance_events_within_run': cross_dp_stats,
            'graph_forwards_per_rank': graph_counts, 'attention_implementations': selections,
            'reused_legacy_both_evidence': legacy_both_seen,
            'static_kernel': {'installed_packages': installed, 'compiler_errors': compile_errors}}, records, configs


def compare_rows(left, right):
    if [(r['rank'], r['key']) for r in left] != [(r['rank'], r['key']) for r in right]:
        raise ValueError('不同组合的请求或顺序不同')
    details = []
    for a, b in zip(left, right):
        if len(a['tokens']) != len(b['tokens']):
            raise ValueError('输出长度不同，不能用zip截断比较')
        mismatch = [i for i, (x, y) in enumerate(zip(a['tokens'], b['tokens'])) if x != y]
        fields = [k for k in ('drafts', 'proposed', 'accepted', 'histogram') if a[k] != b[k]]
        details.append({'rank': a['rank'], 'key': a['key'], 'token_mismatches': len(mismatch),
                        'first_difference': mismatch[0] if mismatch else None, 'dspark_fields': fields,
                        'acceptance_event_sequence_equal': a['events'] == b['events'],
                        'first_difference_ids': ([a['tokens'][mismatch[0]], b['tokens'][mismatch[0]]]
                                                 if mismatch else [])})
    changed = sum(bool(r['token_mismatches']) for r in details)
    stats_changed = sum(bool(r['dspark_fields']) for r in details)
    events_changed = sum(not r['acceptance_event_sequence_equal'] for r in details)
    return {'token_status': 'FAIL' if changed else 'PASS',
            'status': 'FAIL' if changed or stats_changed or events_changed else 'PASS',
            'requests': len(left), 'tokens': sum(len(r['tokens']) for r in left),
            'token_mismatched_requests': changed, 'token_mismatches': sum(r['token_mismatches'] for r in details),
            'dspark_mismatched_requests': stats_changed,
            'acceptance_event_mismatched_requests': events_changed,
            'unique_token_mismatched_requests': len({r['key'] for r in details if r['token_mismatches']}),
            'unique_dspark_mismatched_requests': len({r['key'] for r in details if r['dspark_fields']}),
            'unique_event_mismatched_requests': len({r['key'] for r in details
                                                      if not r['acceptance_event_sequence_equal']}),
            'earliest_difference_zero_based': min((r['first_difference'] for r in details
                                                   if r['first_difference'] is not None), default=None),
            'details': details}


def repeat_conditions(original, repeated, original_workers, repeated_workers, labels):
    """重复运行保留相同自动容量策略；记录容量变化，不重配参数抹平差异。"""
    requested = [json.loads((root / 'configuration.json').read_text()) for root in (original, repeated)]
    capacities, startup_arguments = {}, {}
    for label, root in zip(labels, (original, repeated)):
        capacities[label], startup_arguments[label] = [], []
        for rank in range(16):
            log = (root / f'rank{rank}.log').read_text(errors='replace')
            arguments = re.findall(r'non-default args: (.+)', log)
            # 仅排除每次随机生成的通信标识及观察结果目录；其他参数原样比较。
            startup_arguments[label].append([
                re.sub(r"engine_id='[^']+'", "engine_id='<RUN_ID>'",
                       re.sub(r"'offline_acceptance_output': '[^']+'",
                              "'offline_acceptance_output': '<OUTPUT>'", value))
                for value in arguments])
            memory = re.findall(r'Available KV cache memory: ([\d.]+) GiB', log)
            tokens = re.findall(r'GPU KV cache size: ([\d,]+) tokens', log)
            capacities[label].append({'rank': rank, 'kv_memory_gib': memory,
                                      'kv_tokens': [int(value.replace(',', '')) for value in tokens]})
    errors = []
    if requested[0] != requested[1]:
        errors.append(f'{labels[0]}两次启动配置不同')
    if original_workers != repeated_workers:
        errors.append(f'{labels[0]}两次worker实际配置不同（包括event模式）')
    startup_equal = [bool(left) and left == right for left, right in
                     zip(startup_arguments[labels[0]], startup_arguments[labels[1]])]
    if not all(startup_equal):
        errors.append(f'{labels[0]}两次启动日志参数不同或缺失（仅排除运行ID和输出目录）')
    if any(not row['kv_memory_gib'] or not row['kv_tokens'] for rows in capacities.values() for row in rows):
        errors.append(f'{labels[0]}重复运行缺少自动KV容量记录')
    return {'status': 'FAIL' if errors else 'PASS', 'errors': errors,
            'requested_configuration_equal': requested[0] == requested[1],
            'worker_configuration_equal_all_ranks': original_workers == repeated_workers,
            'startup_arguments_equal_per_rank': startup_equal,
            'requested_configurations': dict(zip(labels, requested)),
            'kv_capacity': capacities, 'kv_capacity_equal': capacities[labels[0]] == capacities[labels[1]],
            'scope': '相同启动参数、冻结源码、固定bank与worker配置；自动KV容量可能随启动内存改变，'
                     '如实记录，不宣称调度轨迹或所有中间浮点结果逐bit一致。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--mean-range', nargs=2, type=float, default=(3.0, 4.0),
                        help='主批3 4，其他batch指定1 5；不单独为某个实现调题')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compact-details', action='store_true',
                        help='完整逐rank明细压缩保存，可读JSON只保留汇总和明确标注的rank0样例')
    parser.add_argument('--native-repeat', type=Path,
                        help='相同配置再次运行全Native的结果目录；仅CPU合并为六组，不重跑旧五组')
    parser.add_argument('--pto-repeat', type=Path,
                        help='相同配置再次运行性能CSA+PTO HCA的结果目录；与Native第二遍合并为七组')
    args = parser.parse_args()
    report = {'configurations': {}, 'pairwise': {}, 'errors': [], 'batch': args.batch,
              'scope': '固定输入的整模型生成token与DSpark；首个分歧后上下文不同，不当作同输入逐层浮点误差',
              'acceptance_policy': '保持同一批输入，不为某个实现单独调题；逐项报告均值/六种边界资格'}
    outputs, configs = {}, {}
    repeats = {}
    if args.native_repeat:
        repeats['native2'] = ('native', args.native_repeat, 'Native2（全Native同配置第二遍）', 'native_repeat')
    if args.pto_repeat:
        repeats['pto2'] = ('csa_performance_pto_hca', args.pto_repeat,
                           'PTO2（CSA性能版 + PTO HCA第二遍）', 'pto_repeat')
    names = []
    for original in CONFIGURATIONS:
        names.append(original)
        names.extend(name for name, repeat in repeats.items() if repeat[0] == original)
    for name in names:
        configuration = repeats[name][0] if name in repeats else name
        root = repeats[name][1] if name in repeats else args.root / name
        try:
            check, outputs[name], configs[name] = validate(
                root, args.bank, args.batch, configuration, tuple(args.mean_range))
        except FileNotFoundError as exc:
            # 整组缺少任何rank都不能用rank0或已完成子集代替16卡精度结果。
            check = {'label': CONFIGURATIONS[configuration][0], 'source': str(root.resolve()),
                     'execution_status': 'INCOMPLETE', 'validation_errors': [str(exc)],
                     'complete_rank_files': sorted(p.name for p in root.glob('rank*.mixed.json')),
                     'completed_client_markers': sorted(p.name for p in root.glob('mixed_done_0_rank*')),
                     'finished_requests_per_rank': {
                         str(rank): len((root / f'request_stats_rank{rank}.jsonl').read_text().splitlines())
                         if (root / f'request_stats_rank{rank}.jsonl').exists() else 0 for rank in range(16)}}
            outputs[name], configs[name] = None, None
        if name in repeats:
            check['label'] = repeats[name][2]
        report['configurations'][name] = check
        report['errors'].extend(f'{name}: {e}' for e in check['validation_errors'])
    # PyPTO初始化会将进程级event设为硬件模式。显式报告这项运行时差异，
    # 不把它冒充算术配置相同，也不据此直接给token差异做因果归属。
    report['worker_runtime_config_rank0'] = {name: rows[0] for name, rows in configs.items() if rows}
    report['event_modes_per_rank'] = {}
    for name, rows in configs.items():
        if not rows or not configs['native']:
            continue
        report['event_modes_per_rank'][name] = [row[0]['cann_event_work_mode'] for row in rows]
        for rank, (actual, native) in enumerate(zip(rows, configs['native'])):
            def without_event(workers):
                return [{k: v for k, v in worker.items() if k != 'cann_event_work_mode'} for worker in workers]
            if without_event(actual) != without_event(native):
                report['errors'].append(f'{name}/rank{rank}: 除event模式外的worker实际配置与Native不同')
    report['event_mode_scope'] = ('Native默认软件event=0，PTO初始化使用硬件event=1；其余worker配置严格比较。'
                                 '本轮比较实际集成路径，尚未用相同event模式隔离差异原因。')
    for left, right in itertools.combinations(names, 2):
        report['pairwise'][f'{left}__vs__{right}'] = (
            compare_rows(outputs[left], outputs[right]) if outputs[left] and outputs[right]
            else {'status': 'NOT_COMPARED', 'reason': '至少一组16卡执行不完整，不比较部分结果'})
    for name, (original, repeated_root, _, prefix) in repeats.items():
        if outputs[original] and outputs[name]:
            report[prefix + '_conditions'] = repeat_conditions(
                args.root / original, repeated_root, configs[original], configs[name], (original, name))
            report['errors'].extend(report[prefix + '_conditions']['errors'])
            report[prefix + '_status'] = report['pairwise'][f'{original}__vs__{name}']['status']
    report['status'] = ('FAIL' if report['errors'] or any(r['status'] != 'PASS'
                        for key, r in report['pairwise'].items() if key.startswith('native__vs__')) else 'PASS')
    if args.compact_details:
        full_path = args.output.with_suffix('.full.json.gz')
        with gzip.open(full_path, 'wt', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False)
        report['full_report'] = full_path.name
        report['detail_scope'] = '统计覆盖全部16rank；可读样例仅rank0，不代表其余rank；完整逐rank明细见full_report'
        for comparison in report['pairwise'].values():
            if 'details' in comparison:
                comparison['rank0_examples'] = [row for row in comparison.pop('details') if row['rank'] == 0]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'errors': report['errors'],
                      'vs_native': {k: {a: b for a, b in v.items() if a not in ('details', 'rank0_examples')}
                                    for k, v in report['pairwise'].items() if k.startswith('native__vs__')}},
                     ensure_ascii=False))
    raise SystemExit(report['status'] != 'PASS')


if __name__ == '__main__':
    main()
