# SPDX-License-Identifier: Apache-2.0
"""比较两种 CSA 版本：原始位模式与整模型 token/DSpark，后者两侧均为 PTO。"""
import argparse
import json
import re
from pathlib import Path


def bits(root):
    import torch
    from offline_pd.run import ulp_gap

    torch.set_num_threads(4)
    sides = {name: torch.load(root / name / 'outputs.pt', map_location='cpu', weights_only=True)
             for name in ('performance', 'precision')}
    assert sides['performance'].keys() == sides['precision'].keys()
    popcount = torch.tensor([i.bit_count() for i in range(256)], dtype=torch.int64)
    report = {'scope': '真实第2层权重、合成固定历史输入；单层 CSA 张量，不是整模型 hidden/logits',
              'source': str(root.resolve()), 'arithmetic': '两侧 atomic_add=0，单卡样本 deterministic=0',
              'tensors': {}}
    for name, left in sides['performance'].items():
        right = sides['precision'][name]
        assert left.dtype == right.dtype and left.shape == right.shape, name
        a, b = left.contiguous(), right.contiguous()
        xor = torch.bitwise_xor(a.view(torch.uint8), b.view(torch.uint8)).reshape(-1, a.element_size())
        changed = xor.ne(0).any(dim=1)
        item = {'dtype': str(a.dtype), 'shape': list(a.shape), 'elements': a.numel(),
                'bitwise_different_elements': int(changed.sum()),
                'different_element_percent': float(changed.float().mean() * 100),
                'different_bits': int(popcount[xor.long()].sum()),
                'total_bits': a.numel() * a.element_size() * 8,
                'exact': not bool(changed.any())}
        if name.startswith('state.'):
            item['scope'] = '允许写入区域的原始字节；不将字节差当作浮点误差'
        if a.is_floating_point():
            finite = torch.isfinite(a) & torch.isfinite(b)
            delta = a.float()[finite] - b.float()[finite]
            gap = ulp_gap(a, b)[finite]
            item.update(finite_pairs=int(finite.sum()),
                        unmatched_nonfinite=int(((a != b) & ~finite).sum()),
                        max_abs=float(delta.abs().max()),
                        rmse=float(delta.square().mean().sqrt()),
                        mean_abs=float(delta.abs().mean()),
                        ulp_p50=float(gap.float().quantile(0.50)),
                        ulp_p95=float(gap.float().quantile(0.95)),
                        ulp_max=int(gap.max()))
        if name == 'CSA':
            worst = int(ulp_gap(a, b).reshape(-1).argmax())
            item['max_ulp_example'] = {'performance': float(a.reshape(-1)[worst]),
                                       'precision': float(b.reshape(-1)[worst])}
        if name == 'idx_topk_scores':
            item['scope'] = '按输出槽位比较；两侧 Top-K ID 不同，不是相同 ID 的分数对照'
        if name == 'idx_topk':
            sa, sb = a.sort(dim=1).values, b.sort(dim=1).values
            item['different_set_rows'] = int((sa != sb).any(dim=1).sum())
            item['rows'] = a.shape[0]
            item['mean_set_overlap_percent'] = float(torch.stack([
                torch.isin(x, y).float().mean() for x, y in zip(a, b)]).mean() * 100)
        report['tensors'][name] = item
    return report


def tokens(root, bank):
    from csa_hca_merge_20260930.functional import validate
    from offline_pd.compare import _load_rank

    plan = json.loads((bank / 'plan.json').read_text())
    report = {'scope': '16卡整模型：同一 HCA，仅切换 CSA 性能版/精度版',
              'source': str(root.resolve()), 'status': 'FAIL', 'errors': [],
              'expected_ranks': 16, 'batch': 16, 'decode_tokens': 192,
              'compared_ranks': 0, 'compared_tokens': 0, 'token_mismatches': 0,
              'dspark_mismatched_cases': 0, 'cases': [], 'functional': {}, 'static_kernel': {}}
    for variant in ('performance', 'precision'):
        report['functional'][variant] = validate(root / variant, plan, 16, 192)
        report['errors'].extend(f'{variant}: {e}' for e in report['functional'][variant]['errors'])
        leader_log = (root / variant / 'rank0.log').read_text(errors='replace')
        counts = [int(x) for x in re.findall(r'OFFLINE_STATIC_KERNEL[^\n]*installed_packages=(\d+)', leader_log)]
        compiler_errors = leader_log.count('UserWarning: Rank 0: execute op_compiler error')
        report['static_kernel'][variant] = {'installed_packages': counts, 'compiler_errors': compiler_errors}
        if not counts or min(counts) <= 0 or compiler_errors:
            report['errors'].append(f'{variant}: static kernel 未完整编译/安装成功')
    for rank in range(16):
        keys = [c['key'] for c in plan['cases'] if c['p_dp_rank'] == rank % 4]
        loaded, configs = {}, {}
        for variant in ('performance', 'precision'):
            data = json.loads((root / variant / f'rank{rank}.json').read_text())
            expected = {'variant': variant, 'atomic_add': '0', 'deterministic': True,
                        'hccl_deterministic': 'true', 'pto_runtime': 'tensormap_and_ringbuffer',
                        'eplb_enabled': False, 'pto_attention': 'both', 'decode_dp': 16}
            for key, value in expected.items():
                if data.get(key) != value:
                    report['errors'].append(f'{variant}/rank{rank}: {key}={data.get(key)!r}, expected={value!r}')
            configs[variant] = data['worker_runtime_config']
            loaded[variant] = _load_rank(root / variant, 'pto', rank, keys, 16, 192, 5)
        if configs['performance'] != configs['precision']:
            report['errors'].append(f'rank{rank}: 两侧 worker 实际运行配置不同')
        if rank == 0:
            report['worker_runtime_config'] = configs
        report['compared_ranks'] += 1
        for left, right in zip(loaded['performance'], loaded['precision']):
            mismatches, examples = 0, []
            for request, (a, b) in enumerate(zip(left['output_token_ids'], right['output_token_ids'])):
                for position, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        mismatches += 1
                        if len(examples) < 8:
                            examples.append({'request': request, 'position': position,
                                             'performance': x, 'precision': y})
            stats = {v: c['validated_spec_decode'] for v, c in
                     (('performance', left), ('precision', right))}
            fields = [k for k in stats['performance'] if stats['performance'][k] != stats['precision'][k]]
            report['cases'].append({'rank': rank, 'key': left['key'], 'token_mismatches': mismatches,
                                    'first_mismatches': examples, 'spec_decode': stats,
                                    'spec_decode_different_fields': fields,
                                    'unique_performance_token_ids': len(set(sum(left['output_token_ids'], [])))})
            report['token_mismatches'] += mismatches
            report['compared_tokens'] += 16 * 192
            report['dspark_mismatched_cases'] += bool(fields)
    report['spec_decode_totals'] = {
        v: {k: sum(c['spec_decode'][v][k] for c in report['cases'])
            for k in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens')}
        for v in ('performance', 'precision')}
    report['token_status'] = 'PASS' if not report['errors'] and not report['token_mismatches'] else 'FAIL'
    if report['token_status'] == 'PASS' and not report['dspark_mismatched_cases']:
        report['status'] = 'PASS'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['bits', 'tokens'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = bits(args.root) if args.mode == 'bits' else tokens(args.root, args.bank)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    summary = {k: v for k, v in result.items() if k not in ('cases', 'functional')}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.mode == 'tokens' and result['status'] != 'PASS':
        raise SystemExit(1)
