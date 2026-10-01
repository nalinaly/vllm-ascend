# SPDX-License-Identifier: Apache-2.0
"""为一次性Native后缀prefill准备固定bank，并在导出后核对边界和序列化结构。"""
import argparse
import json
import shutil
from pathlib import Path


def prepare(source, destination):
    plan = json.loads((source / 'plan.json').read_text())
    destination.mkdir(parents=True, exist_ok=False)
    for case in plan['cases']:
        shutil.copyfile(source / case['tokens'], destination / case['tokens'])
        case['source_history'] = case['history']
        case['history'] += case['decode_suffix_tokens'] - 1
        case['decode_suffix_tokens'] = 1
    plan.pop('inherited_prefix_bank', None)
    plan.update(extended_from_bank=str(source.resolve()), prefill={'tp': 1, 'dp': 16, 'ep': 16},
                boundary='Native在TP1/EP16下一次性prefill问题后缀；后续恢复同一份固定target/draft cache')
    plan['layout']['weight_nz_mode'] = 2
    (destination / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')


def audit(bank):
    from offline_pd.prefix import cache_contract
    from safetensors import safe_open

    plan = json.loads((bank / 'plan.json').read_text())
    expected = cache_contract(json.loads((Path(plan['model']) / 'config.json').read_text()))
    rows = []
    for case in plan['cases']:
        folder = bank / case['key'] / 'tp0'
        manifest = json.loads((folder / 'manifest.json').read_text())
        tokens = json.loads((bank / case['tokens']).read_text())
        assert len(tokens) == case['history'] + 1 == manifest['history'] + 1
        assert manifest['p_dp_rank'] == case['p_dp_rank'] and manifest['p_tp_rank'] == 0
        with safe_open(folder / 'cache.safetensors', framework='pt', device='cpu') as payload:
            assert set(payload.keys()) == set(manifest['entries']) == set(expected)
            for name, entry in manifest['entries'].items():
                tensor = payload.get_slice(name)
                assert tensor.get_shape() == [len(entry['logical_blocks']), *entry['shape']]
                assert entry['compress_ratio'] == expected[name]
        rows.append({'key': case['key'], 'history': case['history'], 'tensors': len(expected),
                     'payload_bytes': (folder / 'cache.safetensors').stat().st_size})
    report = {'status': 'PASS', 'scope': '核对Native导出边界/owner/完整target与draft覆盖/结构；无hash或全量位差扫描',
              'cases': rows}
    (bank / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'PASS: {len(rows)} Native fixed request caches')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'audit'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--bank', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source, args.bank)
    else:
        audit(args.bank)
