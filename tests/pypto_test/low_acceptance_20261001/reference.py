# SPDX-License-Identifier: Apache-2.0
"""导出/检查固定混合请求参考输出；直接比较token和接受统计，不使用hash。"""
import argparse
import gzip
import json
from pathlib import Path


def extract(root):
    rows = []
    for rank in range(16):
        result = json.loads((root / f'rank{rank}.mixed.json').read_text())
        events = {r['key']: r['events'] for r in map(json.loads,
                  (root / f'request_stats_rank{rank}.jsonl').read_text().splitlines())}
        for request in result['requests']:
            steps = events[request['key']]
            rows.append({'rank': rank, 'key': request['key'], 'tokens': request['output_token_ids'], 'events': steps,
                         'drafts': len(steps), 'proposed': sum(v[0] for v in steps),
                         'accepted': sum(v[1] for v in steps),
                         'histogram': [sum(v[1] == n for v in steps) for n in range(6)]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export', 'check'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='check结果JSON')
    args = parser.parse_args()
    rows = extract(args.root)
    if args.command == 'export':
        # 参考一经生成不覆盖；新版需使用新文件并明确记录迁移原因。
        with gzip.open(args.reference, 'xt', encoding='utf-8') as stream:
            json.dump({'schema': 2, 'source': str(args.root.resolve()), 'rows': rows,
                       'scope': '指定实现的回归参考，不代表Native输出正确性的独立证明'}, stream, ensure_ascii=False)
        print(f'exported {len(rows)} requests to {args.reference}')
        return
    if args.output is None:
        parser.error('check要求--output')
    with gzip.open(args.reference, 'rt', encoding='utf-8') as stream:
        expected = json.load(stream)['rows']
    if [(r['rank'], r['key']) for r in rows] != [(r['rank'], r['key']) for r in expected]:
        raise ValueError('参考与新结果的rank/请求数量/顺序不符')
    differences = []
    for old, new in zip(expected, rows):
        fields = [k for k in ('tokens', 'drafts', 'proposed', 'accepted', 'histogram') if old[k] != new[k]]
        if 'events' in old and old['events'] != new['events']:
            fields.append('events')
        if fields:
            differences.append({'rank': old['rank'], 'key': old['key'], 'fields': fields})
    result = {'status': 'FAIL' if differences else 'PASS', 'requests': len(rows), 'differences': differences}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'{result["status"]}: {len(differences)}/{len(rows)} requests differ from the pinned reference')
    raise SystemExit(bool(differences))


if __name__ == '__main__':
    main()
