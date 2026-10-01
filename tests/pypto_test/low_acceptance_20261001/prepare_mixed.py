# SPDX-License-Identifier: Apache-2.0
"""固定40条不同问题，复用审计前缀；实际接受分布只能由模型实测。"""
import argparse
import importlib.util
import json
from pathlib import Path

from low_acceptance_20261001.questions import QUESTIONS
from tokenizers import Tokenizer


def questions():
    templates = dict(QUESTIONS)
    groups = {'python_trace': [], 'transition_trace': [], 'ledger_transform': [], 'easy_repeat': []}
    for i in range(16):
        q = templates['python_trace']
        q = q.replace('[19, 7, 41, 13, 29, 3, 37]', str([19 + i, 7 + 2*i, 41 - i, 13 + i, 29 + i, 3 + i, 37 + 2*i]))
        q = q.replace('s = 23', f's = {23 + 7*i}').replace('a * 17', f'a * {17 + 2*(i % 5)}')
        groups['python_trace'].append(q)
    for i in range(12):
        q = templates['transition_trace'].replace('S=4、寄存器R=137', f'S={i % 11}、寄存器R={137 + 13*i}')
        q = q.replace('31*a', f'{31 + 2*i}*a').replace('17*S', f'{17 + i}*S')
        groups['transition_trace'].append(q)
    for i in range(6):
        q = templates['ledger_transform'].replace('3*fee', f'{i + 2}*fee')
        q = q.replace('net*7', f'net*{7 + 2*i}').replace('fee*11', f'fee*{11 + 2*i}')
        groups['ledger_transform'].append(q)
    for i, sentence in enumerate(['缓存保持一致。', 'The cache is consistent.', '先读取，再计算，最后写回。',
                                  '一个请求，一份状态。', 'Read, compute, write.', '天地玄黄，宇宙洪荒。']):
        groups['easy_repeat'].append(
            f'背景材料结束。请连续重复下面这句话{80 + i}次，用换行分隔，不编号、不解释、不省略：\n{sentence}')
    # 按固定比例交错，较小batch也取得多类请求；不是按接受率筛完后排序挑最快样本。
    schedule = ['python_trace', 'transition_trace', 'python_trace', 'ledger_transform',
                'transition_trace', 'easy_repeat'] * 6 + ['python_trace'] * 4
    result = []
    for kind in schedule:
        result.append((kind, groups[kind].pop(0)))
    assert len(result) == 40 and len({q for _, q in result}) == 40
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bank', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--encoding', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_bank.resolve()
    plan = json.loads((source / 'plan.json').read_text())
    if json.loads((source / 'audit.json').read_text())['status'] != 'PASS':
        raise ValueError('源bank必须已审计')
    base = plan['cases'][0]
    history = base['history']
    prefix = json.loads((source / base['tokens']).read_text())[:history]
    spec = importlib.util.spec_from_file_location('chat_encoding', args.encoding)
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    tokenizer = Tokenizer.from_file(str(Path(plan['model']) / 'tokenizer.json'))
    args.bank.mkdir(parents=True, exist_ok=False)
    cases, records = [], []
    for i, (kind, question) in enumerate(questions()):
        key = f'h{history}_q{i:02d}'
        rendered = encoding.encode_messages([{'role': 'user', 'content': question}], thinking_mode='chat')
        rendered = rendered.removeprefix(encoding.bos_token)
        suffix = tokenizer.encode('\n\n' + rendered, add_special_tokens=False).ids
        (args.bank / key).symlink_to(source / base['key'], target_is_directory=True)
        token_file = f'{key}.tokens.json'
        (args.bank / token_file).write_text(json.dumps(prefix + suffix) + '\n')
        cases.append({'key': key, 'tokens': token_file, 'history': history,
                      'decode_suffix_tokens': len(suffix), 'p_dp_rank': i % 16, 'question_kind': kind})
        records.append({'key': key, 'kind': kind, 'question': question,
                        'suffix_tokens': len(suffix), 'cached_history': history})
    plan.update(cases=cases, inherited_prefix_bank=str(source),
                boundary='相同已审计前缀+40个不同真实问题；后缀必须真实prefill',
                mixed_requests=True)
    (args.bank / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    (args.bank / 'audit.json').write_text(json.dumps({
        'status': 'PASS', 'scope': '继承源cache审计，所有问题只追加未缓存后缀；源cache不写入',
        'source_bank': str(source), 'source_case': base['key'], 'source_audit': str(source / 'audit.json')},
        ensure_ascii=False, indent=2) + '\n')
    args.record.write_text(json.dumps({'bank': str(args.bank.resolve()), 'cases': records},
                                     ensure_ascii=False, indent=2) + '\n')
    print(f'prepared {len(cases)} distinct questions, H={history}')


if __name__ == '__main__':
    main()
