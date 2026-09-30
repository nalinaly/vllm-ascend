# SPDX-License-Identifier: Apache-2.0
"""CPU比较冻结任务的输出、TopK和cache/state写入区，不做hash。"""
import argparse
import json
from pathlib import Path

import torch


def compare(reference, candidate):
    expected = torch.load(reference / 'outputs.pt', map_location='cpu', weights_only=True)
    actual = torch.load(candidate / 'outputs.pt', map_location='cpu', weights_only=True)
    assert expected.keys() == actual.keys()
    result = {}
    for name, value in actual.items():
        baseline = expected[name]
        assert baseline.shape == value.shape and baseline.dtype == value.dtype, name
        mismatch = (value != baseline).sum().item()
        item = {'elements': value.numel(), 'mismatches': mismatch, 'exact': mismatch == 0}
        if value.is_floating_point():
            delta = torch.where(value == baseline, 0.0, value.float() - baseline.float())
            item.update(max_abs=delta.abs().max().item(), rmse=delta.square().mean().sqrt().item(),
                        nonfinite=int((~torch.isfinite(value)).sum()))
        if name == 'idx_topk':
            item['different_sets'] = int((value.sort(dim=1).values != baseline.sort(dim=1).values).any(dim=1).sum())
        result[name] = item
    return {'reference': str(reference), 'candidate': str(candidate),
            'all_exact': all(item['exact'] for item in result.values()), 'tensors': result}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = compare(args.reference, args.candidate)
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    print(output)
