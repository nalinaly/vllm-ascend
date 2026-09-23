"""Read the three bounded regression results; no device execution."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
import torch

torch.set_num_threads(2)
root = Path(__file__).resolve().parent
repo = root.parents[4]
manifest = json.loads((root / 'manifest.json').read_text())
report = {'status': 'PENDING', 'scope': manifest['scope'], 'source_hashes_match': all(
    hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected
    for path, expected in manifest['sources'].items()), 'cases': [], 'pending': []}
queue = []
for task in manifest['tasks']:
    state_text = subprocess.run(['task-submit', '--status', task['task']], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    queue.append({'task': task['task'], 'status_output': state_text})
    output = Path(task['output'])
    if 'completed' not in state_text:
        report['pending'].append(task['task'])
        continue
    current = json.loads((output / (task['phase'] + '.json')).read_text())
    row = {'case': output.name, 'task': task['task'], 'status': current['status'], 'queue_exit_zero': bool(re.search(r'completed.*exit=0', state_text)), 'equivalent': False}
    if task['phase'] == 'full_compare':
        checks = [current['output'], current['topk'], *current['cache_comparisons'].values(), *current['native_untouched'].values(), *current['pto_untouched'].values()]
        old = torch.load(root.parent / 'adapter_redundancy_v1' / output.name / 'outputs.pt', map_location='cpu', weights_only=False)
        new = torch.load(output / 'outputs.pt', map_location='cpu', weights_only=False)
        row.update(checks=len(checks), all_checks_pass=all(check['status'] == 'PASS' for check in checks), output=current['output'], native_rope_zero_copy=current['native_rope_zero_copy'], native_indices_zero_copy=current['native_indices_zero_copy'], old_new_bitwise={key: torch.equal(old[key].contiguous().view(torch.uint8), new[key].contiguous().view(torch.uint8)) for key in ('pto', 'pto_topk', 'pto_scores')})
        row['equivalent'] = row['all_checks_pass'] and row['native_rope_zero_copy'] and all(row['native_indices_zero_copy'].values()) and all(row['old_new_bitwise'].values())
    else:
        checks = [check for case in current['cases'] for check in case['checks'].values()]
        files = list((output / 'graph_profile').rglob('kernel_details.csv'))
        assert len(files) == 1, files
        with files[0].open() as stream:
            kernels = list(csv.DictReader(stream))
        profile = {'path': str(files[0]), 'simpler_aicpu': sum('simpler_aicpu_kernel_exec' in k['Name'] for k in kernels), 'aicore_kernel_mode': sum('aicore_kernel_mode' in k['Name'] for k in kernels)}
        row.update(checks=len(checks), all_checks_pass=all(check['status'] == 'PASS' for check in checks), fixed_addresses=current['fixed_addresses'], native_external_metadata_events=current['native_external_metadata_events'], profile=profile)
        row['equivalent'] = row['all_checks_pass'] and row['fixed_addresses'] and row['native_external_metadata_events'] and profile['simpler_aicpu'] == 7 and profile['aicore_kernel_mode'] == 7
    row['equivalent'] &= row['queue_exit_zero'] and row['status'] == 'PASS'
    report['cases'].append(row)
if not report['pending']:
    report['status'] = 'PASS' if report['source_hashes_match'] and all(row['equivalent'] for row in report['cases']) and len(report['cases']) == 3 else 'FAIL'
(root / 'queue_snapshot.json').write_text(json.dumps(queue, indent=2) + '\n')
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
