# SPDX-License-Identifier: Apache-2.0
"""Parse saved device captures after the NPU worker has exited (CPU-only)."""
import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('report',type=Path)
args=p.parse_args()
d=json.loads(args.report.read_text())
for row in d['cases']:
    for side, result in row.get('device',{}).items():
        if 'median_us' in result:continue
        raw=args.report.parent/result['raw']
        with (raw.parent/'cann_export.log').open('w') as log:
            subprocess.run(['msprof','--export=on','--output='+str(raw.resolve())],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        csvs=sorted(raw.glob('mindstudio_profiler_output/op_summary_*.csv'))
        if not csvs:raise RuntimeError(f'Missing op_summary under {raw}')
        with csvs[-1].open() as f:
            records=[r for r in csv.DictReader(f) if r['OP Type']==result['kernel']]
        samples=[float(r['Task Duration(us)']) for r in records]
        if len(samples)!=3:raise RuntimeError(f'Expected three {result["kernel"]} samples: {samples}')
        result.update(status='parsed',samples_us=samples,median_us=statistics.median(samples),
                      trace=str(csvs[-1].relative_to(args.report.parent)))
        args.report.write_text(json.dumps(d,indent=2)+'\n')
        print(row['name'],side,result['median_us'],flush=True)
