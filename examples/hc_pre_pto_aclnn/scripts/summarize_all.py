# SPDX-License-Identifier: Apache-2.0
"""Update the comparison Markdown from a completed or partial all-case report."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('report',type=Path)
p.add_argument('--output',type=Path,help='Write a standalone run summary without replacing historical comparisons')
args=p.parse_args()
run_report = json.loads(args.report.read_text())
if args.output is not None or run_report.get('weight_source', {}).get('mode') == 'random':
    target = args.output or args.report.parent / 'summary.md'
    source = run_report.get('weight_source', {'mode': 'checkpoint', 'layer': 2, 'branch': 'attn'})
    lines = ['# HC_pre 单轮测试结果', '',
             f"权重来源：`{json.dumps(source, ensure_ascii=False)}`。", '',
             '随机权重结果不能视为 10-12 版本真实权重测试的同条件复现。核耗时为设备 profiler 三次中位数，单位 μs。', '',
             '|B|S|T|Native kernel|ACLNN PTO kernel|PTO 对 Native|PTO 对 CPU|Native 对 CPU|更新输入 PTO 对 CPU|更新输入 Native 对 CPU|',
             '|---:|---:|---:|---:|---:|---|---|---|---|---|']
    def verdict(group):
        return '未完成' if not group else ('通过' if all(v['passed'] for v in group.values()) else '未通过')
    for row in run_report['cases']:
        timings = [row.get('device', {}).get(k, {}).get('median_us') for k in ('native', 'pto')]
        values = [f'{v:.2f}' if v is not None else '未采集' for v in timings]
        checks = [row.get('pto_vs_native'), row.get('cpu', {}).get('pto'), row.get('cpu', {}).get('native'),
                  row.get('pto_updated_cpu'), row.get('native_updated_cpu')]
        lines.append('|' + '|'.join([str(row['B']), str(row['S']), str(row['T']), *values,
                                    *[verdict(g) for g in checks]]) + '|')
    lines += ['', '图重放样本、逐输出误差和权重 SHA256 见同目录 report.json；阈值沿用 10-12 版本 benchmark，未因随机权重调整。', '']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines))
    print(target)
    raise SystemExit(0)
root=Path(__file__).resolve().parents[1]
baseline_archive=json.loads((root.parents[1]/'tests/pypto_test/hc_pre_optimization_20261012/latest14_summary.json').read_text())
old={(r['batch_size'],r['seqlen']):r for r in baseline_archive['rows']}
report=json.loads(args.report.read_text())
baseline=json.loads((root/'reports/original_pto_baseline.json').read_text())
local={(r['B'],r['S']):r for r in baseline['runs']['device_baseline_1']['rows']}
repeat={(r['B'],r['S']):r for r in baseline['runs']['device_baseline_0']['rows']}
md=root/'全场景性能对比.md'
s=md.read_text().split('## 3. ACLNN 接入测试结果')[0]
s+='''## 3. ACLNN 接入测试结果

Native 实测 / ACLNN PTO 实测两列均为设备 profiler 的 kernel 耗时，3 次取中位数，单位 μs。
10-12 版本 PTO 与原版 PTO 实测均取完整 L1 跨度；原版实测列复用此前同源码的较后一轮完整实测（未接 ACLNN）。
ACLNN 列取 CANN kernel 时间，与 L1 的统计起止不同，不能把两者差值直接解释为接入开销。
Native 源码双方均为 `cann-recipes-infer@4c3d1e1258053c4d810322198a7fd1722c89897e`。

|B|S|T|10-12 版本 Native|10-12 版本 PTO L1|原版 PTO L1 实测|Native 实测|ACLNN PTO 实测|Native 较 10-12 版本|ACLNN PTO 较 Native|PTO/Native 精度|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
'''
for r in report['cases']:
    o=old[(r['B'],r['S'])]
    if not all('median_us' in r.get('device',{}).get(k,{}) for k in ('native','pto')):continue
    n=r['device']['native']['median_us'];q=r['device']['pto']['median_us']
    ok=all(v['passed'] for v in r['pto_vs_native'].values())
    s+=f"|{r['B']}|{r['S']}|{r['T']}|{o['native_p50_us']:.2f}|{o['pto_l1_p50_us']:.2f}|{local[(r['B'],r['S'])]['pto_l1_us']:.2f}|{n:.2f}|{q:.2f}|{(n/o['native_p50_us']-1)*100:+.1f}%|{(q/n-1)*100:+.1f}%|{'通过' if ok else '未通过'}|\n"
s+='\n### 同源码原版 PTO：两轮实测参考（未接 ACLNN）\n\n主表固定使用较后一轮 baseline_1，不按性能挑选较快值；每轮各 3 次取中位数。\n\n|B/S|10-12 版本 PTO L1|原版 baseline_0|原版 baseline_1（主表）|主表较 10-12 版本|\n|---|---:|---:|---:|---:|\n'
for key, r in local.items():
    q=r['pto_l1_us']; o=old[key]['pto_l1_p50_us']
    s+=f"|{key[0]}/{key[1]}|{o:.2f}|{repeat[key]['pto_l1_us']:.2f}|{q:.2f}|{(q/o-1)*100:+.1f}%|\n"
s+='\n数据来源：2026-10-13 的两轮原版复现，分别记为 baseline_0、baseline_1；使用未修改的 10-12 版本源码。样本及报告校验值已保存到 `reports/original_pto_baseline.json`。\n\n两轮均 14/14 通过，均在物理设备 0；主表左侧的 ACLNN 首轮在设备 8，因此左侧数据不是同卡配对运行。原版与 10-12 版本（`f04320413`）的源码及当次记录的 SHA256 一致：`ff90619b80ade29e673dd742d1c6fe42e237f5aa3ad2f6b4f300f80baec3df0c`。权重层、分支、输入种子、B/S 均一致。\n\n原版使用 PyPTO `00d5ab9f651b03a1f459abb30c58f143a4880cfa`、Simpler `17e342455e2e8d9112c0613b6c9cd6eca79e71d0`，包含 PyPTO 与 Simpler 的 ABI 适配；“算子源码相同”不代表 10-12 版本归档与本次测试的软件环境相同。三次样本、两轮结果与报告校验值见 [原版 PTO 数据来源](reports/original_pto_baseline.json)。\n'
s+='''
百分比为耗时变化，负数表示更快。精度列比较 ACLNN PTO 与 Native 的三个输出。

### 完整图重放（单列，单位 μs）

图外 NPU Event 包住一次 replay，交替测两侧各 20 次取中位数；不等于上表 kernel 时间。

|B|S|T|Native 图重放|PTO 图重放|
|---:|---:|---:|---:|---:|
'''
for r in report['cases']:
    if 'replay' not in r:continue
    s+=f"|{r['B']}|{r['S']}|{r['T']}|{r['replay']['native']['median_us']:.2f}|{r['replay']['pto']['median_us']:.2f}|\n"
s+='''
### 验证详情

|B/S|普通调用 PTO 对 CPU|Native 对 CPU|图重放与普通调用|更新输入后重放与普通调用|更新输入 PTO 对 CPU|更新输入 Native 对 CPU|
|---|---|---|---|---|---|---|
'''
def passed(*ds):return '通过' if all(v['passed'] for d in ds for v in d.values()) else '未通过（见原始误差）'
for r in report['cases']:
    s+=f"|{r['B']}/{r['S']}|{passed(r['cpu']['pto'])}|{passed(r['cpu']['native'])}|{passed(r['pto_graph_vs_eager'],r['native_graph_vs_eager'])}|{passed(r['pto_updated_eager'],r['native_updated_eager'])}|{passed(r['pto_updated_cpu'])}|{passed(r['native_updated_cpu'])}|\n"
s+='''
阈值取自 10-12 版本 benchmark：BF16 atol/rtol=0.01/0.01，FP32=1e-5/1e-4；逐元素判定，不是已核实的官方验收标准。
普通调用、图重放及修改输入的逐输出误差保存在报告中；CPU 参考使用 FP32。

## 4. 复现与原始数据

- 执行方式和依赖：[README](README.md)。
- 权重：DeepSeek-V4-Flash-0731-w8a8，第 2 层 attn 的 fn/scale/base，报告记录每份权重 SHA256。
- 输入：BF16，seed=20261009+B×100+S，与 10-12 版本十四场景设置相同。
- 10-12 版本历史采集与本次并非同一次运行，编译环境、设备状态差异仍可能影响耗时。
'''
s+="- 随仓数据摘要：[performance_samples.json](reports/performance_samples.json)；运行时完整报告与 trace 保存在指定输出目录。\n"
s+=f"- Torch / torch_npu：`{report['torch']}` / `{report['torch_npu']}`。\n"
s+=f"- 已记录 {len(report['cases'])}/14 个场景；只有上表已填入设备耗时的场景完成性能采集。\n"
s+=f"- 本次物理设备：`{report.get('physical_device', '见任务日志')}`。\n"
md.write_text(s)
print(md)
