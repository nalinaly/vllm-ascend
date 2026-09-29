# 同进程 PTO 候选筛选

目的：减少每个候选重新加载权重的成本，尝试减小跨进程漂移。它直接注册两个独立命名的
PTO root，复用正式第3层权重和同一份Native metadata/cache；两张独立的npugraph_ex图
按ABBA交错重放。dynamic=False、inplace/static开启，PTO不启用SuperKernel。

这只是候选筛选，不代替生产服务入口、Native对照或整模型token验收。只接受数值中性候选：
各自eager与graph、两侧完整输出/cache/state必须逐bit一致，保护区必须通过。

## 已完成的校准

2026-09-30，CANN9.2、NZ2、atomic0、128K/B16；每侧10次纯设备span，单位μs：

| 对照 | base min/max/mean | candidate min/max/mean |
| --- | --- | --- |
| 同源码A/A | 579.50 / 609.50 / 590.58 | 585.00 / 623.25 / 595.40 |
| O-B激活复用 | 581.00 / 622.00 / 589.38 | 582.00 / 606.25 / 595.45 |

A/A仍有4.825μs均值差；本工具没有证明能稳定辨认几微秒的优化，更不能减去这次A/A差值
作为另一轮的“校正”。O-B在这个窗口没有整体收益，最大值降低只是线索。
每次重放均为1个AICPU和1个AICore执行核，20段顺序与ABBA一致；原始CSV和span保留。
profiler报告缺少ACL到NPU的flow连线，CPU/NPU因果连线不可用于分析；纯设备CSV存在且分段数通过检查。

任务：A/A为`task_20260930_020143_94987910516`，O-B为`task_20260930_020316_98154225752`。
结果摘要见[result_summary.json](result_summary.json)，原始目录位于
`tests/pypto_test/results/hca_pair_screen_20260930/`。
首次A/A任务`task_20260930_015853_88851217018`因测试脚本遗漏`enable_custom_op()`初始化，
在Native metadata构造时失败；已补齐，失败任务没有性能数据。

## 复现

先在CPU使用`--check-only`确认两个别名的独立注册；这一步不代表设备编译通过。
候选应另用`hca_residual_reuse_20260930/compile.py`显式编译后上卡。

```bash
task-submit --device auto --max-time 900 --timeout 45 --run \
  'bash /绝对路径/hca_pair_screen_20260930/run.sh /新的结果目录 --baseline /冻结基线包 --candidate /冻结候选包 --history 131072 --batch 16'
```

算子包必须在执行前冻结，运行中不编辑源码。默认5个ABBA周期，首次筛选不扩大采样。
有明确收益后再用既有服务入口脚本对照，并核对相关任务的泳道。
