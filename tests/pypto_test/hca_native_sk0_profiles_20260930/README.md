# HCA七档 Native SK0 task展开图与最终PTO泳道（2026-09-30）

**21个JSON已齐：每档Native profiling、最终PTO profiling、最终PTO单次泳道各一份。**
七档为128K B4/B8/B16/B24、8K B16/B24/B32。目录可直接整体下载：

`tests/pypto_test/results/hca_native_sk0_profiles_20260930/download_hca_7cases_sk0_task_details_29917c4e/`

## 内容与来源

- Native：复用 `results/hca_native_sk0_20260929/` 的七档完整原始采集。
  CANN9.2、NZ2、npugraph_ex dynamic=False、inplace/static开启、SK关闭，
  每档均为预热后3次单层图重放；不是本轮新跑，也没有从更长trace中截取。
- PTO：复用9月30日最终29917c4e的七档30步PyTorch profiling及独立单次泳道。
  没有重复占卡；PTO文件名明确标30Steps，不能误作新的三步采集。
- HCA第3层真实权重、合成历史，HC_pre+norm+HCA+HC_post区间，非CSA、非整网forward。
- 两侧采集日期不同，本包供查看kernel/task细节，不替换Native SK1正式性能基线。
  Native原report没有记录Git提交，仅保留原source路径与实际编译证据，不推断commit。

## 文件命名

前缀01–07依次对应七档，后跟上下文与batch。每档三份：

- `01_Native_SK0_Static1_PyTorch_3Steps.json`
- `02_PTO_Final_PyTorch_30Steps.json`
- `03_PTO_Swimlane_SingleHCA_SyntheticHistory.json`

下载目录内README和manifest列出每档源路径，Git内精简证据见 [summary.json](summary.json)。

## 已核实的内容

Native七档均MEASURED，实际static编译成功且安装包数为1，static_super_flags为false；
inplace/static开启，CANN与当前PTO采集一致，形状/NZ2/三个重放及保护区检查通过。
原始trace内每档能看到3次SparseAttnSharedkv和3次TransposeBatchMatMul。
128K/B4、B8每步23个kernel，其余五档每步24个，不再被单个SuperKernel隐藏。

`collect.py`读取原报告、链接原JSON并生成索引，不修改事件或时间戳，不进行hash扫描。
重新汇聚只需CPU命令：

```bash
python tests/pypto_test/hca_native_sk0_profiles_20260930/collect.py
```

## 排队记录与取舍

起初没有找到这套已有采集，错误地新增了七个重复任务；队列守护进程随后核实为未运行，
当前账号sudo需要密码，任务始终pending。进一步检索找到完整且符合用户三步要求的原始七档，
因此复用现有证据，七个重复任务已全部在启动前取消，未占卡，也没有运行16卡。
[原句柄](tasks.json)、[取消结果](task_status.json)、[队列状态记录](queue_status.json)保留审计。
本次文件交付不再依赖管理员恢复队列，不宣称队列服务已经修复。

无生产代码修改；定向Ruff和Git diff检查通过。
`bash format.sh ci`已执行，但因未安装pre-commit退出，未宣称全量格式检查通过。
SK1正式性能与完整状态验收见 [收尾报告](../hca_five_closeout_20260930/README.md)。
