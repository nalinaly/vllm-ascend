# HCA长档跨query延续流水（2026-09-30）

已接入。基线`bb4c2831`包含mHC residual复用、O-B激活复用和softmax max/sum UB环。
本轮只改HCA长档attention，不改Native流程、cache布局、任务数/依赖或公共工具链。
私有包与改动见[source.json](source.json)、[stream.patch](stream.patch)；
生产代码只补了注释和换行，与设备验证快照的Python AST一致。

## 修改及Native参考

参考本地`ops-transformer`提交`28f40354`的
`experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_swa_kernel.h:790`：
Native跨query递增gloop，只在当前核最后一项追加PRELOAD_NUM轮排空。
这不表示已证明安装包在本形状采用完全相同的内部分块。

旧PTO对每条query执行`work_count + 2`轮。新实现保持24组MIX核及原有三槽环，
让QK/PV跨query延续，只有worker最后一条query排空。128K/B16每worker有4条query，
因此从循环结构上省掉3次重复排空。

QK可以领先PV两步：每个AIV在UB环保存query编号、局部块号和剩余块数。
PV消费第一块时复位归约状态，消费最后一块时使用对应query的RoPE和输出地址。
压缩块→raw的顺序、BF16概率量化点、sink和每query的归约算术保持不变。
不添加GM metadata或AICPU任务。

## 整段计时

CANN9.2 beta2、NZ2、atomic0、deterministic0，正式第3层权重、合成输入/历史。
npugraph_ex dynamic=False，inplace/static开启，PTO不启用SuperKernel。
下表是实际逐次重放的纯设备span，单位μs；不是各进程中位数的min/max。

| 口径 | base min / max / mean | stream min / max / mean |
| --- | --- | --- |
| 128K/B16 同进程ABBA，每侧10次 | 586.50 / 631.75 / 608.45 | 574.50 / 604.50 / 584.33 |
| 128K/B16 服务入口ABBA，每侧18次 | 587.00 / 648.75 / 605.71 | 558.50 / 634.75 / 583.90 |
| 8K/B24 同进程ABBA，每侧10次 | 622.25 / 669.75 / 637.35 | 633.00 / 684.75 / 644.73 |

长档服务入口mean减少21.81μs（3.60%），max减少14.00μs；两种筛选方向一致。
短档代码未改，本窗口mean增加7.38μs（1.16%），保留该波动，不据此宣布短档提速或零回退。
两档同进程mean相对变化按8:2加权为−2.94%，保留候选。
这仍是PTO前后对照，尚未重取Native七档及整模型结果，领先Native20%的目标未达成。

## 多task联动

独立DFX采集，start/end相对该次泳道首个worker事件，单位μs。
DFX含采集开销，绝对跨度不与上表正式计时混用；worker耗时包括等待。

| 任务组 | base核内mean | stream核内mean | base start→end | stream start→end |
| --- | ---: | ---: | --- | --- |
| cmp_work_gather | 22.31 | 30.13 | 233.84→268.88 | 233.44→272.94 |
| Q投影 | 60.95 | 60.96 | 158.90→291.50 | 161.78→298.06 |
| Q反量化/RMS/RoPE | 46.01 | 42.86 | 297.78→347.78 | 304.56→352.92 |
| attention AIC | 157.81 | 144.88 | 352.94→513.16 | 358.86→505.86 |
| attention AIV | 161.74 | 149.03 | 353.18→517.46 | 359.16→510.50 |
| O-A | 26.94 | 25.18 | 523.84→609.20 | 516.86→596.12 |
| O-B | 10.85 | 9.26 | 580.90→645.36 | 573.10→630.28 |
| mHC post | 35.25 | 25.89 | 652.40→691.50 | 637.10→667.92 |

attention AIV核内min/max/mean从157.90/163.86/161.74变为146.80/150.94/149.03，
mean下降7.86%。但本次起跑晚5.98μs，整组结束只提前6.96μs。
未修改的O-B、mHC post也更快，因此不能把整段21.81μs全部归因于attention自身的12.71μs。
更不能仅凭前后泳道重叠就认定资源竞争的唯一因果；全任务数据保留在[result_h131072_b16.json](result_h131072_b16.json)。

泳道：

- [base](../results/hca_query_stream_20260930/h131072_b16/swimlane_base/dfx/merged_swimlane.json)
- [stream](../results/hca_query_stream_20260930/h131072_b16/swimlane_stream/dfx/merged_swimlane.json)

## 功能边界

- 128K/B16、8K/B24及服务入口ABBA：两侧完整输出/cache/state逐bit一致，eager/图重放和保护区通过。
- 16507/B6：36条query分给24组worker，覆盖1/2条query不均分配、压缩行128→129的块边界；
  同图有效请求6→5→1→6通过，包含worker前后query的work_count变化。
- 16384/B2：12条query，覆盖部分worker无query、只有压缩块+raw的短循环；2→1→2通过。
  边界用例cycles=1，不作为性能数据。
- B2首次失败源于旧测试把零个有效压缩metadata行判失败；旧base复现，输出和保护区本来就通过。
  修正为完整负slot缓冲精确比较，未消费cos/sin不作空张量比较；仅补跑此受影响用例。

## 复现和任务记录

在基线bb4c2831准备新的私有目录后运行`prepare.py`；不要在已有冻结目录上重复覆盖。
先显式CPU解析/编译/加载，再排队设备任务，不能以注册成功代替编译成功：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 python tests/pypto_test/hca_residual_reuse_20260930/compile.py stream \
  --operator-source /冻结stream目录 --output tests/pypto_test/hca_query_stream_20260930
task-submit --device auto --max-time 1200 --timeout 45 --run \
  'bash /绝对路径/hca_query_stream_20260930/run.sh 131072 16'
```

`run_edges.sh`是短档/边界用例入口；结果目录必须未被占用。
服务入口数据用`hca_residual_reuse_20260930/collect.py --experiment-root 本目录 --candidate-label stream`
汇总，再执行本目录`collect.py`。历史B2误报诊断也写入摘要，原始性能数据不会被覆写。

| 任务 | 用途 | 退出码 |
| --- | --- | ---: |
| task_20260930_023827_14556589273 | 128K/B16同进程筛选 | 0 |
| task_20260930_024008_148388132717 | 服务入口ABBA与两侧独立泳道 | 0 |
| task_20260930_024147_15204237872 | 短档、B6通过；B2空metadata误报 | 1 |
| task_20260930_024640_1641343341 | 旧base复现相同空metadata误报 | 1 |
| task_20260930_024822_16781511383 | 修正检查后的B2定向回归 | 0 |

新增脚本Ruff、shell语法、diff检查通过。attention文件原有60条I001/E501保持不变，
未做无关整文件格式化。完整format.sh ci仍依赖当前环境未安装的pre-commit。
