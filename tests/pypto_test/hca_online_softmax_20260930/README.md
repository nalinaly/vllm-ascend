# 长档累计最大值与BF16概率转换（2026-09-30）

生产算子仍为0742f07c；本实验未接入生产。CANN9.2/NZ2/atomic0/det0、正式第3层权重、BF16残差。
PTO以npugraph_ex、dynamic=False、inplace/static开启、SK关闭计完整mHC pre/norm/HCA/O/post。

## 实现与上游差异

当前生产和pypto-lib2164563的`models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py:593–636`
先按块内max计算概率并转BF16，PV之后用alpha/beta分别缩放旧累积和新PV，再相加。

最新本地ops-transformer28f40354的
`experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_swa_block_vector.h`
在442–467行给SoftmaxFlashV2传前一块的max/sum，537–543行在此之后转BF16；
801–803行只缩放旧PV累积再相加。该SWA模板同时用于CFA；入口的非SCFA分支与
swa_kernel.h的CFA_TEMPLATE参数可核对。Native vLLM入口是
`DeviceOperator.get_dsa_sparse_attn_op()`返回的`npu_sparse_attn_sharedkv`，compress_ratio128
不传稀疏索引。这里只借鉴明确的归约策略，不声称安装二进制的全部tiling/块序与候选一致。

候选保持当前压缩块先于raw、K128、24组MIX、三槽及跨query流水：

- softmax生产端按query独立维护累计max，概率先按它缩放再转BF16。
- PV消费者仍延迟两个tick，从UB环读取对应块的累计max与块sum；仅缩放旧累积，新PV直接相加。
- query切换时只重置生产端max，消费者继续完成前一个query；描述符及所有依赖保持。
- 保留sink、负页/尾块mask、cache布局、inverse RoPE；短档入口未改。

这是算术策略变化，不是逐bit优化。CPU v1把生产端和消费端循环值共用初始tile，编译器正确拒绝
同片上buffer承担两个独立状态；v2对每query独立加载sink建立生产端初值，解析/编译/load通过。
未修改PyPTO/Simpler/PTOAS/PTO-ISA，失败日志本地保存为`compile_online_v1.log`。

## 完整HCA的同进程对照

每轮5次预热，ABBA五组、每侧10个实际设备span。单位μs，min/max/mean。
每行只与自己的控制比较；第二行反转身份，原报告base=online、candidate=生产版。

| 档位/顺序 | 生产版 | online候选 |
| --- | --- | --- |
| 128K/B16，生产先测 | 565.000/609.500/579.175 | 551.750/575.250/561.375 |
| 128K/B16，online先测 | 576.250/596.250/584.725 | 566.750/617.750/586.600 |
| 8K/B24，生产先测 | 627.000/683.000/640.800 | 615.000/646.000/629.300 |

首轮长档mean少17.800μs（−3.073%），五个ABBA小组都改善；反转后mean多1.875μs
（+0.321%），P50也多1.250。首轮base最大值609.5和反转后online最大值617.75都在
各自profile第一个重放；不能将前一轮max下降34.25称为稳定削峰。
不事后删首项、不把两卡绝对值混成一个更好看的均值，也不将首轮3.073%写成已验证整体收益。

短档源码路径未改且完整输出逐bit，但仍观测到−11.500μs的mean差，主要集中在前两小组；
它提示运行窗口/加载顺序存在影响，不能归因于这次长档核内优化。后续有更大结构性收益再测，
不继续用小样本追逐此处噪声。生产保持不变，候选用于后续核内与流水组合。

## 独立泳道与联动

另一次同卡DFX，非上述计时窗口：

| 项目 | 生产版 | online |
| --- | --- | --- |
| Attention AIC核内min/max/mean | 181.14/184.76/182.986 | 176.10/182.40/179.711 |
| Attention AIV核内min/max/mean | 185.14/188.82/187.273 | 180.08/186.58/183.660 |
| Attention AIV跨度 | 189.22 | 186.82 |
| 最后Q反量化完成 | 301.58 | 322.56 |
| 压缩gather跨度 | 46.22 | 43.60 |
| Attention AIV启动/结束 | 304.60/493.82 | 324.14/510.96 |
| O-A所有group累计跨度 | 79.34 | 70.10 |

保留约3.61μs（1.93%）AIV核时改善线索，但不能把17.80μs的无DFX整段变化全部归因于它。
本窗口Q反量化有两组跨度从约41–45变61μs，Attention因此晚起跑；不能只看softmax改动。
O-A包含8组任务，64个worker是累计值，不是一个64核组。

泳道：`../results/hca_online_softmax_20260930/dfx/{base,online}/dfx/merged_swimlane.json`。

## 数值与功能边界

真实层长档输出对生产版：1572864元素中114248不同，max_abs=0.015625，RMSE=0.0007110773，
非有限数0；三类完整cache/state逐bit。各自eager/graph精确、全部保护区通过。
保留输出零容差FAIL及待独立精度/模型token验收标记；不把这个差异自动判成正确性bug或验收通过。
短档四类完整状态跨版逐bit。

`probe.py`使用独立CPU FP64密集softmax+sink及inverse RoPE参考，不复制候选块归约算法。
B5×6=30query让部分worker连续处理两个query；131072正常分页，16507另含压缩尾块、负页、
补位请求、未初始化行NaN；输出padding填17作保护区。单独Attention自身图重放逐bit，
输出保护区精确，补位输出为0，所有有效输出有限。

| 合成历史 | 生产版max_abs/RMSE | online max_abs/RMSE |
| --- | --- | --- |
| 131072 | 0.001118948 / 0.000126769 | 0.001072411 / 0.000130052 |
| 16507，含边界 | 0.002377886 / 0.000206907 | 0.002190986 / 0.000209545 |

候选最大误差略小、RMSE略大，不能笼统说精度更好。没有根据测得误差反推容差再宣布通过。
这两组独立参考只补实现诊断，不等于真实权重全模型token/DSpark验收。当前不为收益未稳定候选
扩大到七档/16卡模型测试。

## 复现与证据

[source.json](source.json)、[online.patch](online.patch)、[summary.json](summary.json)、
[incore.json](incore.json)、[probe_result.json](probe_result.json)、[tasks.json](tasks.json)。
私有包及probe副本只读；`prepare.py`仅在新路径创建，不可覆写。
CPU编译复用`hca_residual_reuse_20260930/compile.py`，probe用`probe.py --compile-only`。
所有NPU通过`task-submit --device auto --max-time 900`自动排队；pair入口是
`hca_pair_screen_20260930/run.sh`，长档加显式`--diagnostic-output-differences`，短档严格逐bit。
DFX复用`hca_mix_schedule_20260930/run_dfx.sh hca_online_softmax_20260930 base online`，
独立参考执行`run_probe.sh`。无需新增全套测试入口。
定向Ruff、shell语法、diff检查通过；全仓`format.sh ci`仍因缺少pre-commit未通过。
