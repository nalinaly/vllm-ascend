# Q RoPE整块Gather：保留核内收益，继续解决组间等待

基线a7a9f314，CANN9.2、NZ2、atomic0、det0，正式第3层权重。
修改本仓性能版公共Q helper，供HCA调用；没有修改CSA工作树或公共PyPTO/Simpler依赖。

## 改动与源码依据

原`pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)`将8×64块编译为8次逐行
Gather、临时行搬运和标量循环。现在给每行index加`row * 64`，使用flat Gather一次处理整个块。
每个head组只准备一次索引，head的stage2流水、48个workers、任务依赖、RMS/量化/舍入顺序不变。
完整行之外的尾块路径不改，索引仍来自Native的interleaved布局准备结果。

本地ops-transformer 28f40354的`posembedding/interleave_rope/op_kernel/interleave_rope_b11d.h:127`
按`factor * hiddenDim`批量做向量重排，可参考其减少逐行标量控制的思路；
未声称安装Native在HCA形状选择该模板，也没有复制其不同的RoPE算术。
pypto-lib 2164563的完整块仍使用axis Gather，本改动有意改用当前PyPTO已支持的flat Gather，
差异只在局部索引表达与lowering，输出顺序及数值规则一致。没有为此修改工具链。

CPU显式解析两个根、设备代码编译及load通过；生成代码中`gather_inp_row`循环归零，
完整块为8×64 TGATHER。生产源码与已测私有包AST一致。

## 代表档性能

同进程同卡ABBA各10次真实设备span；不同档各有自己的同卡base。
单位μs，min / max / mean：

| 档位 | base | flat Gather |
| --- | --- | --- |
| 128K/B16 | 587.50 / 616.50 / 596.75 | 578.75 / 624.75 / 597.23 |
| 8K/B24 | 617.00 / 671.25 / 637.78 | 621.00 / 638.25 / 631.03 |

长档mean+0.48μs，max+8.25μs；短档mean−6.75μs，max−33.00μs。
两档相对mean按8:2加权为−0.148%，效应很小，不能称稳定整段提速。
按用户要求保留明确核内收益，继续检查生产者/消费者和并发任务，不用单个task变快代替整段验收。

## 独立DFX的联动

核内包含等待；start/end相对各自泳道首个worker，不与上述计时直接相减。

| 任务 | base核内mean / 起止 | flat核内mean / 起止 |
| --- | --- | --- |
| Q投影 | 62.04 / 159.80–290.58 | 61.82 / 163.06–238.46 |
| Q反量化/RMS/RoPE | 44.55 / 296.50–345.74 | 35.48 / 258.54–323.28 |
| 压缩gather | 23.23 / 228.62–274.00 | 18.26 / 236.18–273.80 |
| Attention AIV | 161.49 / 350.28–515.26 | 176.95 / 326.16–507.44 |
| O-A | 25.56 / 520.76–605.16 | 23.13 / 511.04–586.76 |
| O-B | 11.47 / 575.68–640.06 | 10.15 / 560.16–620.78 |
| mHC post | 32.86 / 646.00–683.92 | 29.18 / 627.14–660.66 |

修改目标Q反量化核内min/max/mean从40.78/48.60/44.55变为34.52/36.52/35.48，
mean下降约20.4%、max下降12.08μs。与此同时组跨度49.24→64.74变宽，
Attention更早启动但核内更慢，未修改Q投影的启动分布也明显变化。
flat泳道Q反量化与压缩gather有重叠，base窗口没有；这是下一轮调度定位依据，
不是已证明所有时间变化都由竞争造成。完整任务数据见[incore.json](incore.json)。

## 精度与边界

128K/B16、8K/B24、16507/B3的四类完整输出/cache/state跨版本逐bit一致；
各自eager/图重放和保护区通过。B3的18行覆盖完整8行块和2行尾；
同址图有效请求3→2→1→3通过，覆盖seq_lens=0、负slot及补位cache/state保护。
B3每侧只有2次筛选重放，仅作功能证据，不用于性能结论。
没有新增整机token/DSpark或七档验收；本轮Native对照仍是改动前a7a9f314。

## 复现与证据

[摘要](summary.json)、[冻结来源](source.json)、[CPU编译](compile_flat.json)。
原始profile与泳道在`../results/hca_q_flat_gather_20260930/`，两侧泳道分别是
`h131072_b16/swimlane_base/dfx/merged_swimlane.json`与`swimlane_flat/dfx/merged_swimlane.json`。
`prepare.py`从指定冻结基线生成新包，CPU编译复用`hca_residual_reuse_20260930/compile.py`；
设备筛选复用`hca_pair_screen_20260930/run.sh`，本目录`run_dfx.sh`、`run_edges.sh`均必须经队列。
`collect.py`重建摘要；不覆写已有结果目录。

| task | 用途 | 退出码 |
| --- | --- | ---: |
| task_20260930_032523_227608629660 | 长档同进程筛选 | 0 |
| task_20260930_032730_233911522570 | 两侧独立DFX | 0 |
| task_20260930_033103_241944627715 | 短档与混合尾块/补位图 | 0 |

定向脚本Ruff、shell与diff检查通过；全仓`format.sh ci`因缺pre-commit未完成。
