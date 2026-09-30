# 按 Native 的 512 列归约组织长档 Attention（2026-09-30）

状态：`macro512_v2` 已完成 CPU 编译/load、单卡独立 FP64 诊断、完整层性能与独立泳道。
完整层和 Attention 核内均回退，不接入生产。生产仍为 0742f07c。

## 参考与改动

本地 ops-transformer 28f40354 的
`experimental/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.h:554`
固定 `sInnerSize_=512`，对应 cpp:1537–1544 设置 `s2BaseSize`。
`op_kernel/arch22/sparse_attn_sharedkv_swa_block_cube.h:59–60` 的 M/N 微块仍为 128，
PV 在 K 方向累加后写出。不能把微块 128 误认为整个 softmax 的列块为 128；
这里引用本地源码的实现策略，不据此声称已解析安装版二进制的实际 tiling。

控制是第126节未接入生产的 single-query online 候选，源码身份见 [source.json](source.json)。
其每块先更新累计 max 再转 BF16；本候选将四个 128 列压缩块合为一个 512 列 softmax/PV：

- QK 仍使用 128 列微块，四块写入同一 512 列 score 后通知 Vector。
- Vector 按 16 heads 处理 softmax，再发布 BF16 概率。
- Cube 将四个 K128 的 PV 累积在四片 64×128 L0C，最后才写出 64×512 FP32 结果。
- 保留跨 query 流水、sink、mask、负页保护和逆 RoPE；两槽、一步延迟。末 query 排空。
- QK/PV 使用独立 L1 KV 区域，PV 重新加载 KV，未保留原三槽常驻 KV。
- 短档入口不改；所有布局、权重、cache/state 接口和任务依赖不改。

128K 每个 query 的压缩历史为 1024 行：PV 中间结果从 8 个压缩块＋1 个 raw 块变为
2 个压缩块＋1 个 raw 块。B16×6 时，按实际 tile 字节数估算减少约 144 MiB 的 PV
GM 往返，但额外 KV 重载约 108 MiB；净逻辑传输减少约 36 MiB。
这些是满历史形状的逻辑字节数，不是测得的 HBM 流量或耗时收益。
同时减少 Vector 全宽累积和 FFTS 交接，但会推迟首轮概率发布、改变 Cube/Vector 重叠。
必须联合观察 gather、Q 反量化、Attention 和 O-A 的时序，不能单独按数据量判优。

## 编译与正确性边界

v1 在 `ConvertToSSA` 因 QK 分支重绑定外部 tile、又显式回传四个标量失败。
v2 使用 `raw_loaded`/`cmp_loaded` 局部结果，保留底层复用缓冲区，编译/load 通过。
生成码确认 raw 的零次后续 K 循环保留初值，compressed 的四片 L0C 在循环内累加、循环后写出；
这只是生成码检查，仍需真机验证跨 query 描述符、尾块、补位与重放。

[memory.json](memory.json) 记录生成码中的分配地址上界（32 字节对齐的 tile 末端）：

| 内存 | 字节 |
| --- | ---: |
| Mat/L1 | 344064 |
| Acc/L0C | 131072 |
| Left/L0A | 32768 |
| Right/L0B | 65536 |
| Vec/UB | 178112 |

512 列统一 max/BF16 以及 Cube PV 累加会改变舍入；不要求与 128 列版浮点输出逐 bit 相同，
但 cache/state/保护区必须保持精确。先用独立 FP64 密集参考检查 B5×6 的跨 query、
131072 正常历史和 16507 尾块/负页/NaN补位，自身图重放精确。
诊断 helper 的 `base` 是 single-query online，`online` 标签在本实验代表 **macro512_v2**，
以 [probe_sources.json](probe_sources.json) 为准。报告误差，不依据观测值反设容差；
独立诊断不等于整模型 token 验收。

## 实测结果

CANN9.2/NZ2/atomic0/det0、真实第3层、128K/B16，同进程 ABBA 每侧10次。
完整 HCA min/max/mean（μs）：控制 **538.500/594.500/559.350**，
候选 **592.750/620.250/605.725**。五个 ABBA 小组全部回退，mean 增加46.375。
两侧各自图重放和保护区通过，三类完整 cache/state 精确；层输出98884/1572864个元素不同，
max_abs=0.015625、RMSE=0.0006500071、非有限数0。零容差比较为FAIL，按算术差异如实记录。

独立 FP64 参考在131072历史的 max_abs 从0.001072411变0.001097228，
RMSE从0.0001300515变0.0001307870；16507边界的max_abs均0.002190986，
RMSE基本相同。两档自身图重放、保护区、空请求零输出通过；不是模型精度验收。

独立泳道的 Attention AIV mean **180.967→214.870 μs**，跨度183.30→218.08；
Cube mean177.137→209.645。尽管 gather跨度52.78→42.28、Attention启动317.68→303.30，
其完成仍从500.98推迟到521.38。不能将前置任务提前当成Attention核内收益。
当前没有PMU或核内阶段计时证明唯一原因；QK四个微块的同一L1缓冲串行复用及PV KV重载
是可检查的流水差异，不因逻辑字节减少就排除这些代价。
本版不扩短档/七档/16卡。原始样本、算术诊断和泳道摘要见[summary.json](summary.json)、
[incore.json](incore.json)，完整结果在`../results/hca_macro512_20260930/`。

## 复现

`prepare.py` 从第126节冻结控制复制完整包并生成 v1；`prepare_ssa.py` 在新副本生成 v2。
已有副本不覆盖。冻结源码的精确增量依次为 [macro512.patch](macro512.patch)、
[ssa_fix.patch](ssa_fix.patch)，应用需 `git apply --unidiff-zero`。
prepare 中模板排版可不同，冻结包和 patch 是本次编译/排队的实际来源。
CPU 复用 `hca_residual_reuse_20260930/compile.py`；独立诊断复用
`hca_online_softmax_20260930/probe.py` 的 `--experiment-root/--baseline/--candidate`。

真机只通过 task-submit 自动分配单卡，句柄见 [tasks.json](tasks.json)。
本实验三个任务（probe/pair/DFX）均exit0；没有将零容差输出差异说成逐bit通过。

## QK双缓冲增量：完整层改善，独立泳道排队

针对v2重复使用同一QK L1 tile，`prepare_pipeline.py`从冻结v2复制新包：
compressed微块加载改为循环内局部`pl.load`，四块循环使用`pl.pipeline(stage=2)`；
raw也使用局部加载，PV和Vector算术、两槽跨query流水均保持。
不是重新测原失败方案，也不是改变softmax精度策略。

CPU依赖图、编译/load通过。生成码compressed微块使用L1地址196608与327680的两个
128×512 BF16区域，循环步长2并有尾分支；不是仅写了stage属性而仍只有一个缓冲。
[memory_qkpipe.json](memory_qkpipe.json)记录Mat458752、Acc131072、Left/Right65536、Vec178112。
task_20260930_084830_254199630115已exit0。以macro512_v2为控制，完整输出/cache/state逐bit、
自身图重放和保护区通过，没有启用算术差异豁免。
128K/B16同进程ABBA每侧10次，min/max/mean（μs）：
v2为606.750/657.250/628.550，QK双缓冲为604.750/634.250/614.500；
P50从627.750到612.500，五个ABBA小组mean均改善，保留增量。
这没有证明整个512列方案优于128列或生产，也不能把不同窗口的差值相加。
独立泳道用新窗口qkpipe_control（同一v2源码）与macro512_qkpipe，任务仍pending；
待核对Attention核内与相邻task之后决定下一项，不先扩七档或模型。
