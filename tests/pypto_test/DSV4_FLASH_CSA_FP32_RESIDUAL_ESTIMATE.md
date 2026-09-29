# 层间FP32残差流的性能预估

2026-09-29。前半部分保留测试前估算；用户随后要求的两版实测已完成，
[最终实测与上游处理逻辑](results/csa_fp32_residual_upstream_20260929/REPORT.md)为当前结论。
生产模型数据类型未修改。

讨论范围是`[T,4,4096]`的mHC层间残差流改为FP32，权重、QKV、KV cache及GEMM主路径维持当前类型。
上游pypto-lib采用FP32残差流；当前接入版为对齐Native采用BF16输入/输出，HC内部主要以FP32计算。

## 可以省什么

- 入口可以直接借用FP32输入，省掉`x_hc32`临时副本及BF16→FP32转换。
- `hc_pre_linear`只消费输入，不依赖RMS结果；取消加宽副本后，有机会与RMS并行启动。
- HC_post可以省掉四路残差的加宽和最终输出BF16转换；attention输出本身的BF16舍入可继续保留。

RMS平方、归约和rsqrt仍然要做，HC门控、Sinkhorn、混合及HC_post的乘加也要保留。
目前HC_post已经按块只加载/转换四路残差各一次，并复用到四个输出，因此不存在可额外消除的16次重复转换。

## 已有时间尺度

来源是632dd00a七档的[四窗明细](results/csa_flat_gather_seven_20260929/TASKS.md)，
不是FP32候选A/B，也不是当前46cec3a0的新七档。

| 项目 | 已有时间 | 对估计的含义 |
| --- | ---: | --- |
| 七档hc_widen_rms首start至末end | 约12–19μs | 包含必须保留的RMS，不能全计为转换成本 |
| 128K/B16 hc_widen_rms每Worker核时均值 | 13.343μs | 12份并行Worker，不乘12推算延迟 |
| 128K/B16 hc_pre_linear首start | 25.050μs | 相对该窗起点，含依赖/调度；不等于可直接省25μs |
| 128K/B16 proj_b_act_hc_post每Worker核时均值 | 27.465μs | 还含O投影收尾和HC乘加，不能全计为cast |
| 128K/B16完整CSA正式事件均值 | 964.806μs | 只作量级分母，与独立DFX不相减 |

工程预估：净节省可能为每CSA约5–15μs，即128K/B16约0.5%–1.5%，只作为候选优先级判断。
这不是置信区间或已测收益；额外读写/调度可能抵消收益，零收益甚至退化都未排除。
也没有证据支持10%量级收益。

## 流量、数值与模型边界

B16/S6时T=96，一份残差BF16为3MiB，FP32为6MiB；B24/S6对应4.5MiB和9MiB。
外部残差读写字节数翻倍，但原路径已写出内部FP32副本，因此整体GM流量不能简单说翻倍。
只计上一次HC_post写出、下一次RMS读入和加宽副本写出，每元素现有2+2+4=8字节，
FP32方案4+4=8字节。它本身没有净字节收益；还需计入残差后续读取、复用和真实缓存命中。
可能的价值主要在转换指令、临时副本和关键依赖链，而非假定带宽节省。

直接保留FP32数值会取消现有层间BF16舍入，属于算术策略变化，需重新检查逐token与DSpark。
若为了保持原数值而先舍入BF16再存FP32，则仍保留转换，潜在收益更小。
模型每层attention和FFN各有HC边界，另有HCA路径；只改CSA边界再转回BF16无法获得端到端方案的完整收益。
当前没有这些边界的统一FP32模型profile，不把单CSA百分比直接当整模型forward百分比。

源码依据：性能版`decode_csa.py`的`hc_widen_rms`，共享`hc_pre.py`的`hc_pre_gates_from_rms`，
共享`hc_post.py`的`hc_post_block`，以及`models/deepseek_v4.py`的层forward。

## 用户要求实测后的上游对照

对照本地pypto-lib2164563的`models/deepseek_v4_flash_dspark/`：

| 路径 | 上游实现 | 本次私有试验 |
| --- | --- | --- |
| embedding | BF16 embedding读取后一次加宽，复制到四路FP32 HC残差 | 以同值BF16输入一次加宽模拟，放在计时外 |
| HC_pre RMS/门控 | FP32残差直接读取；RMS和linear独立生产，RMS允许early | 第一版只去副本保留原标志；第二版按上游组织两生产者 |
| HC_pre混合/归一化 | FP32四路混合后舍入BF16；RMSNorm统计包含这个BF16舍入，再输出BF16 | 保留同样边界 |
| attention | 输入/输出BF16，投影及Indexer等继续混合量化计算 | 保持现有BF16/INT8主路径及KV类型 |
| HC_post | BF16 attention输出加宽；与FP32残差乘加，结果直接FP32写出 | 相同类型/舍入边界，保留本地O投影融合收尾与残差四路复用 |
| FFN/MoE | HC_pre收FP32、混合结果BF16；expert输出BF16；HC_post再输出FP32 | 本轮不改FFN，仅用两次CSA调用检查FP32低位消费 |
| 层间缓冲 | x_ping、x_pong、x_attn_active、x_moe_next均为FP32；active/capacity间仍可能有FP32复制 | 尚未贯通Native模型，单CSA时间不能冒称模型结果 |
| 最后HC head | 接收FP32残差，缩并/归一化输出BF16送后续head | 本轮不改 |

关键源码位置：`lookup_embedding.py:56`、`hc_pre.py:51`/`:343`、`hc_post.py:37`、
`moe.py:537`/`:579`、`decode_fwd.py:391`、`hc_head.py:62`。
不能把“FP32贯穿残差流”理解为所有激活、GEMM、KV或通信数据全部升为FP32。

本轮按用户新增要求实测；上文5–15μs仍只是测试前预估，最终以实测报告为准：
[先只改残差流](results/csa_fp32_residual_20260929/README.md)、
[再参考上游HC_pre组织](results/csa_fp32_residual_upstream_20260929/source.json)。
