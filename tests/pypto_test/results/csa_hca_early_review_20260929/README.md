# HCA 补齐 early 标志对 CSA 的借鉴

2026-09-29。结论：值得借鉴的是检查整条依赖链的生产者标志；CSA 有明确的未覆盖链路，
但不能直接套用 HCA 的 16 个源码位置或收益。此次只读代码并复用已有八个 DFX 窗口，
没有修改生产算子、工具链或正在运行的私有副本，没有新增设备采样。

## HCA 的实际修改与证据

HCA 分支提交 `c5f4252a` 包含两项独立改动：

- `allow_early_resolve=True` 补 16 处：Compressor 四处、入口两处、三条 Sparse 路径各两处、
  weight warm 两处，以及复用的 KV projection/RMS 两处。三条 Sparse 路径互斥，16 是源码位置数。
- 共享 `q_projection.py` 的 `QPROJ_WORKERS` 从 24 改成 20。

HCA [日志 §70](../../../../../vllm-ascend-dsv4-hca-pto-0251rc1/tests/pypto_test/DSV4_FLASH_HCA_OPTIMIZATION_LOG.md)
记载 nz2、128K/B16、同卡 ABBA 六样本下，early 单项约 −20 μs；worker 单项满样本约 −16.5 μs，
组合约 −34.75 μs。七档 PTO/Native 的 8:2 加权比值 1.157→1.117 是组合结果。
不能把组合收益全归因 early，也不能把 HCA 时间当成 CSA 的预期加速。
六样本有少量范围重叠，按原始数组理解，不采用提交说明中“完全分离”的过强措辞。

## API 与实际依赖

标志在生产者上，允许消费者在生产者完成前预先派发；消费者真正开始计算仍受依赖完成保护。
需要核对消费者的全部有效前置，包括元数据和显式门控，不能只给消费者自己加标志。
普通 ready 任务和 early 任务的优先级、整组准入及核占用仍由 runtime 决定，开启不保证提速。

当前 Simpler A3 的实现依据：
`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/orchestrator.cpp` 的
`all_claimed_fanin_allow_early_resolve`；同期 `docs/RUNTIME_LOGIC.md` 的 NORMAL/EARLY 与
sync_start 调度通道说明。当前 runtime 已有 early sync_start 整组路径，不能仅凭旧教程中
“sync_start 不能逐 block 预置”就认定 `kv_proj_matmul` 上的标志一定无效。

## CSA 的对应关系

| 位置 | 当前情况 | 处理建议 |
| --- | --- | --- |
| 原始 KV projection→RMS/RoPE→cache write | 三个生产者均未开启；前两处与 HCA 修改完全同源 | 可单独做 KV 链对照，保留原 sync_start 和数学 |
| Indexer Compressor | projection、pool、boundary init、RMS/RoPE、Hadamard 五处未开启；后续 key/scale write 已开启 | 优先按完整链路补齐，不能只改 write |
| Attention Compressor | projection 未开启，pool/cache write 已开启；其 projection 还门控 Indexer Hadamard | 与 Indexer 链的前置一起核对 |
| `csa_row_offsets` | 未开启；是 RMS、两类 cache write、scale commit 的实际前置 | 必须纳入链路对照，属于 HCA 没有直接对应位置的 CSA 额外元数据工作 |
| `hc_widen_rms` | 未开启，HC linear 未实际预派发 | 旧 §212 的独立 widen 在 8K/B40 无明确收益，不能称当前已验证收益；当前融合 RMS 后如单独复查，须明确新依据 |
| Q_B matmul | CSA 与 HCA 共用 helper，CSA 仍是 24 workers，ND/NZ 两个分支均未开 early | HCA 的 20 workers 思路可另做一项；不要与补标志混测，两个因素会相互影响 |
| 长档 Score→Top-K merge | Score 刻意关闭 early，配合整组准入控制尾部；短档开启 | 保留现有长短策略，不用“补齐”覆盖历史长尾措施 |
| Sparse plan、inverse RoPE、QK/PV 和 O 投影/HC 收尾 | 主要生产者已开启，Sparse 已实际预派发 | 不是本轮补齐重点 |
| state commit、atomic seed | state commit 主要是状态输出；当前 atomic0 不执行 seed | 不按缺标志数量凑修改；先确认存在本次调用中的关键消费者 |

## 已有泳道中的证据

来源是 `7b296153` 对应冻结候选（单行 HC 收尾融合已采用）的两档各四个窗口，
与当前在跑的 Sparse 计划候选独立。官方 raw join 校验后取全部非分配直接前置的最晚 FIN，
再到消费者首个 kernel start；有未计时 dummy 的任务不计算该间隔。

| 消费者 | 128K/B16 平均 FIN→start（μs） | 8K/B24 平均 FIN→start（μs） | 未开启的前置 |
| --- | ---: | ---: | --- |
| Indexer pool | 6.655 | 5.825 | Indexer projection |
| Indexer boundary init | 6.355 | 5.190 | Indexer pool |
| Indexer RMS/RoPE | 9.360 | 13.395 | pool、boundary init、row offsets |
| Indexer Hadamard | 7.005 | 4.950 | RMS/RoPE、Attention Compressor projection |
| Indexer key/cache write | 6.905 | 21.905 | Hadamard、row offsets |
| 原始 KV RMS/RoPE | 4.455 | 4.845 | KV projection |
| 原始 cache writeback | 5.380 | 5.065 | KV RMS/RoPE |
| Q_B dequant | 6.440 | 7.805 | Q_B matmul |
| HC linear | 5.715 | 4.910 | HC widen/RMS |

表中这些消费者两档各四窗均未实际提前派发，存在明确的排查依据。
但这些间隔包含核资源等待，且多条链互相重叠，**不能相加当成可回收的 CSA 时间**。
个别短档窗口更慢，例如 Indexer RMS/RoPE 达 36.26 μs，key/cache write 达 29.30 μs，
必须保留分布，不能只用均值描述稳定性。

对照：Sparse QK/PV 两档 FIN→start 仅 0.810/0.640 μs，已 full/partial early；
短档 Top-K merge 已 full early，间隔 0.610 μs。长档 merge 的 5.620 μs 对应刻意关闭的 Score early，
不能在没有长尾对照的情况下删除限制。

## 下一项可独立验证的候选

优先试完整的 Indexer cache 生产链，只加七处标志：
`csa_row_offsets`、Indexer 的 projection/pool/boundary init/RMS/Hadamard 五处，
以及作为 Hadamard 门控前置的 Attention Compressor projection。
保持依赖边、任务数量、worker 数、cache 布局、数值、Score sync/early 策略不变。
这七处是基于真实前置筛出来的候选，不是已证实能节省表中全部间隔。

若测试，使用独立冻结整包，先同卡 128K/B16 和 8K/B24，检查完整 CSA、P95/max、
Score/Sparse 启动是否延后，以及现有八类状态；按用户 8:2 口径决定保留。
KV 链和 Q_B worker 数分别对照，不与这项或当前 Sparse 计划候选叠加。
本次尚未进行这些设备对照，不能声称 CSA 已获得 HCA 的约 20 μs 收益。

[只读分析入口](analyze.py)、[八窗逐任务前置/标志/时戳与原始路径](existing_windows.json)。

## 后续结论

本页上文是设备对照前的只读审查。之后七处Indexer链标志已在CSA验证并采用（f4861832）：
长B16/短B24完整CSA−0.728%/−1.872%，8:2−0.957%；长档P95+1.980μs单列。
[独立A/B及边界](../csa_indexer_early_chain_20260929/README.md)、
[包含后续已采用策略的七档实测](../csa_early_chain_seven_20260929/RESULTS.md)。
Q_B 24→20为[另一个独立候选](../csa_qb_workers20_20260929/README.md)，尚无设备收益结论。
