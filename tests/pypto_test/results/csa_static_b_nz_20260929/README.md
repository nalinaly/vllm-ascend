# 静态 matmul B 逐张 NZ 对照

基线为生产369ad2c1；当前生产尚未接入本轮任何NZ候选。各行是各自同卡A/B，不能跨行挑选最快基线。
ND/NZ仅表示本行改动的权重；两侧四张Q/O仍使用mode2 NZ。保持原K顺序、任务分工和依赖。

先看完整CSA绝对耗时与P95，128K/B16和8K/B24按8:2；性能后检查八类完整张量及索引、保护区。
仅核内更快不能代替完整区间收益；精度失败不接入。单位μs，百分比为耗时变化，负值更快。

| 权重 | 128K/B16 基线→NZ | 8K/B24 基线→NZ | CSA 8:2 | 精度 | 取舍 |
| --- | ---: | ---: | ---: | --- | --- |
| [主KV](../csa_kv_native_nz_20260929/RESULTS.md) | 969.455→970.119 | 919.268→913.069 | -0.080% | PASS | 暂缓：区间收益不明确 |
| [Indexer Q](../csa_idx_q_nz_20260929/RESULTS.md) | 957.030→966.292 | 919.489→935.123 | +1.114% | PASS | 不接入 |
| [Indexer head投影](../csa_idx_weights_nz_20260929/RESULTS.md) | 967.587→971.545 | 919.207→947.245 | +0.937% | PASS | 不接入 |
| [主Compressor wkv](../csa_cmp_wkv_nz_20260929/RESULTS.md) | 961.629→982.969 | 911.449→911.962 | +1.787% | PASS | 不接入 |
| [主Compressor wgate](../csa_cmp_wgate_nz_20260929/RESULTS.md) | 966.279→966.485 | 939.797→906.600 | -0.689% | PASS | 复核未保住收益，暂缓 |
| [Indexer Compressor wkv](../csa_inner_wkv_nz_20260929/RESULTS.md) | 975.051→967.129 | 949.969→926.813 | -1.137% | PASS | 复核未保住收益，暂缓 |
| [Indexer Compressor wgate](../csa_inner_wgate_nz_20260929/RESULTS.md) | 956.620→974.555 | 923.882→936.842 | +1.780% | PASS | 不接入 |
| [共享Hadamard](../csa_hadamard_nz_20260929/RESULTS.md) | 971.293→971.507 | 937.610→926.751 | -0.214% | FAIL | 精度失败，禁止接入 |
| [主wgate＋inner wkv](../csa_compressor_pair_nz_20260929/RESULTS.md) | 962.850→967.722 | 918.730→921.210 | +0.459% | PASS | 组合回退，不接入 |

Hadamard的−0.214%仅为失败输出下的原始计时，不能当成有效收益。长档idx_topk有49149/49152个元素不同，
x_out最大绝对差0.2421875；短档Top-K有73669/73728个元素不同，x_out最大差0.20703125。
此候选存在功能问题，目前未完成最小复现，不归因为可接受量化差异，也不未经验证归因工具链。

主wgate独立DFX核时：长18.654→18.079（−3.080%），短30.782→28.362（−7.861%）。
DFX与正式计时为独立任务，不能相减推算调度开销。它与inner wkv组合后CSA加权反增0.459%，不能相加单项收益。
inner wkv反序复核完整CSA长959.464→962.505、短917.629→920.767，8:2 +0.322%；P95均上升，
八类完整状态仍精确一致。首轮−1.137%没有重现，因此暂缓采用，不补边界/整模型。
主wgate独立复核也未重现首轮收益：长970.169→970.859、短935.773→938.735，8:2 +0.120%，
短P95 957.840→969.660；八类完整状态精确一致。两轮均保留，不择优发布第一轮。
**本轮收口：没有确认稳定缩短完整CSA的新NZ候选，生产保持369ad2c1算子。**
明确回退或精度失败的候选不接入；仅首轮获益的候选暂缓，不宣称NZ本身永远无效。

所有已通过状态检查均为八类完整输出/缓存/状态零容差，包括x_out、idx_topk、SWA、压缩cache、
两套Compressor state、Indexer INT8 cache及scale；还检查自身图重放、metadata/保护区和Top-K结构。
这些是本轮单层同初态检查，**不是新的整模型token/DSpark或EP16验收**。

Native已有NZ的主KV/Indexer Q/head候选直接借用原存储；Compressor Native保留ND，PTO初始化另备NZ。
四张Compressor全部转换需额外20MiB/CSA层；单独inner wkv为2MiB，主wgate为8MiB。
共享Hadamard为32KiB。初始化副本不计入每步CSA，但驻留成本必须记录；Native流程未改。

环境：CANN9.2 beta2、同工具链与真实第二CSA层权重；mode2/atomic0/det0，5预热20正式事件。
PTO现有图方式、inplace_pass=True，未重测Native。CPU编译通过不替代设备性能/精度。
[inner wkv反序复核](../csa_inner_wkv_nz_confirm_20260929/RESULTS.md)、
[主wgate反序复核](../csa_cmp_wgate_nz_confirm_20260929/RESULTS.md)。
