# 当前生产 HCA 与 Native 的核内差距分布

范围：128K/B16、正式第3层、CANN9.2、NZ2；生产算子0742f07c。
本次只分析已有数据，没有新增占卡测试。
结论：明显的AIC占用缺口主要在O-A、Q-B、Q-A、O-B，Attention本体已接近Native。
后续核内优先从O-A的数据复用、L1/L0流水和任务分解入手，不能继续只盯Attention。

## 数据与口径

- Native：`results/hca_native_sk0_20260929/h131072_b16/`，9月29日3次重放，卡1。
  已核实static_compile真实开启、SuperKernel关闭，dynamic=False、inplace开启。
- PTO：9月30日四个独立DFX窗口，都是生产0742f07c的冻结base：
  `hca_ob_fullwidth`、`hca_qgroups_sync`、`hca_online_softmax`、`hca_qgroups_workers`。
  完整路径、设备、冻结包身份记录在[comparison.json](comparison.json)。
- Native从`kernel_details.csv`按输入形状拆分Q-A/KV和Q-B/O-B。
  不能直接用report的top汇总：它按Name合并，会把不同矩阵乘混成同一个均值。
- PTO只取Worker View的`kernel-duration-us`，不包括`local_setup_us`。
  64个O-A worker是八组各八块，不是64个物理Cube同时运行。
- 由于块数不同，主表使用每次捕获的**总AIC核占用/24**，单位μs。
  Native为PMU单核均值×Block Num/24；PTO为所有worker核内耗时之和/24。
  min/max/mean分别跨上述Native三次重放、PTO四个捕获窗口统计。
  它们不是整层单次min/max/mean，也不是纯MAC时间或可直接回收的延迟。
- 两侧不是同卡同窗口，且PMU与DFX采集机制不同；此表用于定位明显差距，
  不能声称几μs差异已经严格归因。优化采用仍以同卡完整区间ABBA为准。

## AIC占用差距

| 功能 | Native min/max/mean | PTO min/max/mean | mean差值 | Native/PTO任务块数 |
| --- | --- | --- | ---: | --- |
| O-A：分组输出低秩投影 | 25.98 / 32.88 / 28.38 | 53.35 / 70.99 / 62.64 | +34.26 | 24 / 64 |
| Q-B：INT8展开到64个head | 35.02 / 36.76 / 35.96 | 48.98 / 52.90 / 51.41 | +15.45 | 22 / 16 |
| Q-A：4096→1024低秩投影 | 10.62 / 11.21 / 10.84 | 21.54 / 21.93 / 21.70 | +10.86 | 22 / 24 |
| O-B：INT8输出投影 | 17.96 / 18.46 / 18.18 | 24.14 / 30.46 / 27.82 | +9.64 | 16 / 64 |
| Attention QK/PV | 178.35 / 181.30 / 179.46 | 168.32 / 182.99 / 176.03 | −3.43 | 24 / 24 |

四个投影缺口约70.21μs的满24核占用当量中，O-A占约49%。
这不是整层能直接减掉70μs：任务存在重叠，核内时间含搬运、标量及等待，且算术组织不同。

### O-A：最大且跨窗口一致的缺口

当前PTO按8组，每组8个N128块，M物理128/有效96，K256共16段。
它与Native都使用同一NZ权重，不能归因于“PTO没开NZ”。
Native对应一次`TransposeBatchMatMul`，24个AIC承担全部8组。
PTO最后一个已有窗口单worker平均24.14μs，Native平均28.38μs；
仅看单worker会得出错误结论。64×24.14/24=64.38，明显高于Native28.38。

源码参考：`ops-nn/matmul/transpose_batch_mat_mul/op_kernel/transpose_batch_mat_mul.h`
的batch/block统一迭代，以及`pp_matmul_ein_sum_kernel.h`的L1预取和L0 ping-pong。
这是可借鉴实现，不证明安装版在该形状恰好命中了这个模板。
下一步需查看PTO生成码的实际复用、流水和全组任务分工；不能把64块本身当作唯一根因，
也不能忽略当前按组提前交给quant/O-B的重叠收益。

### Q-B与Q-A：Cube侧比反量化Vector更值得优先查

Q-B当前是四组、每组4个Cube；各worker约73–79μs，总占用仍高于Native22核实现。
Native的QuantMatmul还包含反量化，其AIV与Cube重叠；PTO的INT32写回后接独立反量化/RMS/RoPE。
功能对照必须同时看两种核，不能拿PTO整条Q链与Native纯Cube一项直接相比。
最近PTO窗口Q反量化/RMS/RoPE的48个worker均值35.04μs；
Native Q-B的Vector加后续Q RMS/RoPE折算48核约37.24μs，未显示同量级的Vector占用缺口。
Q-A本体则稳定约21.5–21.9μs满核当量，Native约10.8μs，应列入核内优化优先项。

### O-B：分组计算及后处理带来额外组织成本

PTO按8组1024 K维分别量化/矩阵乘，生成8份INT32 partials，末端反量化累加并融合mHC post。
Native是8192宽度动态量化、完整K矩阵乘，反量化在QuantMatmul内部完成，随后单独HcPost。
PTO每块约9–11μs比Native单核约27μs短，但64块总占用仍更大。
不应把PTO `hca_oproj_hc_post`直接与Native `HcPost`相比，因为前者还做8组反量化归约。
此前整宽O-B候选完整区间回退见LOG124；本表不推翻该实验，也不能原样重测期待收益。

## 另两类代价应与incore分开

1. **mHC pre＋norm的跨任务链**：最新PTO窗口从widen核内开始到mix/norm结束约96.84μs；
   Native对应HcPre＋RMS三次平均64.50μs。PTO拆成widen、linear、mix和旁路Sinkhorn，
   不能把这约32μs全归因核内。其AIV总占用折算约35.96μs，反而低于Native55.13μs；
   差距包含串联与任务间等待，需要下一轮联动调度分析。
2. **Attention前的准备**：PTO有raw gather、compressed gather、valid/sign等额外任务；
   最新窗口compressed gather核内mean23.67μs、组跨度33.92μs，raw gather组跨度24.50μs。
   Native在Attention里直接分页读。它们与Q等任务重叠，不能直接把跨度相加。
   第134节Cube直读虽删除暂存，却使完整区间变慢65.9μs，尚无可采用的消除方案。

Attention本体AIC两侧约176/179μs，PTO AIV约181μs、Native约182μs，当前属于相近水平。
继续优化Attention可帮助拉开优势，但不是现有最大落后项。
没有本次当前版本其他六档的核内对照，不外推所有batch/seqlen。

`python tests/pypto_test/hca_incore_gap_20260930/collect.py`只读取已有文件，重建原始样本及统计。
