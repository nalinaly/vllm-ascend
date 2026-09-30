# 性能优化迁移到精度版（2026-09-30）

基线为 `0327f3b2` 的完整 `ops/pypto` 副本。本轮只改 CSA 的 PTO 实现；Native、cache 分配布局、HCA 及模型执行路径未改。

## 迁移范围与算术边界

| 部分 | 迁移内容 | 保留的精度规则 |
| --- | --- | --- |
| HC_pre | BF16 加宽时同时计算 RMS，去掉独立 RMS 任务；补齐早释放 | 512 列平方和次序、高精度 rsqrt、尾行清零 |
| QR/KV | 非 atomic 路径不再清零；QR/KV 增加独立 M 块并行；QR 权重 BYPASS；KV 早释放 | QR N32、Native K256 分组/遍历次序；KV K256；RMS 与 RoPE 的 BF16 边界；atomic 默认值不变 |
| Indexer | 复用性能版分页 GM 视图、FIXPIPE 缩放、Score 双缓冲、query 分组、长短档及 Top-K 调度；长档 sync_start；短于 8K 使用原精度算法的直接分页读取 | Q/RoPE、Hadamard、权重投影仍用精度实现；系数明确先 FP32 相乘再转 FP16；QK 的 1/1024、FP16 及第二次 Cube 规约保留 |
| Indexer compressor | pool 补早释放 | 原投影、pool 与量化算术不变 |
| Sparse attention | 在 QK/PV 尾部完成归一化、逆 RoPE、分组打包，去掉完整 FP32 输出落地及 merge_norm 任务 | 512 候选累计最大值、概率 BF16 round、PV 累加及逆 RoPE 前 BF16 舍入不变；L 从既有传递区读取 |
| O projection | 共用两版数值中性的 ND/NZ MatMul helper、形状分档；WO-B 反量化融合 HC_post；T96 48 任务 + sync_start | WO-A 保留 K256，不开启改变 L0 分块的实验流水；八组 BF16 后共用一套 token scale；INT32 先求和，再 channel scale/token scale，BF16 后进入 HC_post |

没有照搬性能版的 QR N128/顺序 K、Compressor K512/在线 pooling、Sparse K128/块内最大值、WO-B 每组量化尺度。
这些是算术策略差异。调度和分块也按精度版的 N/K 约束调整，并非把性能版源码直接覆盖过来。
Indexer 的精度系数由内部 `pl.constexpr` 参数选择，性能入口显式传 `False`，精度入口传 `True`，没有新增产品环境变量。

## 计时与精度看护

- 正式七档：128K B4/B8/B16/B24；8K B16/B24/B32。
- 两侧显式 `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0`，避免精度默认 1、性能默认 0 混入对照。生产默认值未改；默认 atomic=1 编译分支另有通过记录。
- TMR、CANN 9.2.0-beta.2、NZ mode=2、deterministic level=0、同一第 2 层正式权重和 seed=1024。
- 单次完整 CSA 区间，5 次预热、20 次事件计时，报告 min/mean/max 和原始样本/P95；同卡配对，初态 D2D 恢复不计时。
- 迁移看护比较旧/新精度版：128K/B16、8K/B16、128K/B24（覆盖不同 O 投影和 M 并行分支）、H257/B1 倒序页表（尾行与不足一页候选）。保存允许写入区域原始字节并检查其余区域，无 hash。
- 不做 16 卡，不将合成历史单层证据称为整模型 token/DSpark 或 Native 精度验收。

## 复现

从仓库根目录执行，NPU 一律走队列：

```bash
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 2400 \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/precision_port_20260930/group.sh a'
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 2400 \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/precision_port_20260930/group.sh b'
source ../env-dsv4-0251rc1.sh
python tests/pypto_test/precision_port_20260930/collect.py
```

原始目录：`../results/precision_port_20260930/`。`baseline_ops/`、`candidate_v11/`、`final_ops_v2/` 为冻结包；排队期间不编辑。
只读结果见 [RESULTS.md](RESULTS.md)，完整样本和保护区检查见 `evidence.json`；迁移前后逐元素证据见 `migration_guards.json`。
各过程原始文件通过结果 JSON 内的路径关联；编译失败的候选仅留在 ignored 原始目录，不参与性能统计。
本轮编译适配均在算子侧处理：显式 SPMD block index 供 NZ 边界证明、TaskId 数组跨分支、标量读取单元素 scale，以及显式绑定 constexpr。
没有修改 PyPTO、Simpler、PTOAS 或 PTO-ISA。

两版差异记录中的 `state.*` 是写入区原始字节比较；不同字节数不代表浮点误差容差判定。
