# CSA 精度版封存（2026-10-01）

按用户决定，CSA 性能版成为默认且唯一持续维护的 PTO CSA 实现。精度版停止接入、优化迁移和新增验收；历史源码和测试结论保留。

## 封存内容与恢复

精度版源码已展开并迁到 [deepseek_v4_flash_dspark/](deepseek_v4_flash_dspark/)，可以直接浏览和检索。
来源为提交 `23e10b1289bebc400053ec9d07ea6731bd6f376b` 中完整的
`vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/`（22 个 Python 文件），内容保持原样。
原压缩包已移除，避免维护两份相同封存内容；压缩包历史仍可通过提交 `e05bc83a` 查看。

| 精度版部件 | 封存源码 |
| --- | --- |
| CSA 根入口 | [decode_csa.py](deepseek_v4_flash_dspark/decode_csa.py) |
| QKV 投影及 RoPE | [qkv_proj_rope.py](deepseek_v4_flash_dspark/qkv_proj_rope.py) |
| 主 Compressor | [decode_compressor_ratio4.py](deepseek_v4_flash_dspark/decode_compressor_ratio4.py) |
| Indexer Compressor | [decode_indexer_compressor.py](deepseek_v4_flash_dspark/decode_indexer_compressor.py) |
| Indexer | [decode_indexer.py](deepseek_v4_flash_dspark/decode_indexer.py) |
| Sparse Attention | [decode_sparse_attn_csa.py](deepseek_v4_flash_dspark/decode_sparse_attn_csa.py) |
| O 投影 | [decode_o_proj.py](deepseek_v4_flash_dspark/decode_o_proj.py) |

当时的公共辅助模块和适配也保存在同一目录。该目录不属于生产包，不由运行入口导入。
这些是历史源码，保留原导入关系；精度版还依赖同提交的性能包辅助函数、模型接入和测试脚本，
不能直接从新位置导入执行，或单独复制此目录到当前源码作为完整复现环境。
需要复现历史时，在仓库内使用独立目录恢复整份对应提交：

```bash
git worktree add --detach ../vllm-ascend-csa-precision-archive 23e10b1289bebc400053ec9d07ea6731bd6f376b
```

按原报告固定环境、权重、输入 bank 和运行配置。原七组实验的冻结源码
`tests/pypto_test/results/low_acceptance_20261001/source_v9_sync_load` 保持不动；
版本与任务见[原实验源码记录](../../low_acceptance_20261001/source.json)。
历史精度脚本只能在对应完整历史源码环境中使用，不再纳入当前测试矩阵。

## 当前运行规则

| 配置 | 行为 |
| --- | --- |
| 未设置 `PTO_CSA_VARIANT` | 默认 `deepseek_v4_flash_csa` |
| `performance` / `perf` | 同一性能实现 |
| `precision` / `prec` / `pkg:deepseek_v4_flash_dspark` | 明确报错，提示封存位置；不会静默替换算术 |
| `pkg:<name>` | 仅保留性能版私有实验副本工作流，避免排队任务读取正在编辑的 JIT 源码 |
| 未设置 `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD` | 默认 0；显式 0/1 能力不变 |

TMR/HBG、ND/NZ 和长短档策略继续使用同一 CSA 实现。HCA 的算术和 Native 接入流程不改。
路径合并及删除未启用分支不改变现有 CSA 的有效算术、分块和调度策略。

当前 `ops/pypto` 中只保留 `deepseek_v4_flash_csa` 这一套 CSA 实现，目录名不带 `perf`，
也不复用旧精度版的 `deepseek_v4_flash_dspark` 名称。
配置、布局、Native 存储/权重绑定、HC_pre/HC_post、RMSNorm、Q 展开和服务准入/metadata 代码已直接合入 CSA 包；
HCA 从这个包复用辅助函数，不再设置独立 `common` 包。旧精度目录和旧性能目录均已移除，不保留导入别名。
现有 `performance/perf` 配置仍兼容并选择唯一 CSA 实现；性能私有实验副本须完整复制新 CSA 包。
Indexer 中原先仅供精度版使用的 `precision_coefficients` 开关及分支已删除，保留原性能版的 FP16 转换/乘法顺序。

## 决策依据与未关闭项

[七组完整低接受率结果](../../low_acceptance_20261001/RESULTS.md)显示：
128K/B24、8K/B40 的 Native/Native2 和性能 CSA＋PTO HCA/PTO2，各自 token、DSpark 统计和逐轮事件完全复现；
精度版和性能版均仍与 Native 存在差异，没有证据证明继续维护精度版能解决这些分歧。
封存是用户确认的维护决策，**不代表 Native/PTO 精度已通过，也不自动把差异认定为可接受的量化误差**。

性能版后续继续区分功能问题和数值策略差异，固定首分歧前相同输入，先查 metadata/cache/state/保护区，再查 hidden/logits。
[原精度优化迁移](../../precision_port_20260930/RESULTS.md)、
[旧迁移证据](../../results/csa_baseline_20260926/precision_port/migration.json)、
[旧模型看护](../../results/csa_baseline_20260926/model_precision_migration/comparison.json)、
[筛选请求三方一致复现](../../matched_requests_20261001/RESULTS.md)继续保留，各自适用范围不变。
原完整低接受率基准不会被筛选后的简单题替代。

## 本次验证

唯一实现的目录合并及定向检查见[验证日志 §519](../../DSV4_FLASH_CSA_VALIDATION_LOG.md)。

路径迁移仅展开原封存文件，22个Python文件语法解析及diff检查通过；未改运行时代码，未重跑CPU/NPU功能测试。
过程见[验证日志 §518](../../DSV4_FLASH_CSA_VALIDATION_LOG.md)。以下为上一次封存入口变更的验证记录。

结果及边界见[验证记录](verification.json)和[验证日志 §517](../../DSV4_FLASH_CSA_VALIDATION_LOG.md)。
检查默认入口、退役配置拒绝、atomic 默认/覆盖/冻结、实际 ND/NZ 根签名、服务 metadata 绑定和 HBG schema；
再做一轮 B4/H8192 单卡 CSA/HCA 图回放与补位检查。本次不重新跑七档性能或 16 卡精度矩阵。
