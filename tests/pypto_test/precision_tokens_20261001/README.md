# 128K/B16：CSA 性能版与精度版的 bit / token 对照

算子代码为 `84dc9a3f`，即向精度版迁移数值中性的性能优化之后的版本。本轮不改算子算术、默认 atomic 开关或 HCA。

## 范围与配置

- 单层：复用本提交对应的七档测试中 128K/B16 的两份原始 `outputs.pt`，重新按原始字节做 XOR/popcount；没有重复跑计时。真实 layer_index=2 权重、固定合成历史输入、seed=1024、TMR、NZ2、atomic=0、deterministic=0。
- 整模型：新跑 16 卡，TP1 / DP=EP16 / 每卡 B16，历史 131072，DSpark 出5验6，每条请求生成192个 token，共256条请求、49,152个 token 位置。两侧都启用21层 PTO CSA +20层 PTO HCA，只切换 CSA 的 performance/precision。
- 整模型共用 CANN9.2.0-beta.2、正式 `/data/model/DeepSeek-V4-Flash-0731-w8a8`、NZ2、TMR、atomic=0、FULL_DECODE_ONLY、static kernel、相同 capture sizes 6/24/48/96、无 EPLB。精度诊断开启 `set_deterministic_level(1)` 和 `HCCL_DETERMINISTIC=true`，保留 AIV。
- 共用已经审计通过的 `release_offline_pd_20260923/h131072_bank`。它包含4份固定历史，各 rank 按 rank%4 选取，每卡16请求复用同一历史。这不是256份不同自然语言题目，也不代表任意输入均一致。
- 源码复制到独立目录后才排队，不在执行期间改动。两侧相同 HCA、cache 布局、初始化状态和运行环境；固定 atomic=0 排除两版默认值差异。未测试默认 atomic=1 的精度版对照。

## 单层数值与 bit 结果

| 指标 | 性能版 vs 精度版 |
| --- | ---: |
| CSA 输出类型 / 形状 | BF16 / `[96,4,4096]` |
| 位模式不同的元素 | 441,669 / 1,572,864（28.0806%） |
| 不同的实际 bit 数 | 988,846 / 25,165,824 |
| 最大绝对误差 | 0.03125 |
| RMSE | 0.00221235 |
| 平均绝对误差 | 0.000928999 |
| ULP 中位数 / P95 | 0 / 3 |
| CSA 非有限值 | 两侧均无 |
| Top-K 集合不同的行 | 96 / 96 |
| Top-K 集合平均重合率 | 98.2341% |

两版不逐 bit 一致。P95为3 ULP不意味着所有差异都只有几个ULP：最大30506 ULP出现在性能版0.00631714、精度版−0.00334167这一对，绝对差0.00965881。
Top-K按槽位有45,527/49,152项不同，其中包含排序变化，因此集合重合率更适合描述选中了多少相同候选。
`idx_topk_scores` 的逐槽位误差不是相同ID的分数误差；`state.*` 是允许写入区原始字节，不把不同字节数当成浮点容差。
这些差异与两版保留的规约/量化/Softmax策略差异相容，但本报告不将“精度版”视为数学真值，也没有重新对照Native。

完整位差和误差见 [bit_difference.json](bit_difference.json)。单层样本不能替代整模型token验收。

## 整模型结果

首轮16卡任务 `task_20261001_002841_37675311650` 的49,152个token及全部DSpark统计一致。
但回查发现两侧各4个图档位的静态打包均失败：CANN把长cwd编入shape_info文件名，超过255字节后触发`Errno 36`，框架警告后继续执行，组长`installed_packages=0`。
该轮只保留为初步数值证据，不能声明static kernel实际生效。已将编译cwd移至短路径，补跑任务`task_20261001_004814_7390367673`；最终比较器要求安装数量大于0且无静态编译错误。

正式补跑 `task_20261001_004814_7390367673` 完成，比较结果 **PASS**：

| 指标 | 性能版 | 精度版 / 对比 |
| --- | ---: | ---: |
| 覆盖 rank / 每卡请求 | 16 / 16 | 16 / 16 |
| 生成 token 位置 | 49,152 | 49,152 |
| 两版 token 不同数 | — | **0** |
| DSpark 草稿轮数（跨请求求和） | 8,192 | 8,192 |
| DSpark 提出 / 接受 token | 40,960 / 40,960 | 40,960 / 40,960 |
| 每个草稿位置的接受数（全卡合计） | `[8192,8192,8192,8192,8192]` | 完全一致 |
| DSpark 统计不同的 rank-case | — | **0 / 16** |
| 每 rank 的真实 FULL_tokens96 forward 次数 | 33 | 33 |
| 组长实际安装静态包 / 编译错误 | 4 / 0 | 4 / 0 |

两侧全部 rank 的实际 worker 运行配置相同，完整21层CSA+20层HCA捕获和真实生成期图重放检查通过。
每份历史生成结果包含39种token ID，并非只比较重复的单个token。4份历史及请求复用方式仍限制结论的覆盖面。
结论是：**两版单层bit不一致，但本次128K/B16的16卡整模型输出token及DSpark统计完全一致。**
这不是整模型hidden/logits逐bit对照，亦不能外推任意提示词或默认atomic=1。
完整逐rank结果、实际配置和功能证据见 [token_comparison.json](token_comparison.json)。本轮未采集性能数据；16张卡均已释放。

## 复现与原始记录

- [run.sh](run.sh)：顺序运行两侧整模型、验证实际41层PTO图执行。
- [compare.py](compare.py)：CPU位差及token/DSpark比较；两侧backend均为PTO，不伪装成Native。
- [source.json](source.json)：算子提交、冻结源码位置及固定参数。
- 原始整模型：`../results/precision_tokens_20261001/model_pair_static/{performance,precision}/rank*.json` 和 `rank*.log`。
- 原始单层：`../results/precision_port_20260930/matrix/h131072_b16/{performance,precision}/outputs.pt`。

```bash
source ../env-dsv4-0251rc1.sh
export PYTHONPATH="$PWD/tests/pypto_test:$PYTHONPATH"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/pypto_test/precision_tokens_20261001/compare.py bits \
  --root tests/pypto_test/results/precision_port_20260930/matrix/h131072_b16 \
  --output tests/pypto_test/precision_tokens_20261001/bit_difference.json
python tests/pypto_test/precision_tokens_20261001/compare.py tokens \
  --root tests/pypto_test/results/precision_tokens_20261001/model_pair_static \
  --bank tests/pypto_test/results/release_offline_pd_20260923/h131072_bank \
  --output tests/pypto_test/precision_tokens_20261001/token_comparison.json
```

整模型必须经task-submit申请16卡执行run.sh；传入新的结果目录和冻结源码目录。原始结果目录不能覆盖。
