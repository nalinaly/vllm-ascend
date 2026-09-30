# CSA / HCA 的 HBG 单卡精度复核（2026-09-30）

本轮检查 HBG 是否相对既有 PTO ring 路径引入数值或状态变化，并记录与 Native 的差异。
使用正式权重、合成输入和历史，不是整模型 token/DSpark 验收，不采性能数据。

## 结果

**四组 HBG/ring 比较均通过，所有比较张量的差异字节数为 0。**
图重放、保护区、只读 metadata 和非有限值检查均通过。

| 算子 / 档位 | history | HBG/ring 逐字节比较 | 图检查 | 对 Native 输出 max_abs | 对 Native 输出 RMSE |
| --- | ---: | --- | --- | ---: | ---: |
| CSA B4/8K | 8192 | 8 项一致 | A→B→A 通过 | 0.03125 | 0.0018983364 |
| CSA B4/128K | 131072 | 8 项一致 | A→B→A 通过 | 0.015625 | 0.0022274856 |
| HCA B4/8K | 8190 | 4 项一致 | 服务 eager / 图重放通过 | 0.015625 | 0.0019002262 |
| HCA B4/128K | 131070 | 4 项一致 | 服务 eager / 图重放通过 | 0.015625 | 0.0013158137 |

CSA 八项是输出、Top-K、SWA cache、compressed cache、compressor state、
Indexer key/scale 及其 compressor state；同输入两次执行的 Native/PTO 自一致性也通过。
HCA 四项是输出和 SWA/压缩 cache/state 的完整原始 allocation（包括保护区）。
HCA 两档 compressed cache 分别实际改写 3818 / 3831 字节，确认压缩写入确实被覆盖。

Native/PTO **不是逐 bit 一致**。CSA 两档各有 24/24 行 Top-K 集合不同，合计替换
117 / 217 个索引；无越界、重复或选取数量错误。这些差异同样存在于 ring，
HBG 的 Top-K 与 ring 完全一致。这里只证明 HBG 未引入额外差异；未把既有
量化/Top-K 差异自动判为符合整模型精度要求。

四个任务全部 exit=0；[精简证据](evidence.json) 保留任务 ID、原始路径、
逐张量差异和 Native 诊断。没有扩大到其他 batch 或 16 卡测试。

## 环境与判据

- CANN 9.2.0-beta.2；算子源码冻结自 `9c46bf05`，排队后不再编辑。
- PyPTO 基底 `d8aed6f82`，另将 kernel ABI pin 和 `runtime` 子模块同步至
  Simpler `ac165482292e6905df27b89deee5fa81ab9a4980`；两侧原生扩展重新构建。
  此配套修复已本地提交为 `4bddb1209`，测试时的源码树与该提交一致。
- 权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`；CSA 第 2 层性能版，HCA 第 3 层。
- B4，每请求 6 个 query token，NZ2，atomic_add=0，deterministic_level=1，
  HCCL_DETERMINISTIC=true，EPLB 关闭，heap 设置 320 MiB。
- CSA history=8192/131072；HCA history=8190/131070，使当前 6 个 token 跨越
  128-token 压缩边界，覆盖 compressed cache 的实际写入。
- HBG/ring 输出、整数索引和 cache/state 必须逐字节一致；图重放与自身 eager
  零容差比较，保护区和只读 metadata 不得改写，不允许 NaN/Inf。
- Native 对照保留逐元素统计及 Top-K 结构/集合诊断，不临时放宽容差；原始报告的
  `MEASURED` 不等于 Native 精度验收通过。

CSA 8K 复用先前已验证的 ring 快照，其余三组在同一任务分配的卡上依次运行 fresh
ring/HBG 独立进程。四个排队任务各占一张卡，没有 16 卡作业。

## 版本更新后的环境修复

前两轮尝试均在调用 PTO 算子前被版本检查拒绝，不能算作精度失败：

1. Simpler 合并上游历史后源码未变，但 editable 扩展仍携带旧提交号，import 拒绝。
2. 重建 Simpler 后，PyPTO 的 kernel ABI 仍固定旧提交号，又在 runtime 初始化拒绝。
3. 同步 PyPTO ABI pin 和 Simpler 子模块，重新构建安装 PyPTO（含 Torch NPU
   适配层）。原生扩展、ABI 描述符、Torch 适配层三方提交号一致；两个相关 CPU
   契约测试通过，再排 NPU 用例。没有绕过版本检查或改动 CSA/HCA 算术。

因此更正此前“合并仅改变历史、无需重建”的说法：这个环境的 kernel mode 绑定了
精确 Git revision，历史变化也需要配套更新声明和构建产物。

## 复现

复用已有单层驱动；[run.py](run.py) 增加 HBG/ring 完整快照比较，
[run_pair.sh](run_pair.sh) 在一个单卡任务内按顺序启动两种 runtime。
在仓库根目录，用新目录冻结源码后提交：

```bash
repo="$(pwd)"
result_root="$repo/tests/pypto_test/results/hbg_accuracy_repeat"
mkdir -p "$result_root"
cp -a vllm_ascend/ops/pypto "$result_root/frozen_ops_pypto"
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 1200 \
  "bash $repo/tests/pypto_test/hbg_accuracy_20260930/run_pair.sh \
  $result_root csa 8192 csa_8k --device 0"
```

另外三组的参数分别为 `csa 131072 csa_128k`、`hca 8190 hca_8k`、
`hca 131070 hca_128k`。`--device 0` 是映射后的逻辑卡号，必须显式传入。
脚本使用公共环境并要求 task-submit 分配单卡。

原始证据目录：`tests/pypto_test/results/hbg_accuracy_20260930/`。
每组 `report.json` 包含 Native 诊断、保护区及图检查；`hbg_vs_ring.json` 为
HBG 相对 ring 的判定，`states.pt` 是本地完整快照。编译产物和大快照不入 Git。

仓库要求的 `bash format.sh ci` 已执行，但缺少 pre-commit，未完成全量格式检查。
