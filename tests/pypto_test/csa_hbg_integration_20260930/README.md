# CSA 独立 HBG 入口与 Host metadata（2026-09-30）

性能版已新增独立的 `decode_csa_tp1_layer_hbg`，Torch 注册名为
`dsv4_csa::attention_hbg`。原有 56 个 Tensor 参数保持顺序，在末尾增加必填的
`host_max_seq_len: pl.Scalar[pl.INT32]`；原 ring 入口的运行时 ABI 仍是 56 个 Tensor。
本轮是接入与单卡功能验证，不是 HBG 性能优化或整模型验收。

## Host 值的来源与用途

`CSAHostMetadata.from_native(indexer_metadata.decode)` 读取 Native 已生成的
`max_seq_lens: int`。其含义是本批次最大的**未压缩 KV 长度，包含当前六个 query token**，
例如 history=8192 时传入 8198，而不是 8192 或压缩后的 2049。
禁止从设备 `seq_lens.max().item()` 取值；适配层拒绝 Tensor、bool、负值和 INT32 越界值。

Host 标量只选择 Indexer Score/Top-K 的任务分支。设备端的 `kv_seq_lens` 继续传入，
用于逐请求可见长度、padding、位置 mask 和任务内循环。Native cache 布局、权重来源及
数值策略没有另建一套；精度版没有增加 HBG 入口，默认 runtime 也没有切换。

原始 HBG Host 访问检查定位到五处：Indexer 两次遍历长度 Tensor，及 HC_pre 的三个
scale 读取。前两处改为消费显式标量；三个 scale 仍保留设备 Tensor，在实际消费它们的
`split_pre_post` / `comb_sinkhorn` SPMD 内读取。

相关实现：

- [两个根入口共用一个函数体](../../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_csa.py)
- [Host metadata 合同](../../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/host_metadata.py)
- [独立注册与 Native 参数绑定](../../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/native_adapter.py)
- [服务入口](../../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/service.py)

入口使用 Python factory 在 IR 解析前选择依赖，HBG 依赖图不会包含旧的 Host Tensor 读取。
旧入口的占位参数是 constexpr，编译时移除。快照工具按运行时参数过滤 constexpr，
已有 Tensor-only 快照仍可回放；HBG 标量尚不支持写入旧的 Tensor 快照格式，显式报错。

## 图重放合同

PyPTO 捕获时会固定 Host 标量，单纯更新设备长度无法改变捕获图的 Host 分支。
当前长度对应三档，判据与算子一致：

| 最大未压缩长度 L | 压缩行数 `L // 4` | 分支 |
| --- | --- | --- |
| 0～8191 | <2048 | Vector Score / Top-K |
| 8192～32771 | 2048～8192 | 短档 Cube / 单叶 Top-K |
| ≥32772 | >8192 | 长档 Cube / 多叶 Top-K |

同一档内可以更新设备 metadata；跨档必须选择或重新捕获对应图。
服务在捕获时记录各 Indexer prefix 对应的 Host metadata，`ACLGraphWrapper`
在重放前检查本步 Native CPU metadata。缺失或跨档会在重放前报错，避免静默复用旧图。
直接使用 `torch.npu.NPUGraph` 的调用方需调用 `validate_replay` / `validate_graph_replay`。

**本轮没有实现整模型自动预捕获三档图，也没有进行 16 卡联合 HCA、逐 token、DSpark
统计或 torch.compile 图验收。** 整模型尝试 HBG 时先使用 `enforce_eager=True`；
手工单卡测试已验证同档更新和跨档重新捕获，不将它称为任意长度的整网图重放已完成。

## 已完成验证

环境：CANN 9.2.0-beta.2；PyPTO `88f605986`；Simpler `a54c05095`；
算子基底 `bc451fe1` 加本提交。正式模型第 2 层权重
`/data/model/DeepSeek-V4-Flash-0731-w8a8`，NZ2、atomic_add=0、单卡、合成历史。
各 NPU 任务均通过 task-submit 自动分配一张卡，冻结算子包后执行。

| 验证 | 结果 | 任务 |
| --- | --- | --- |
| ring 与 HBG 完整 kernel 编译 | 通过，包括 HBG Host 契约检查 | CPU 编译 |
| B4/8K HBG eager、固定 metadata 图 | 通过 | `task_20260930_181118_41325916380` |
| B4/8K ring 对照 | 与 HBG 的八项输出及状态逐 bit 相同 | `task_20260930_181305_45386016233` |
| B4/128K HBG eager、固定 metadata 图 | 图与 eager 逐 bit 相同 | `task_20260930_181607_52190218742` |
| B4 同址 metadata A→B→C→A | 四步均与各自独立 eager 参考逐 bit 相同；跨档拒绝旧图，捕获两张图 | `task_20260930_182217_6605761459` |
| 实际 `dsv4_csa_forward` HBG 服务入口 | eager、图重放均与直接 HBG 入口逐 bit 相同，确认未回退 Native | 同上 |
| CPU ABI、Host 值、模型图保护及快照回归 | 13 passed | [原始测试输出](cpu_tests.txt) |

八项是 `x_out`、`idx_topk`、SWA cache、compressed cache、compressor state、
Indexer key、Indexer scale、Indexer compressor state。写保护区、只读 metadata 和
非有限值检查通过。Native/PTO 的零容差差异仅作原有数值策略诊断，不宣称与 Native
逐 bit 相同，也没有将 `MEASURED` 解释为整模型精度通过。

长度更新用例取 8198→8202→8190→8198；B/C 同时改变输入、位置、页表与 compact metadata，
固定输入地址。每一步恢复各自初态，属于同址更新验证，不冒充连续 decode 轨迹。
服务验证使用实际 opaque custom op，并检查其确实发布了当前 Native Host metadata。

汇总与原始文件位置见 [evidence.json](evidence.json)，其中状态是从原始报告提取的摘要，
不是设备原始日志。HBG 完整编译输出见 [compile_hbg.txt](compile_hbg.txt)。

## 初次 launch 的容量错误

去掉 Host 读取之后，默认 HBG heap 仍使 B4/8K 首次 launch 返回 `-1008`。
在 `KernelDeviceResources::bind` 处读取请求，GM heap 实需 279414784 bytes，
大于默认 268435456 bytes，差 10979328 bytes。初始化后的资源区域已经冻结，不能临时扩大。
原始 bind 片段见 [heap_bind_excerpt.txt](heap_bind_excerpt.txt)。

HBG **kernel mode** 使用 `ring_heap[0]` 作为 GM heap，同时预留相同大小的图定义区域；
不能套用 program-mode 的“忽略 ring_heap”说明。本轮通过已有接口显式设置
`PTO_CSA_RING_HEAP_MB=320`，B4/8K 和 B4/128K 均通过，不修改 PyPTO/Simpler。
这不是其他 batch 的容量保证，也不是 HBG 整体显存只占 320 MiB。
尾部曾带出 allocator/stream 错误文本，容量请求证据将主因定位到冻结 heap 的不足。

## 使用与复现

模型侧选择性能版的独立 HBG 入口：

```bash
export PTO_CSA_VARIANT=performance
export PTO_CSA_RUNTIME=host_build_graph
export PTO_CSA_RING_HEAP_MB=320
```

手工适配层使用 `HBGCSAOperators.register()`、`HBGNativeCSACall(...,
host_metadata=CSAHostMetadata.from_native(indexer_decode))`。
runtime 必须在 `pypto.torch.init` 时设为 `host_build_graph`；ring/HBG A/B 使用独立进程。
HBG 和 ring 共享 tensor/cache/weight 适配，Host 标量没有隐式设备回读兜底。

在本分支 checkout 根目录，以新的输出目录冻结源码并排单卡任务：

```bash
case_output=tests/pypto_test/results/csa_hbg_repeat
mkdir -p "$case_output"
cp -a vllm_ascend/ops/pypto "$case_output/frozen_ops_pypto"
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 600 \
  --env PTO_CSA_RING_HEAP_MB=320 \
  "bash $(realpath tests/pypto_test/csa_hbg_integration_20260930/run.sh) \
  $(realpath "$case_output") --device 0 --batch 4 --history 8192 --hbg-metadata-replay"
```

`run.sh` 将分配的物理卡映射为逻辑 0；提交命令中显式 `--device 0`，避免队列在末尾
自动追加物理卡号覆盖它。验证中有一次因此在设备初始化时报 107001，未执行 CSA，
修正提交命令后重跑通过。
固定 metadata 检查将最后一个选项换为 `--graph --save-state`；128K 使用 `--history 131072`。

CPU 编译使用本目录 `compile.py --output <新目录>`；加 `--legacy` 验证 ring。
CPU 单测需通过 `dsv4_csa_env.activate()` 选择当前 checkout，然后运行
`tests/ut/ops/test_dsv4_csa_hbg.py` 与 `tests/pypto_test/test_csa_replay.py`。
Python/Bash 语法和 `git diff --check` 通过；仓库要求的 `bash format.sh ci`
因环境缺少 pre-commit 未通过，见 [format_check.txt](format_check.txt)。
