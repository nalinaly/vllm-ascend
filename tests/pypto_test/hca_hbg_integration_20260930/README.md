# HCA 独立 HBG 入口与 Host 标量（2026-09-30）

本轮直接修改 `dsv4-flash-pto-v0.25.1rc1`，基底 `d468bd64`，未新建分支。
新增 `decode_hca_tp1_layer_hbg` / `dsv4_hca::attention_hbg`，在原 36 个 Tensor
之后增加三个必填 FP32 标量 `host_hc_scale0/1/2`。默认仍是 ring。
这是功能接入：HBG 图重放通过，但设备区间严重慢于 ring，尚不能作为性能路径。

## 为什么这里没有增加 Host seqlen

当前 HCA 的 `seq_lens`、`positions`、页表、slot 都在 CORE_GROUP/SPMD 设备任务内读取。
Attention 长短策略由 `cmp_table` 的**形状容量**选择，实际可见长度由设备 metadata mask。
与 CSA 的 Indexer 不同，HCA Host orchestrator 不读取长度 Tensor 的内容，
因此不增加没有消费者的 Host seqlen，也不引入每步 D2H 同步。

真正阻塞 HBG 的是 `hc_pre_fused.py` 的六次 Host `pl.read(hc_scale, ...)`：
`_hc_pre_partials` 三次没有数值消费者；`hc_pre_norm` 三次参与门控。
本轮用 Python factory 在 IR 解析前选择依赖，两个入口共享算术、tiling 和缓存适配。
ring 分支保留原来的六次读取；HBG 分支只使用显式标量。没有重复复制整套计算源码，
文件 diff 中大量变化只是放入 factory 后的一级缩进，可用 `git diff -w` 审查。

加载期 `prepare_weights(..., host_scalars=True)` 从已准备的 FP32 `[3]`
`hc_attn_scale` 一次取出 Python float，检查有限值；调用和图重放时直接传入。
Tensor 权重和三组 Native cache 的绑定继续复用原实现。原 Torch ABI 仍为 36 Tensor，
HBG 是 36 Tensor + 3 float，没有默认零值。原始 JIT 的 `param_names` 可能包含 constexpr，
适配层按实际 Torch schema 绑定参数。

Host 标量属于该层的常量权重，不是每步长度。权重重新加载或原地更新后，必须重新
准备 Host 权重并重新捕获图；只改设备 scale Tensor 不会修改已捕获的 Host 标量。
HCA 本身不需要 CSA 的长度分支 guard。联合模型仍受 CSA 的 Score 拓扑捕获约束，
不能把 HCA 的同址更新验证外推为 CSA/HCA 联合整网任意长度图已经通过。

## 验证与结果

CANN `9.2.0-beta.2`，PyPTO `88f605986`，Simpler `a54c05095`；本轮没有修改后两个仓。
正式 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 第 3 层权重，合成输入及历史，NZ2，
atomic_add=0。所有设备命令均经 task-submit 分配单卡，排队前冻结整个算子包。
`PTO_CSA_RING_HEAP_MB=320` 仅记录本轮容量设置，不代表其他 batch 的下限或总显存。

- HBG 和 ring 的真实 kernel ABI CPU 编译均通过；HBG Host 访问检查通过。
- 两项 CPU 单测验证独立 ABI、必填参数、初始化快照及非有限值拒绝。
- B4/8K 和 B4/128K：HBG 与改动前 ring 的输出、SWA cache、compressed cache、
  compressor state 四项逐 bit 一致；服务 custom op、图重放、保护区通过。
- 同一张 HBG 服务图：history **124→131070→124**，固定张量形状与地址、改变
  23 项 metadata 以及输入/位置/页表内容。每步与自身 eager 比较，B 另与独立
  张量上的直接算子比较，回到 A 再与首次 A 比较；四项输出/状态和 18 项保护检查通过。
  两份 fixture 使用长档页表容量；每步恢复对应初态，不称为连续 decode 轨迹。
- 原入口兼容性回归通过：新 factory 的 ring 路径在 B4/8K 与改动前快照四项逐 bit
  一致，服务/图重放及保护区通过，见 `evidence.json` 的 `ring_compat`。

Native 零容差对照仍存在既有数值差异，两个档位输出 max_abs 都为 0.015625。
HBG 与原 PTO 的比较为零差异；没有做 16 卡、整模型 token/DSpark 或联合 HBG 验收。

设备图重放诊断，单位 μs，预热 5 次后取 10 次，保留全部样本：

| history / B | ring min | ring mean | ring max | HBG min | HBG mean | HBG max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 / 4 | 310.96 | 324.63 | 332.32 | 67892.06 | 68801.62 | 69915.18 |
| 131072 / 4 | 349.74 | 365.74 | 382.70 | 74300.36 | 75613.50 | 76683.20 |

区间为 mHC pre + norm + HCA + mHC post，含本层 compact metadata；同一卡上的
ring/HBG 为先后独立进程，各进程内交替 Native/PTO 并恢复初态。直接算子 HBG
均值分别 69014.42 / 75556.21 μs，说明移除服务层 compact metadata 不能消除退化。
这是手工 NPUGraph 的接入诊断，未走正式 `torch.compile(backend="npugraph_ex")`
+ SK1 全七档验收，不替换已有 Native/PTO 正式性能表。

## 对旧 HBG 结论的修订

[优化日志第 94.5、96 节](../DSV4_FLASH_HCA_OPTIMIZATION_LOG.md)的慢现象仍有复现；
“与 NPUGraph 不兼容、每次 replay 在 Host 重建任务图”的成因解释撤回。
本轮证明当前 HBG kernel mode 可以捕获并正确重放，且同址 metadata 更新生效。

Simpler `a54c05095` 的实际路径是：

- `src/common/host_build_graph/device/kernel_graph_execution.cpp` 的
  `consume_kernel_task` 在 AICPU thread 0 调用 `restore_graph_packet`，随后才 `aicpu_execute`。
- `kernel_graph_restore.cpp` 先校验图包及逐任务/参数描述，清零本次使用的临时 heap，
  复制图镜像、重建调度状态，再 flush heap 与镜像。
- `kernel_graph_wire.h` 的 `validate_graph_packet` 校验覆盖图包内容的 checksum。

这些是**源码确认的每次设备调用工作**；尚未分别计时，不能把 68–76 ms 全部归因于
其中任何一步，更不能再归因于 Host 每次重建。后续性能定位应优先量出这些环节，
本轮没有为了提速跳过校验、清零或更改 Simpler。

## 复现与证据

原始结果目录：
`tests/pypto_test/results/hca_hbg_integration_20260930/`。
冻结的 `legacy_ops` 来自基底提交，`hbg_ops` 来自本次已验证源码。
报告、完整 allocation 快照、编译产物留本地；精简证据见 [evidence.json](evidence.json)。

```bash
source ../env-dsv4-0251rc1.sh
VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 \
  python tests/pypto_test/hca_hbg_integration_20260930/compile.py --output /tmp/hca_hbg_compile
# 原入口 CPU 编译加 --legacy。

# repo/result_root 使用绝对路径；首次复现需先从相应版本冻结 legacy_ops/hbg_ops。
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 1200 \
  "bash $repo/tests/pypto_test/hca_hbg_integration_20260930/run_pair.sh $result_root 8192 --device 0"
# 长档改为 131072。--device 0 是映射后的逻辑卡号，必须显式提供。

/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 900 \
  "PTO_CSA_RING_HEAP_MB=320 bash $repo/tests/pypto_test/run_hca_metadata_replay.sh \
  $result_root/metadata_hbg --runtime host_build_graph --batch 4 \
  --history-a 124 --history-b 131070 --operator-source $result_root/hbg_ops --device 0"
```

服务使用已有 `PTO_CSA_RUNTIME=host_build_graph` 开关选择 HCA HBG 注册入口；联合 CSA 时
还需 `PTO_CSA_VARIANT=performance`。同进程初始化后不要切换 runtime。

`bash format.sh ci` 已执行，但环境缺少 pre-commit，未完成全量格式检查。
CPU 单测、编译及单卡结果不等同于该检查通过。
