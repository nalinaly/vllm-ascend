# PTO CSA 服务 forward 接入

2026-09-23：服务入口已实现。正式权重下 B4/B40 的真实单层服务调用、编译边界、
graph A→B→A 与 Native 回退已验证。按用户要求删除离散 batch 白名单并补齐完整链路尾块后，
最新版本正式B1启动capture/重放及B1→2→3→4→5→40→1同产物切换均通过，见验证日志第83节。
本页不代表完整模型 HTTP、16卡或 P5 验收。

## 入口与数据路径

显式模型架构 `PyptoCSADeepseekV4ForCausalLM` 继承 Native 模型、权重加载和其余计算，
只替换 target 的 C4 attention 实例：

```text
DeepseekV4Attention.forward(positions, hidden_states, llama_4_scaling)
  -> torch.ops.vllm.dsv4_csa_forward
  -> CSAServiceRuntime / NativeCSACall
  -> 一次完整 PTO CSA
```

positions、五组 metadata、六份 cache/state 使用本次 Native Tensor。模型权重量化完成后
准备 ND 权重和每层工作区，Native Hadamard 在首次普通 warmup 时准备；capture 不允许
临时初始化。Native COMPRESSOR/INDEXER/ATTENTION 事件等待以及 connector 的等待、
KV 写入通知和保存都保留。kernel 异常直接报错，不在部分写入后偷偷回退。

实际 B/T 是 `B_DYN`/`T_DYN` 运行时维度，`T=B*6`。配置 `max_num_seqs` 决定服务工作区
容量；本版算子内部固定分配上限为64请求，因此每层分配 `min(max_num_seqs,64)*6` 行，
容量以内不使用离散 B 白名单。这不代表 S 可变，也不代表任意大的 B 已支持。
Prefill、非均匀 S6、超出工作区容量、padding/dummy 请求等继续走 Native。
启动 capture 用的整批占位输入单独验证，不等同于真实调度中的 padding 支持。

ACL Graph 每个 bucket 仍有自己的固定地址和形状。runner 在选择 graph 时检查实际请求数：
会捕获 PTO 的 bucket 必须是完整、无 padding 的 S6 批次，否则使用 eager；其他 bucket
继续使用 Native 图。动态图维度与一张 ACL Graph 跨形状复用不是同一能力。

## 启用参数

沿用真实 Native D 服务的设备、DP/EP 和 PD connector 配置，增加或替换以下参数；
这是参数片段，不是一个可在单卡装入完整模型的启动命令。

```bash
--model /data/model/DeepSeek-V4-Flash-0731-w8a8
--hf-overrides '{"architectures":["PyptoCSADeepseekV4ForCausalLM"],"sliding_window":128}'
--dtype bfloat16 --quantization ascend --tokenizer-mode deepseek_v4
--tensor-parallel-size 1 --pipeline-parallel-size 1
--block-size 32 --attention-config '{"indexer_kv_dtype":"int8"}'
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}'
--additional-config '{"weight_nz_mode":0,"enable_kv_nz":false}'
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[6,12,18,24,30,48,96,144,192,240]}'
```

`vllm serve` 的模型路径是位置参数；使用该 CLI 时用路径替代上述 `--model` 行。
合并已有 additional-config，不丢弃其他服务配置。当前要求 Model Runner V1，
可用已有配置 `VLLM_USE_V2_MODEL_RUNNER=0` 固定。不开 graph 时使用 `--enforce-eager`。
DP>1 的 full graph 仅在 Native 本身选择跳过跨 DP padding 时启用；否则初始化明确拒绝，
先使用 eager。此次没有改 DP padding 协议，也没有恢复暂停的 P4 全矩阵。

当前环境执行 `source ../env.sh`，使用已安装的本地 PyPTO/Simpler 调试分支和锁定 vLLM。
CANN9.0 本地扩展已通过两个忽略跟踪的 `.so` 链接接到 `vllm_ascend/`，指向
`.cache/csa/native-install/` 中已验证的产物；无需由测试 helper 注入模块才能导入。
新机器应按正常包构建流程安装扩展，不提交这些本地二进制链接。

## 验证入口

所有 NPU 执行都经 `task-submit`；以下 Python 命令放在已 source 环境的队列任务脚本中，
`--device` 使用任务分配的 `$TASK_DEVICE`。

```bash
python tests/pypto_test/dsv4_csa_service_forward.py \
  --device "$TASK_DEVICE" --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
  --batch 3 --dummy-capture --output-dir /path/to/new/result

python tests/pypto_test/dsv4_csa_service_dynamic.py \
  --device "$TASK_DEVICE" --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
  --batches 1,2,3,4,5,40,1 --output-dir /path/to/new/switch-result
```

第一项进入安装后的真实 attention.forward，使用正式 Native 量化权重、Native metadata
producer/ExternalEvent，核对输出、Top-K、六份 cache/state、保护区、torch.compile 和
同图重放。第二项在同一进程、同一服务层和同一个已注册算子中切换 B，检查每次都实际
进入 PTO，并检查编译 artifact 没有随 B 变化。
这些仍是单层验证；整模型加载、所有21个C4层、真实 DSpark 调度、PD/EP/EPLB 待 P5 验收。
