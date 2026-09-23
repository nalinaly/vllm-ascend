> **2026-09-23 基线已迁至官方 v0.25.1rc1。** 当前入口为[基线迁移说明](BASELINE_MIGRATION_V0251RC1.md)。
> 当前 release 已完成H255、每卡B4的离线P/D16接入及生成token对照，详见下方当前状态。
> 稳态性能及其他场景尚待验证；旧单层精度/图脚本仍需按新接口适配。

# DSV4 Flash：离线 P 缓存与 D16 CSA 性能对照

2026-09-23 用户确定：先由 P（TP4×DP4，EP16）生成不同场景的离线 KV cache，
释放全部16卡，再供 D（TP1×DP16，EP16）验证 Native/PTO CSA 接入和性能。
仅使用 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 正式75分片 ModelSlim 权重。

## 当前状态

release P TP4×DP4/EP16短场景H255×4已完成，16份缓存副本全部落盘；每份191个tensor，
覆盖43个target层和3个DSpark层。有效前缀数据直接逐bit比较通过，不使用hash校验。
D TP1×DP16/EP16每卡B4的Native轮已完成：64个请求均加载P缓存并各输出128token。
PTO首轮在初始化时发现release A3 compressor norm为BF16、旧适配层要求FP32；
按用户要求改CSA主入口及两路compressor为BF16，直接绑定Native norm地址，
在已有RMS设备任务内转换加载的tile，撤销初始化FP32副本；旧重提任务已取消。
CPU零拷贝准备检查及完整CSA lowering/PTOAS代码生成通过。
新任务`task_20260923_163126_45104725192`已结束，输出decode_pto_b4_bf16_v3；
BF16权重准备和64个请求缓存恢复通过，首次PTO调用前被idx_kv_cache可写别名检查拒绝。
已确认是同字节范围的合法共享视图，更新PyPTO调试分支至5495749（PR #2867）后，
16rank×21层的CPU描述符检查通过。Simpler调试分支更新至166852bf，两者重新安装，
ABI一致性、完整CSA编译及一个eager真机用例通过。
PTO D16重测任务`task_20260923_175252_286523232409`已完成exit0，输出
`decode_pto_b4_updated_v4`。正式权重、history255、每rank B4：64请求共8192个生成
token与已有Native D16结果完全一致，16rank×21个CSA层均记录PTO执行。
这是本场景接入运行及生成token对照通过；首次编译、IO和观察hook仍在计时范围内，
尚无稳态性能结论，其他离线场景与逐层张量精度验收未因此完成。
长场景尚未生成。P3/P4其他场景和剩余精度排查仍暂停。

## 场景与公平比较

| 参数 | 固定口径 |
| --- | --- |
| P 历史长度 H | 255、4095、32767、131071、131072、131073 |
| 每个 H 的输入 | 4份确定性文本 token 序列，每个 P DP rank 生成1份；记录完整token ids |
| D 的每卡 BS | 1、4、8、16、24、32、40；GBS=16×BS |
| 初始负载 | 16个 D rank 均衡、同长度；每个 rank 使用对应 P DP rank 的缓存模板 |
| 缓存模板复用 | 多请求可复制同一模板，但在 D 分配独立物理页；关闭 prefix sharing |
| DSpark | 真正 draft 模型，5个 speculative tokens；target 稳态 query S=6 |
| 量化/布局 | ModelSlim W8A8、BF16 activation、weight ND、KV ND、INT8 indexer、block32 |
| 初始执行 | eager；full graph 待实际 DP padding 条件和接入命中核实后另列 |

H 表示 **P 已计算的历史 token 数**，不是 D prompt 总长度。
D 提交同一序列的 H+1 个 token，先加载 h(H)，再计算最后一个 prompt token。
与 Native hybrid PD 的 N−1 边界一致，避免 compressor state 重复更新。
后续真实 DSpark 接受轨迹可能使 Native/PTO 的请求批次不同；对照报告必须记录
接受率、实际 query 数和 PTO 命中次数，不能只按名义 BS 计算加速比。

## 缓存内容和生命周期

`offline_pd/connector.py` 是测试专用、通过标准 `kv_connector_module_path` 注册的
HMA connector。没有修改 PyPTO、Simpler、pypto-lib 或锁定 vLLM 依赖。

- 保存所有 target 43层以及 draft 的全部 Native cache group；包括 SWA、C4/C128
  压缩 KV、INT8 Indexer K/scale、两个 compressor state 和 draft SWA。
- 读取调度器当步的真实 HMA block table；按各组 logical block size 截取已计算范围，
  记录 SWA 的逻辑页号和空洞。只保存 tensor 的逻辑内容，不落盘设备指针。
- P 在完成整个 prompt 的 target/draft forward 后，由 Native
  `finalize_kv_connector → wait_for_save` 同步保存。跨块 prefill 的中间状态不作为最终缓存。
- TP4四个rank分别落盘。manifest 校验各 rank 的内容、dtype、页布局和层覆盖；
  全部一致后，D 才可选取 tp0 的复制缓存。不能把 TP4 分片未经核对直接当成 TP1 缓存。
- 快照只包含已计算前缀：最后一页超出H（压缩缓存为floor(H/ratio)）的行在CPU序列化副本中清零。
  schema1原始bank保持不变，D加载时按同一边界清零；draft已预测的未来行不能算作历史缓存。
  层覆盖明确识别release的`mtp.0/1/2`。按用户要求，不生成或检查hash；直接比较必要数据与布局。
- D 按自身实际页表重映射，核对全量层名、shape、dtype、block size 和所需逻辑页完整性。
  在报告异步 load 完成之前同步拷贝；之后恢复 Native 调度，不改生产算子输入合同。
- 文件先写临时文件再发布 manifest；拒绝覆盖已完成缓存。每次场景生成用新 bank。

## 执行入口

从仓库根目录执行。`plan`/`audit` 只使用CPU；P/D必须经过 `task-submit`。

```bash
source ../env.sh
python tests/pypto_test/offline_pd/run.py plan --bank /path/to/new/bank

task-submit --device auto --device-num 16 --max-time 7200 \
  'cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1 && source ../env.sh && python tests/pypto_test/offline_pd/run.py prefill --bank /path/to/new/bank --output /path/to/p-logs'

python tests/pypto_test/offline_pd/run.py audit --bank /path/to/new/bank

# P 完成并释放卡、audit通过后，才依次启动两个D后端。
task-submit --device auto --device-num 16 --max-time 7200 \
  'cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1 && source ../env.sh && python tests/pypto_test/offline_pd/run.py decode --bank /path/to/new/bank --output /path/to/native-d-logs --backend native --batch 4'

task-submit --device auto --device-num 16 --max-time 7200 \
  'cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1 && source ../env.sh && python tests/pypto_test/offline_pd/run.py decode --bank /path/to/new/bank --output /path/to/pto-d-logs --backend pto --batch 4'
```

本机默认控制地址192.168.0.106、网卡enp23s0f3、DP端口29683，可用参数替换。
设备仅来自 `$TASK_DEVICE` 的16卡队列分配，P每rank4张、D每rank1张。
任一rank失败会关闭本次启动的其余进程，不清理他人服务。

## 性能结果的出口条件

当前 `rank*.json` 中的 `elapsed_including_io_seconds` 包含离线加载和启动开销，
**不作为 decode latency、吞吐或 CSA 加速比**。后续对照需：

1. 同一bank、同一输入/BS/精度/布局/调度参数分别运行 Native 和 PTO；两端分别预热。
2. 先核对 D 接收的 cache/state 以及所有21个 target C4层的实际入口；明确回退原因。
3. 用独立的无profile计时测稳态 decode step、p50/p95、总token/s、接受率与显存。
4. 另采 PyTorch/NPU profiler 与 PTO 泳道图，区分完整 forward、21层CSA、Indexer、
   MoE/EP通信和draft；排除加载、首次编译、warmup和首个单token恢复步骤。
5. 若实际执行包含padding或Native回退，分别报告，不能将其计为全量PTO加速。

离线回放能验证缓存复用和 D16计算/EP接入，但不能替代在线Mooncake传输、P/D网络延迟、
故障恢复或完整F01～F06验收。现阶段无需安装Mooncake即可开展这一轮工作。
