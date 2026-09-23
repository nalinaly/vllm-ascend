# 16卡环境接续执行计划

本计划承接2026-09-21本机双卡实验。迁移已经由用户确定；下面是新环境待执行项，
不是已经通过的测试。详细用例仍以[主计划](../DSV4_FLASH_CSA_VALIDATION_PLAN.md)为准。

## 1. 先建立新环境事实，不重做已完成的研究

第一轮只收集和核对，不启动16个独立全模型副本：

1. 获取 `dsv4-flash-pto` 分支，确认起点 `34bb51f9`；读根目录移交文档。
2. 记录新机器NPU型号、真实可见卡数、驱动、CANN、Python、torch/torch_npu及各仓库SHA。
   本机通过的是A3 `Ascend910_9362`；若新机器型号不同，必须重新验布局和原生算子分支。
3. 找到**目标**DeepSeek-V4 Flash checkpoint和DSpark相关权重，记录配置与必要Tensor哈希。
   本机 `/usr/.devenv/models/DeepSeek-V4-Flash-W8A8` 是旧compressed-tensors参考，
   原脚本目标为 `DeepSeek-V4-Flash-0731-w8a8-dspark-0819-new`、`quantization=ascend`。
   不假定两者数值和加载协议相同。
4. 使用新环境已有的模型解释器。先查看import来源；不要重蹈“pto_eager有工具链，所以必须换解释器”的弯路。
5. 核对原生C++扩展和custom算子包来自同一个目标源码/配套发布。
   逐项核对 RMS、rotary、Compressor、`scatter_nd_update_sk`、QLIv2、SAS及其metadata的ABI与stride能力。
6. 固定Native/PTO ND，`weight_nz_mode=0`，实际加载后检查format、dtype、shape、stride。
   新环境若是不同quant method，先验其ND支持，不能直接把本机 `make_attention()` 的compressed-tensors构造照搬过去。
7. 核实indexer cache dtype；A3本机必须显式int8才得到目标4160字节共享页。
8. 建立新结果目录，例如 `results/16card_YYYYMMDD/`；保存新manifest，保留本机历史结果不覆盖。

**完成标志：** 能明确说出加载的是哪套扩展、哪个custom vendor、哪套权重及布局。
仅 `import vllm` 成功、算子符号存在或编译成功，不满足本阶段完成条件。

## 2. 优先跑通Native单层CSA（补完P0、启动P2）

本机最后一个可运行入口是 [dsv4_csa_native_forward.py](../dsv4_csa_native_forward.py)，
它实际构造 `DeepseekV4Attention`，加载第2层真实权重，分配完整128K对应的压缩历史，只执行attention。

新环境首先补齐两处环境适配：

- 环境辅助脚本当前默认加载任务目录 `.cache/csa/native-install` 的扩展，以及固定vLLM checkout。
  新机器使用配套安装包时，更新测试专用loader选择，记录实际路径；生产模型不要依赖这些工作区路径。
- 真实目标若为ModelSlim/Ascend权重，使用其原生加载器构造一层；不要把它作为compressed-tensors强行读取。
  仍然限制只加载单层，先完成可解释的基线。

先执行 BS4、query6、历史131072、seed1024。保留真实state页padding和共享indexer页。
在原生输出有限且所有cache guard通过后，再进入完整PTO对照。

本机最后失败是 `Compressor state_cache stride expected4096/got8192`；当前仓库源码支持后者，
本机旧全局vendor不支持。优先换成配套算子包；有条件时从同一源码完整构建所需custom算子。
不要把cache `.contiguous()` 后不写回，或删掉stride参数，作为问题已解决的证据。

**完成标志：** 保存原生输出及六份可变view，确认未触碰区域不变；不是只看进程无报错。

## 3. 实现真正attention-only PTO adapter

这是尚未完成的开发工作。参考[源码导航与方案](SOURCE_MAP.md)，从 `_decode_csa_tp1` 拆出attention计算。

接口必须同时解决：

1. 外层HC/RMS删除后的 `[T,4096]` 输入输出与内部norm保留。
2. 每请求query6，不能按原参考核S8划分请求；closing token压缩数量可能随起始位置变化。
3. 原生五组页表、累计query边界、实际device长度、INT64 positions、无效slot。
4. state原生页stride，压缩逻辑页单位，indexer K/scale共享allocation及FP16 scale语义。
5. W8A8动态量化、权重scale/offset、QKV和O-proj权重排布。
6. RoPE组、压缩位置和inverse RoPE，以及indexer top-k范围/并列规则。
7. 原生DeviceMetadataExecutor的wait和buffer复用；capture/replay时地址稳定。
8. wrapper支持条件：仅target C4 decode；明确处理或拒绝ragged、mixed prefill、dummy、unsupported dtype/layout。

热路径不得为了获得 `actual_seqlen`、页表或位置而新增 `.cpu()` / `.item()`。
已由host掌握的shape、stride、allocator mirror与device实际内容分别处理。
在custom op注册中声明真实可写cache，提供正确fake，先warmup再捕获。

**完成标志：** hook计数和trace证明目标CSA调用确实进入PTO；直接比较Native/PTO输出和状态。
不能把已有metadata probe挂进forward后称作CSA替换完成。

## 4. 完整P2数值矩阵

仍先在16卡机器上取一张卡完成单层测试，再扩大系统范围。

| 维度 | 用例 |
| --- | --- |
| BS | 4、8、16、24、32、40 |
| 初始历史 | 131071、131072、131073，共18个组合 |
| query | 每请求6，不缩短真实计算行 |
| candidate history | 完整32768附近压缩候选/请求，不缩成几百条 |
| 输入与初态 | Native/PTO同一输入、权重、物理页映射和cache初态 |
| seed | 主矩阵1024；BS4/40增加0、1 |
| 权重 | 合成fixture用于定位；最终目标真实单层权重用于接入验收 |

每个case比较attention输出、SWA KV、main compressed KV、main state、indexer K、FP16 scale、indexer state，
以及所有未触碰页、页padding、无效slot、guard区域。

浮点容差在执行前按Tensor冻结。主计划的BF16输出起点为 `atol=1e-2, rtol=1e-2`，
它不能自动套到FP32 state或量化cache。INT8 K先按相同量化语义要求一致；有阈值差异时独立定位。
Top-K并列规则与可接受的等价性须预先说明，不能看到差异后自动放宽。

**完成标志：** 18组结果及额外seed有可复现、逐Tensor误差与写入范围证据；无未解释差异。

## 5. 完整P3跨步状态、graph和请求生命周期

复用本机已验证的原生接受修正链，但把probe替换成真正CSA计算：

- 每档BS至少100步，确定轨迹 `[1,6,2,5,5]`，均值3.8；每步只推进接受后的真实长度。
- 同地址改长度、页表内容、请求位置，图A→B→A不能读到旧值。
- 覆盖原生 `prev_positions` 更新、重排、请求结束、删除/新请求、页释放/复用。
- 补prefix共享历史及copy-on-write；本机只做了BlockTable行操作，没有完整COW数值证据。
- 每一步保留并比较真实compressor state演进；本机trace每步重置sentinel，不能替代这项。
- 用真实DeviceMetadataExecutor外部事件，检查producer→consumer和buffer reuse。
- Eager与graph使用同一判定；记录实际graph bucket、有效token和padding。

诊断同步只能在测试断言/DFX中加入；正式wrapper和正式性能路径不能依赖它保持正确。
Profiler需检查新增D2H和全局sync，不能只通过源码字符串搜索判断。

## 6. 补完P4，再进入16rank协调

已有双卡结果可作回归基线，新机器至少验证一次最小DP2路径。
完成真正CSA后，再跑同一组不均衡负载和模式切换：

| 项目 | 后续必须补验 |
| --- | --- |
| DP2已有通过 | 实际collective、native dispatcher、非空rank的五组PTO probe和共同容量 |
| DP2尚缺 | Native/PTO完整CSA在相同协调条件下数值一致；真实runner的idle/dummy attention路径 |
| PD差异 | 本机kv_transfer_config=None；目标kv_consumer可能选择不同MoE通信方式、跳过all-reduce |
| DP16 | 实际16rank的skip判定、token/padding、图模式、有效请求分布；不能照抄DP2结果 |

必须保留DeepSeek MoE分类，记录 `should_skip_allreduce_across_dp_group` 的实际值和原因。
目标consumer若合法跳过同步，就验证该分支的本地有效数据；不要强制所有rankpadding来掩盖差异。

## 7. P5真实16卡整模型验收

完整模型阶段使用目标真实权重、真实DSpark drafter及完整cache生命周期。
先Native ND基线，再PTO ND；同一输入、随机种子、量化格式和服务配置。

| 编号 | 内容 | 需要留存的证据 |
| --- | --- | --- |
| F01 | 全部目标CSA层接入，其他层/路径原生 | 每层替换计数、支持条件、原生/PTO输出、cache与错误 |
| F02 | TP1、DP=EP16、BS/GBS与graph | 每rank实际BS、token、bucket、group size和通信分支 |
| F03 | PD cache转移与层级通知 | Prefill→decode真实链路、KV connector、恢复/复用；本机随机cache不能替代 |
| F04 | EP16和EPLB | 启用配置、实际group、初始化、至少一次实际重平衡及运行正确性 |
| F05 | 真实出5验6 | 自然接受分布、bonus口径、实际有效推进和模型输出 |
| F06 | 稳定性与性能 | HBM峰值、超时/错误、step latency、吞吐，Native ND与PTO ND同口径 |

从GBS64/单卡BS4起跑，再按已确认的GBS口径覆盖BS8/16/24/32/40。
单个decode调用的工作量仍为6×BS，吞吐分母用真实输出/接受计数。
3.8是目标统计口径，不要求通过伪造接受值使真实drafter实验恰好等于3.8。

只有16张decode卡时，PD prefill服务可能需要外部已部署资源；不能默认把脚本中的prefill DP4×TP4
再塞进同一16卡机器。若缺少该资源，F03保持待验，其余可独立推进。

## 8. 最终交付要求

- 可运行的attention-only PTO代码和明确的支持条件。
- 真实16卡的环境/权重/有效配置manifest，且与本机参考环境区分。
- P0～P5状态矩阵，失败必须有原因和可复现case，未跑项目保持NOT_RUN。
- 每Tensor误差、cache写入范围、连续状态、graph、DP/EP/EPLB、PD和真实接受统计。
- Native ND/PTO ND性能，分开报告算子、wrapper、单层、整模型；DFX与正式计时分开。
- 更新专用验证记录。全流程尚未通过时不能只提交“编译成功/运行未崩溃”作为完成结论。
