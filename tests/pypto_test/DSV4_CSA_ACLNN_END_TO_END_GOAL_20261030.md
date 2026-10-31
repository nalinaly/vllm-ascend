# CSA 后置 ACLNN 端到端优化 Goal

日期：2026-10-30。本文记录本 session 已确认的目标与执行边界。

## 1. 最终目标

在 **128K 输入、单卡 B16** 的本机真实 Decode 服务中，沿用客户当前约 **50 ms**
的同一计时区间和统计口径，将 **target主模型一次完整验证forward** 降至 **≤48 ms**。
不包括draft生成候选、forward之后的接受/拒绝和请求全生命周期。

验收同时要求：

1. 正常静态编译能够完成并运行；SuperKernel 融合覆盖的计算范围与 Native 一致。
2. 21层 CSA 的纯 kernel 优化，以及删除 QLI、C4 SAS metadata 推导的收益，
   在真实网络中兑现。给出核内收益预算、Host/metadata 开销变化和端到端结果的对应证据。
3. 不得用单算子、只含 Device 的计时，或独立用例通过，代替真实服务验收。
4. 同容量下实际长度变化不能因 policy 不同而拒绝 graph replay。

50 ms 是客户提供的约数，尚未在本环境重取。先确认实际计时脚本、边界及统计方式，
然后冻结对照条件；不得更换口径来满足48 ms。均值、中位数、最大值、P95及最慢rank
作为配套证据保留，不擅自改变客户主指标。

客户报告的单次约955→772 μs只作为排查线索。若同一forward确有21次等价调用，
对应核内收益预算约3.84 ms；需用同环境、同计算范围的记录确认。两个metadata约200 μs
也须确认是否位于关键路径，不能把重叠区间机械相加。达到48 ms仍须说明收益是否兑现，
不得掩盖剩余接入开销或将未兑现收益写成已完成。

## 2. 固定范围与环境

- 开发：本地 `nalinaly/vllm-ascend` 的 `dsv4-flash-pto-v0.25.1rc1` 分支。
- 最终验收：本机实际vllm-ascend环境的16-worker服务，不再以C3访问作为前置条件。
- 运行时：**仅使用设备后置选档的 CSA ACLNN**。不使用 Simpler/Python 算子调用，
  不以 Python/PyPTO 路径作为业务或验收回退。离线构建已有C++不等于运行时调用Simpler。
- 沿用当前 PD 部署，Decode 为 DP16/TP1；B16、DSpark出5验6时对应T96。
  其余启动参数、权重、请求集、并行配置与客户实际脚本核对后固定。
- Native/PTO 使用相同业务条件、静态编译配置及等价SK覆盖范围；不能通过减少有效
  token、历史、DSpark工作量或改变请求集获得性能。
- 本地SDK使用 `csa_shared_toolchain/manifest.json` 指定的统一版本，遵守公共
  `env-dsv4-0251rc1.sh`、共享 `env.sh` 与 `--shared-toolchain` 约定。
  客户环境单独记录，不跨工具链或CANN版本计算收益。
- MPMD、其他session及Native既有使用方式不因本目标被随意改动。

## 3. 第一阶段：消除稳态 Host 冗余

### 3.1 删除8K人为下界

- 删除后置ACLNN接入路径中将短于8K请求拒绝、回退或强制eager的Host门槛。
- 支持全部有效短上下文，包括压缩候选不足512、尚未产生压缩token、混长和padding。
- 不把“删除Host检查”写成“核内所有边界已验证”。若存在真实核内缺口，允许修复；
  验证实际有效长度、Top-K填充、cache/state写回与完成同步。
- 使用少量针对性的边界场景；不为此重跑全部94档或扩大整模型档位。

### 3.2 DSpark先删除Host限制

用户明确口径：**先只删 Host 限制，发现真实非法访问再讨论。**

- 删除CSA接入增加的、因Host最大长度及DSpark追加token而拒绝正常路径的限制。
- 不预先扩大页表/缓存、不截断有效长度、不丢弃token，也不通过关闭DSpark或强制eager
  来掩盖问题；这些不是当前已确认的处理方案。
- 若确实复现设备非法访问，记录实际Q/L、容量、页号/slot、追加token数及首个失败
  算子，停止该项相关推进并与用户讨论，不能把访问成功或未立即报错当成正确证明。
- 区分CSA与Native草稿模型的首个失败点。已有客户记录指出先失败的是Native MTP
  `SparseAttnSharedkvBnn`；这不等于PTO数学实现已被证明有错，也不证明实际越界无害。

### 3.3 删除不再需要的QLI/C4 SAS metadata

- 在后置ACLNN接管CSA的路径中，删除无Native消费者的QLI和C4 SAS推导。
- 同时检查warmup、capture、replay前更新及fallback路径，不能只在普通eager少算一次。
- 连带删除为这些无用推导服务的Host kwargs、clone、复制及重复准备，不只隐藏kernel。
- 保留Native HCA、SWA、prefill、MTP等实际消费者所需的metadata；按消费关系判断，
  不按算子名字全局删除。
- 用真实执行记录确认目标路径的两类冗余推导消失，并核对关键路径收益。

### 3.4 删除逐层replay签名检查和policy多图机制

- 统一设备选档ACLNN不再每步逐层调用 `allows_replay()` / `graph_signature()`。
- 删除对应的policy图键、按长度枚举policy图、按policy反复选图及拒绝replay逻辑。
- 沿用Native的容量分档、图生命周期和原有执行条件。T/B容量变化与真实Q/L内容变化
  是不同事情；同容量的长度变化不能再要求新policy图。
- 静态ABI、dtype/layout等合同在首次绑定、缓存重建或结构变化时处理；稳态不保留
  这套CSA专属逐层扫描，也不换一种函数名继续逐层扫描。
- 不通过每步全量hash、数组规约、Tensor读回、额外同步或新增复杂检查替代旧检查。

### 3.5 将ACLNN调用热路径简化到必要成本

- 用户再次明确：以普通ACLNN算子的调用方式为准，不另加CSA专属Host校验层。
  初始化所需固定参数检查不搬到每次调用；不以改名、采样或缓存签名等方式保留旧扫描。
- 检查Python服务入口、binding/rebind、C++ Torch包装及ACLNN派发的实际调用频率。
- 清除稳态重复构造的参数列表、描述符、策略对象，以及重复的全量Tensor合同检查。
  复用已准备的绑定；仅在实际地址/结构变化的既有生命周期中更新必要信息。
- 区分真图replay与eager：真图replay不应重新逐层执行Python/C++算子包装。
  先确认实际路径，不能把仅eager执行的检查费用计到每次真图重放。
- 重点跟踪Host冗余→各rank下发时刻偏差→其他rank通信等待延长→target forward变慢。
  对齐16个rank的下发、通信进入/结束及target首尾，不能把通信区间变长直接归因于带宽。
  不为计时插入额外同步，不只优化孤立的Host函数耗时。

## 4. 静态编译与SuperKernel边界

- 第一阶段必须能正常静态编译、捕图和重放；SK覆盖按语义计算范围及forward边界
  与Native对齐，不能只凭两边都设置了SK开关判定一致。
- 检查真实静态核重选与SK dispatch/融合证据，不能将普通二进制执行标为SK已接入。
- 首次静态编译可能超过30秒而报超时，保留缓存和日志，确认该任务已结束后再检查
  第二次是否复用成功，不因一次冷编译超时直接判定算子失败。初始化耗时、重复静态
  编译的进一步提速留到第二阶段。若编译失败或超时导致服务
  无法启动，仍是本轮运行门槛，需做必要修复。
- Python/PyPTO的 `PYPTO_CACHE` 与ACLNN的TorchAir静态/SK编译缓存分开解释。
  本轮只走后置ACLNN，不将优化Python JIT缓存当作本目标的主要进展。

## 5. 近期任务与顺序

- [x] 核对本机实际导入源码/wheel、启动脚本及target设备事件计时区间。
- [ ] 核对客户约50 ms记录对应的输入分配和版本；当前不将本机基线等同于客户约数。
- [x] 对齐后置ACLNN版本；确认业务不走旧Host选档包或Simpler/Python调用。
- [x] 梳理Worker、图包装器、CSA service及Torch包装中的稳态额外处理，标出删除点。
- [x] 删除后置ACLNN的8K/长度Host门槛、逐层签名和policy多图逻辑。
- [x] 删除无Native消费者的QLI/C4 SAS推导及其冗余准备。
- [x] 精简绑定/包装热路径，将必要静态合同检查移入已有准备或重建阶段。
- [x] 单卡验证短历史、混长、padding、同容量跨长度replay及受影响状态写回。
- [x] 验证静态编译和与Native等价的SK融合覆盖；记录实际执行证据。
- [x] 在本机启动真实16-worker服务，重取同口径Native/PTO对照及Host/Device时间线。
- [ ] 达到≤48 ms，并给出21层kernel收益、metadata消除、Host变化与端到端结果的对应关系。

首先集中128K/B16端到端目标。功能验证先尽量使用单卡场景，再做真实服务验收。
发现稳定态瓶颈后按证据处理，不先扩大核内优化范围或启动无关性能矩阵。

## 6. 验证与记录

- 起服、首次编译、warmup与稳态分开；冻结有效样本规则并保留全部样本。
- 核对实际replay命中/回退、每step实际CSA调用次数、SK覆盖、metadata是否执行。
- 同时保留必要Host区间、Device空隙、通信等待、最慢rank与长尾证据。
  整段Host时间不等于某个函数的独占耗时，也不自动等于新增端到端开销。
- 沿用已验收CSA数值策略与既有token/DSpark验证口径，检查本轮新增功能差异。
  Native与PTO既有复杂问题差异如实披露，不把本目标改成新的Native逐bit对齐项目。
- 测试只补受影响项；不反复逐文件hash，不做过度测试。
- 过程、正负结果、复现方式和阶段结论追加到
  [DSV4_FLASH_CSA_VALIDATION_LOG.md](DSV4_FLASH_CSA_VALIDATION_LOG.md)。
- 每阶段使用详细中文说明及Signed-off-by提交；**未经用户新的明确允许，不push**。
- 重大取舍或真实非法访存按约定讨论，不能悄悄缩小有效支持范围。

## 7. 当前事实与后续阶段

截至2026-10-31，**48ms目标尚未完成**。运行时为0.4.0统一设备选档ACLNN，
真实16-worker测试未调用Simpler/Python算子。T96/T6正常静态编译、捕获与replay，
21层CSA均已合入SK；范围证据见
[sk_capture_members.json](csa_aclnn_host_20261030/sk_capture_members.json)。
逐层签名/Host policy、8K与DSpark接入门槛、无消费者QLI/C4 SAS及其重复准备已清理。

固定输入是Native生成的128K公共前缀加问题后缀，准确历史为131104～131346，
每rank固定相同16题，共256请求；不误写为每请求恰好131072，也不通过换题追求48ms。
主指标为预热后连续10个满批T96 target `_model_forward`的设备事件均值，报告最慢rank；
P95/max取16rank全部160样本，保留长尾。独立三步profile只解释原因，不混入验收均值。

| 已完成对照（ms） | Native最慢rank mean | PTO最慢rank mean | PTO P95 | PTO max |
| --- | ---: | ---: | ---: | ---: |
| 移除无用途replay前Host等待，event=1 | 59.645 | 53.259 | 53.810 | 54.277 |
| 同源码，event=0 | 60.624 | 52.617 | 53.238 | 53.499 |
| 再删未消费压缩slot准备，event=0 | 59.678 | 53.040 | 53.792 | 54.349 |
| PTO二维slot融合，Native沿用上一行 | 59.678 | 52.818 | 53.379 | 53.730 |

最新公共代码为第四行；此前52.617ms只是已有最好结果，不能把单次波动当稳定收益。
两侧各自与前一版的256请求token及接受统计保持一致；这不消除原有Native/PTO复杂
问题差异。完整样本分别见
[replay_async_validation.json](csa_aclnn_host_20261030/replay_async_validation.json)、
[event_mode0_validation.json](csa_aclnn_host_20261030/event_mode0_validation.json)、
[unused_compressed_slots_validation.json](csa_aclnn_host_20261030/unused_compressed_slots_validation.json)。

本轮接入：在原slot kernel内同时写出Native二维slot，复用原builder buffer，
保留一维输出，减少DP协调后重复拆分及copy。单卡整数/保护区对照及31项CPU回归
通过。冻结`model_decode_source_fused_slots_v1`的16卡PTO任务
`task_20261031_072638_331273526041`已通过，metadata准备mean为2.333ms，
target最慢rank mean为52.818ms、P95为53.379ms、max为53.730ms。全部256请求
token及16rank接受统计与上一版一致；相对紧邻基线53.040ms小幅下降，仍未优于
历史最好52.617ms，也未达到48ms。不能将Host省下的0.803ms直接计作网络收益。
Native兼容性对照`task_20261031_075603_898532825`也通过，但target为60.266ms，
未优于原版59.678ms。因此公共接入仅在后置PTO ACLNN初始化时启用，Native保持
原路径；不将Native候选变慢算作PTO收益。公共接入33项CPU回归通过。
证据见[fused_slots_validation.json](csa_aclnn_host_20261030/fused_slots_validation.json)。

已停止的候选：二级Host队列被SDK明确拒绝用于图捕获；per-stream一级队列虽单卡
通过，16卡初始化出现rtEventDestroy相关Host堆错误，未进入计时。没有删SDK保护、
改共享工具链或将Host错误断言为DSpark设备越界。原始负面证据、旧版部分静态覆盖
等被撤回结论均保留在[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)，不再在本目标
文档重复过时的排队/验收状态。

第二阶段保留目标：缩短初始化静态/SK编译、减少重复编译并完善部署复现。
依赖第一阶段功能与端到端收益验收；不以第二阶段工作替代本轮稳态Host精简。
