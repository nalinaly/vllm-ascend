> **2026-09-23 基线已迁至官方 v0.25.1rc1。** 当前入口为[基线迁移说明](tests/pypto_test/BASELINE_MIGRATION_V0251RC1.md)。
> 下方历史 PASS 对应旧基线；本次 release 尚未进行真机数值验收。历史脚本需按新接口适配。

# DeepSeek-V4 Flash CSA：16卡环境接续入口

> 当前session已按用户要求在16卡CANN9.0环境继续开发。以下正文是旧机器移交快照，
> 当前状态见[验证计划](tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_PLAN.md)、
> [验证记录第30节及后续](tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md)和
> [CANN9.0环境记录](tests/pypto_test/handoff/ENVIRONMENT_CANN90.md)。
> 用户于2026-09-23明确要求：后续全部使用正式权重，旧cann_recipe仅保留历史证据。
> 正式权重已下载完成：
> `/data/model/DeepSeek-V4-Flash-0731-w8a8`。75分片结构、Native ModelSlim单CSA层加载/ND布局
> 及正式B4/B40、history131071完整CSA比较已通过；正式P2为2/18。
> 最新SWA RMS修正后，正式B40前47步及B4图重放通过；step46输出不同元素32707→5279，
> RMSE=0.00005738172331，max_abs=0.00390625；剩余attention 35个BF16差异，详见第79节。
> 2026-09-22最新：本仓库完整CSA已接入Native存储；已去掉普通RoPE、token metadata
> 及两次compact metadata外部调用，消费者直接使用Native device Tensor。
> 最新v4直接读写两组Native state，四次state适配与两份14行窗口也已删除。
> B4/B40完整比较、B4图重放及B4 mixed10 eager/graph通过，profiler确认一次完整CSA提交；
> 9月23日服务forward入口已实现，正式B4/B40单层调用与graph验证通过；动态B白名单已删除，
> 完整链路尾块修复后，正式B1图重放及B1→2→3→4→5→40→1同编译产物切换通过。
> 完整P3/P4、正式矩阵和P5未验收，见记录第81～83节。
> 2026-09-23用户指定后续路径：P TP4×DP4先生成多场景离线cache/state，释放卡后由
> D TP1×DP16、EP16做Native/PTO接入和性能对照，见
> [离线P/D方案](tests/pypto_test/DSV4_FLASH_CSA_OFFLINE_PD.md)与记录第84节。

本文件保存截至 **2026-09-21** 的工作上下文。用户已决定把后续开发和完整验收迁到真实16卡环境。
本机不再继续修补环境或运行新的数值实验；刚才已启动的双卡测试正常结束，结果已归档。

**当前交付的是验证代码、证据和接续计划。完整 PTO CSA attention adapter 尚未实现，完整数值对照尚未通过。**
请先阅读本文件，再读下列材料；不需要依赖旧聊天记录或原机器的 `.cache` 才能理解任务进度。

| 材料 | 用途 |
| --- | --- |
| [原始详细验证计划](tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_PLAN.md) | P0～P5、用例编号、支持边界和验收标准 |
| [实际验证过程](tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md) | 已执行操作、失败、纠正过程；最后两节是最新状态 |
| [16卡接续步骤](tests/pypto_test/handoff/NEXT_STEPS_16CARD.md) | 到新机器后依次做什么、何时能进入整模型验收 |
| [环境与构建记录](tests/pypto_test/handoff/ENVIRONMENT_AND_BUILD.md) | 固定版本、本机环境偏差、重建方法、不可直接搬用的产物 |
| [源码导航与设计笔记](tests/pypto_test/handoff/SOURCE_MAP.md) | 原生调用链、PTO参考、尚未实现的适配方案 |
| [验证入口说明](tests/pypto_test/README.md) | 现有脚本、实际命令和适用范围 |
| [交接时结果清单](tests/pypto_test/results/local_20260921/handoff_state.json) | 可机读的最终状态；早期manifest中的历史状态不覆盖它 |

## 1. 获取代码和基线

```bash
git clone --branch dsv4-flash-pto https://github.com/nalinaly/vllm-ascend.git vllm-ascend-dsv4-flash-pto
cd vllm-ascend-dsv4-flash-pto
git log -3 --oneline
```

仓库是 **`nalinaly/vllm-ascend`**，不是早期消息中漏了末尾 `d` 的拼写。
开发分支为 `dsv4-flash-pto`；本次交接提交位于以下官方起点之上：

| 项目 | 提交 |
| --- | --- |
| vLLM-Ascend 官方起点 | `34bb51f93724c565362f5108f5226303e1b56cad` |
| 配套 vLLM | `84030bbe3d74d99bad477a3d2e37a973ccd8865c` |
| pypto-lib 参考 | `205255b4770ee84dfa176bcbc7bbef651953c7e1` |
| Qwen3 参考分支提交 | `093b1eb519f2a706b8b471d40cc4c66c64e68f1b` |

已核对 Qwen 分支和 DSV4 分支的 merge-base 就是 `34bb51f9`。
用户要求的是同一个官方起点，**不包含 nalinaly 原工作区的私有业务修改**。
本轮没有改动生产模型、attention、runner 或 C++ 算子源码；本机兼容实验只存在于隔离构建目录，
相应差异作为历史patch归档。不要把原工作区旧 CSA adapter 整批复制进来。

PyPTO、Simpler、PTOAS、PTO ISA 的精确版本和本地差异见
[source_lock.json](tests/pypto_test/handoff/source_lock.json)。不应把移动中的分支名当成版本锁。

## 2. 用户已经确定的目标

仅替换 target model 中 `compress_ratio=4` 的
`DeepseekV4Attention.forward(positions, hidden_states, llama_4_scaling)`。

- 入口/出口是 `[T,4096]` hidden states，首个目标为 BF16。
- 输入已经过外层 RMS；外层 `hc_pre`、RMS、`hc_post` 不放进 PTO。
- Q/K norm 和 compressor norm 位于 attention 内部，仍须保留。
- Prefill、SWA-only、HCA、drafter、MoE、EP/EPLB 计算继续使用原生路径。
- 参考 Qwen 的实例级 forward 替换、Torch custom op、fake、warmup、graph 接法；
  计算参考 pypto-lib 的 `_decode_csa_tp1`，但不能直接套用它包含 HC 的51参数接口。
- **Native/PTO 全流程 ND**。当前PTO不支持NZ，已明确不做NZ对照。
  当前仓库有效控制为 `additional_config.weight_nz_mode=0`；ND不等于取消W8A8量化。

最终测试参数：

| 参数 | 已确定值 |
| --- | --- |
| 阶段 / 并行 | decode；TP=1，DP=EP=16 |
| 全局基准 | GBS=`16×4=64` |
| 单卡BS扫描 | 4、8、16、24、32、40 |
| 历史长度 | 128K，当前计划按131072 token处理 |
| DSpark | 出5、验6；target每请求6行 |
| 接收步长 | 3.8；本机用确定轨迹，整模型记录真实分布 |
| EPLB | 开启，需要真实16卡验收 |
| Target graph | `FULL_DECODE_ONLY`，保留脚本NPU graph选项 |
| Scheduler | async、prefix caching、hybrid KV manager开启 |
| 容量 | block_size=32；max_model_len=1048576；max_num_batched_tokens首轮400 |
| 请求上限 | 从旧脚本32调整到至少40 |

待在新环境确认的参数含义，不要自行改成已确定事实：

1. GBS=64是BS4基准点，还是所有实验都固定64？当前计划采用前者，均衡扫描可到GBS640。
2. 3.8是否包含bonus token？本机轨迹按每步有效推进 `[1,6,2,5,5]`、均值3.8处理。
3. 最终checkpoint及量化格式、PD服务地址、是否启用DSA-CP/跨DP O-proj TP。
4. 原脚本没有显式EPLB开关，不能以 `--enable-expert-parallel` 代替EPLB启用证据。

## 3. 已验证到哪一步

| 项目 | 实际状态 | 结论边界 |
| --- | --- | --- |
| 两卡各自PTO小核和最小graph | PASS | 不含attention数学计算 |
| 共享INT8 K/FP16 scale、offset、padding | PASS | 单个allocation作为可写载体的方案已由小核验证 |
| P1 M01～M12 | PASS（metadata） | 六档BS、五组原生builder、设备slot展开、负例、A→B→A |
| 100步接受修正 | PASS（metadata） | 六档BS；原生CPU/GPU接受修正；每步检查sentinel，非完整CSA状态 |
| 实际TP1/DP2 | PASS（metadata） | HCCL world + DP CPU Gloo；不均衡负载、模式同步和PTO探针 |
| 真实单层W8A8权重加载/ND | PASS（参考checkpoint） | 第2层21个权重，168.7 MiB；不能替代最终checkpoint验收 |
| 原生完整attention.forward | FAIL | 最新失败为旧Compressor不接受带padding的state stride |
| 完整PTO CSA / P2数值 / 完整P3、P4 | 未完成 | adapter尚未实现；不能据probe通过推断完整计算通过 |
| 16卡整模型、真实DSpark、EP/EPLB、PD、性能 | NOT_RUN | 全部待新环境执行 |

关键证据均在 `tests/pypto_test/results/local_20260921/`：

- `native_layout_v5/`：真实权重及正确INT8 indexer布局。
- `native_groups_v5/`：BS4及拒绝用例；`native_groups_v4/`：其余BS。
- `native_trace_v2/`：六档BS各100步、五组metadata、负例和保护区。
- `native_dp_v1/`：两个真实DP rank，含每次同步值和原生dispatcher结果。
- `native_forward_custom_v3/`：最新完整原生forward失败。
- `logs/`：从原机器被git忽略的 `.cache/csa/*.log` 归档成可下载文本。

DP2用例为 `(4,40)、(40,4)、(8,24)、(16,32)、(0,4)、(0,40)`，再强制某rank为NONE。
实际执行 `NPUModelRunner._sync_metadata_across_dp`，不是mock collective，也未借dense模型绕过同步。
但没有PD connector；idle rank只验同步存活，未执行真正model runner dummy attention。
probe自行捕获的graph与完整模型graph是两层测试，不可混为一个通过项。

## 4. 最新失败：是哪一层“入参对不上”

这是本机原生算子环境与当前源码不配套的问题，发生在执行原生基线时，尚未进入完整PTO attention。

1. 当前仓库RMS custom接口有双scale/双输出，本机CANN9.2内置同名接口不同。
   原始扩展误落入内置接口，报 `yOut != nullptr`。
2. 隔离RMS ABI实验越过该点后，rotary也有custom/内置接口差异，多了 `negate_sin`。
   参数错位产生174764 GiB的虚假workspace需求，不是真实显存不足。
3. 初次编译同名custom op报5输入/4输入冲突，后来定位到增量构建的ops-info JSON过期：
   INI已更新，JSON仍只含CompressorMetadata。刷新JSON后，五个配套custom算子全部构建、安装成功。
4. 使用**原始C++扩展 + 新custom bundle**已越过前述RMS/rotary点。
   最新失败发生在原全局vendor的Compressor：要求state首维stride4096，实际原生页布局为8192。
   当前仓库 `csrc/attention/compressor/op_host/arch32/compressor_tiling.cpp` 已允许stride大于连续值，
   并把该stride传给kernel；本机全局旧vendor尚无此行为。

迁移时优先使用与当前基线匹配的完整原生扩展和custom算子包，至少重新核对 Compressor、SAS、QLIv2、
RMS、rotary、scatter、metadata。不要继续通过删参、把state改连续或缩短历史来制造“基线通过”。
本机只是成功构建了五算子包，**没有完成Compressor/SAS的新构建与完整forward验收**。

## 5. 已确认的接口细节

五个cache owner各用自己的metadata，不能拿SWA的一张表复用给所有cache：

| 对象 | 本机A3/block32实测物理布局 |
| --- | --- |
| SWA K/V | BF16 `[pages,32,1,512]`，每页32768字节 |
| compressed K/V | BF16 `[pages,32,1,512]`，每页32768字节；逻辑128原始token/页 |
| main compressor state | FP32 `[pages,2,1,2048]`，每页32768字节；有效内容只占一半 |
| indexer K + scale | INT8 K `[pages,32,1,128]` + FP16 scale `[pages,32,1,1]`，共享4160字节页 |
| indexer state | FP32 `[pages,2,1,512]`，每页4160字节；有64字节padding |

本机fixture在allocation前后各加128字节保护区，保留非零storage offset；这是测试设置，
不是说生产allocation必须有128字节前缀。

- `query_start_loc`为device INT32累计边界 `[B+1]`；`seq_lens`是device每请求KV总长度 `[B]`。
- CPU长度可能是比device多5的乐观值，不能替代异步接受纠正后的device长度。
- `positions`原生为INT64；slot在A3为INT32 `[N,2]` 页号/页内offset；PTO参考接口为INT64 flat slot。
- CompressorMetadata返回的是紧凑压缩输出slot，容量和token行数不同；尾部页号为-1。
  `expand_compressed_slots`已在device将有效closing token展开为逐token slot。
- 原生页表来自CPU allocator mirror，commit到device；PTO核实际读取device页表。
- Graph允许在图外更新固定地址的device buffer，需保留真实stream/event依赖。
- PyPTO当前Torch入口要求contiguous，并拒绝部分重叠的可写参数；共享K/scale应以一个底层载体传入。
- `indexer_kv_dtype=auto`在本机解析为BF16，并不自动采用checkpoint的INT8描述。
  本机明确设置 `attention_config.indexer_kv_dtype="int8"`；新机器必须重新确认。

## 6. 接手时避免重复的弯路

- 之前Qwen确实已跑过vLLM-Ascend。有效模型解释器是原工作区 `vllm-ascend/.venv/bin/python`；
  复用pto_eager工具链不等于必须换成pto_eager解释器。最终环境快照已记录实际import位置。
- 不要把最初 `environment_packages.json` 的triton3.2.0或manifest的import失败当作交接状态。
  最后使用的模型环境已是triton-ascend3.2.2。
- `csa-lib-ask.html`仅为同事提供的参考；[审阅记录](tests/pypto_test/CSA_LIB_ASK_REVIEW.md)
  已指出其“至少1ms”“图内必须转换”“运行190步等于正确”等结论证据不足，未复现其耗时。
  用户明确要求继续原任务，不再把审阅扩展成主线。
- 3.8是推进统计，不是attention每步只处理3.8行；target仍验6。
- 参考核的 `S=8` 与每请求验6的差异尚待真正适配；总token可被8整除不能证明请求划分正确。
- 不能把裸 `pl.at + pl.range` 当作多核dispatch。PTO原核使用SPMD/独立task表达多核，
  本机小核使用24个SPMD block；scalar写回刻意单task避免cache line竞争。

## 7. 新环境第一步

按[接续步骤](tests/pypto_test/handoff/NEXT_STEPS_16CARD.md)先做原生环境与真实checkpoint核对，
跑通单层Native ND，再实现attention-only PTO适配及完整P2/P3/P4，最后跑P5。
拥有16张卡不意味着此前未完成的正确性检查可以跳过。

可直接把下面这段交给新session：

> 请先阅读仓库根目录 DSV4_FLASH_CSA_HANDOFF.md 及其链接的接续计划、源码笔记和验证记录。
> 继续完成 DeepSeek-V4 Flash CSA attention.forward 接入PTO及真实16卡验收。
> 仅替换target compress_ratio=4 decode attention；排除外层HC/RMS及drafter；全流程ND，保留目标量化。
> 基线34bb51f9，vLLM84030bbe，pypto-lib205255b；不要带入原工作区私有CSA修改。
> 本机metadata/100步probe/真实DP2已过，但完整原生forward最新卡在旧Compressor stride支持，
> 完整PTO CSA adapter尚未实现。先核对新环境配套算子、目标checkpoint和导入来源，再按计划继续。
> TP1、DP=EP16、128K、出5验6、推进3.8、EPLB开、单卡BS4/8/16/24/32/40。
> 每个实际动作、失败和修正继续记入专用Markdown与独立结果目录，中文沟通。
