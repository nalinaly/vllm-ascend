> **2026-09-23 基线已迁至官方 v0.25.1rc1。** 当前入口为[基线迁移说明](BASELINE_MIGRATION_V0251RC1.md)。
> 下方旧阶段统计对应迁移前基线，历史脚本需按新接口适配。最新 release 结果见验证日志第93节：
> H255、每卡B4的真实PTO D16已跑通，8192个生成token与Native一致；尚无稳态性能结论，
> 不据此将旧P0～P5矩阵整体标为通过。其他P3/P4与剩余精度排查仍按用户要求暂停。
> 后续session从[剩余事项交接文档](DSV4_FLASH_CSA_NEXT_SESSION_HANDOFF.md)开始，
> 其中列明当前环境、已完成证据、可继续工作、暂停范围和执行入口。

# DeepSeek-V4 Flash CSA 接入 PyPTO 验证计划

- 日期：2026-09-21；状态更新：2026-09-23。
- 工作分支：`dsv4-flash-pto-v0.25.1rc1`；旧分支已删除。
- 文档状态：已迁到16卡、CANN9.0环境恢复工具链；历史背景见[移交入口](../../DSV4_FLASH_CSA_HANDOFF.md)。
- 当前资源：16个 Ascend910 逻辑设备；下文单/双卡标识表示阶段使用规模。
- 当前机器按单/双卡先完成P0～P4；前置验证、目标权重及资源就绪后在此机器执行16卡P5。
- 权重约束（2026-09-23用户明确）：后续诊断、回归和验收全部使用
  `/data/model/DeepSeek-V4-Flash-0731-w8a8`；旧参考权重结果只保留为历史记录。
- 对照格式：Native/PTO 全流程 ND；当前 PTO 不支持 NZ，本计划不包含 NZ 验证。
- 当前状态：CANN9.0运行时选定18项当前通过；第19项持久缓存仅有撤回补丁前的历史通过证据。
  参考ND/layout、六档100步metadata及DP2 metadata（含QLI长度断言）通过。
  TP1/S=6完整计算链已位于本仓库并对接Native物理存储。QR舍入、QA累加顺序与Top-K
  2048候选块的并列优先级对齐后，再修正Indexer compressor及head weights的K块顺序，
  reference_matrix_v5及去冗余后的adapter_redundancy_v1各18组回归全部通过，
  后者输出、Top-K和分数与前者逐bit相同；普通RoPE直接使用Native交错FP32频率。
  adapter_redundancy_v2删除独立token metadata调用，直接消费Native设备索引；
  v3再删除两次compact展开调用，消费者直接读取Native紧凑slot/cos/sin。
  v3只执行B4/B40完整对比及B4同图重放，共126项通过，两档输出/Top-K/分数与v2逐bit相同；
  v3 profiler为5次PTO提交，详见日志第65节。
  最新v4直接读写两组Native state，删除四次state适配和两份14行搬运窗口。
  B4/B40完整比较、B4图重放及B4 mixed10的eager/graph共496项通过；
  两档输出/Top-K/分数与v3逐bit相同，两份profile均为一次完整CSA提交，见日志第66节。
  该10步轨迹不代替旧100步复现或完整P3验收；Native metadata生产与外部事件等待保留。
  完整CSA的B4/B8/B40同图A→B→A通过；B40最新结果包含精确Top-K和整份存储逐bit对照。
  当前B4/B8 mixed100及B4三种接受边界各eager100+graph100通过，每项3700检查。
  对齐Native softmax概率的CAST_ROUND后，B16/24/32原mixed100各eager100+graph100数值通过；
  三个启动shell在Python完成后因运行中脚本编辑而exit2，数值与执行状态分开记录，见日志第71节。
  B40对齐Native SWA KV投影的K块遍历及单累加器后，原mixed100的eager100+graph100
  全部3700项通过、任务exit0，step46输出超差已消除；graph/eager全量指纹一致，见日志第72节。
  9月23日进一步对齐Query尺度乘法、主compressor投影/池化/RMS顺序，step46输出max_abs
  从0.0078125降至0.00390625。最新源码完成原100步容量下eager0～46及B4同图A→B→A，
  合计936项通过；主compressor state和压缩KV在47步中均与Native逐bit相同，见日志第74～75节。
  此次没有重跑完整100步，仍有容差内输出差异，不把此前100步证据外推到本次全部改动。
  step46剩余差异已分段：Query 6个BF16、SWA 7个独立历史元素（重复读取42次）；
  给定Native Query/KV后attention加逆RoPE仍有31个BF16差异。O投影隔离确认WO-B量化
  有53个INT8边界差异，另有反量化尺度顺序差异；两项小重放exit0，未改生产计算，见日志第76节。
  后续已修正WO-B直接计算127/amax及先权重后激活的反量化顺序：该段给定Native输入
  全量逐bit一致，step46输出不同元素降为41635、RMSE降为0.0001830331603，max_abs仍为
  0.00390625。原轨迹47步+B4图重放936项通过，仍未重跑最新完整100步，见日志第77节。
  用户已于9月23日通知正式W8A8下载完成，75分片结构检查通过；Native ModelSlim单CSA层
  24参数加载/ND布局及正式B4/B40、history131071完整CSA比较已通过，Top-K逐bit相同，
  输出max_abs均为0.00390625。正式P2为2/18，其余16组待验，见日志第78节。
  后续全部切换正式权重，并修正SWA RMS的逐列累加顺序；正式B40原100步容量的eager0～46
  及B4同图A→B→A共936项通过。六份cache/state在47步中数值完全相同，step46输出不同元素
  32707→5279、RMSE降至0.00005738172331，max_abs仍0.00390625、冻结超差0。
  剩余35个attention BF16差异待查；正式完整100步及完整P3仍未通过，见日志第79节。
  B4/B40三种Native生产stage延迟六组通过，最大延迟有submit返回时生产未完成证据。
  其余轨迹及G04～G07等覆盖待验，P3整体未通过，见日志第60～62节。
  服务forward派发已实现；完整动态尾块修复后正式B1启动capture/重放及
  B1→2→3→4→5→40→1同编译产物切换通过。正式完整矩阵及P5未完成，见日志第81～83节。
  PyPTO缓存识别补丁已撤回，持久缓存PASS仅为此前配置的历史证据，详见日志第34～35节。

### 2026-09-21 用户更新（优先于旧移交材料）

- **最终格式已确认恢复W8A8：** 核对
  `vllm-ascend-main/tests/dsv4_perf_accuracy_20260827`后，用户明确“就用W8A8的”。
  P/D脚本默认模型名含W8A8，均显式使用`--quantization ascend`。
- 最终目标为 **W8A8 checkpoint**。2026-09-22用户指定路径为
  `/data/model/DeepSeek-V4-Flash-0731-w8a8`；2026-09-23用户已通知下载完成，75分片结构检查通过。
  已到位的`quant_model_description.json`声明`W8A8_DYNAMIC`，包含DSpark专用模块；
  下载完成后仍须完成Native加载、正式权重数值及全模型验收，不能由分片结构检查代替。
  9月22日的等待通知限制已被上述下载完成通知解除。
  用户进一步确认：`/data/model/dsv4-flash-0731-dspark-w8a8`（48分片）由 **cann_recipe** 量化，
  只作参考，不能用它或46分片的`dsv4-flash-w8a8`替代上述目标权重。
  当前参考单层loader的scale命名/shape适配不等于目标量化格式转换或正式模型加载支持。
- 保留目标`quantization=ascend`与Native/PTO全流程ND要求，不再切换为纯BF16权重。
  Indexer INT8 cache属于独立接口约束，仍须核对目标checkpoint配置。
- 当前没有现成 prefill 服务。P5由本session准备配置并依次启动 P、D，验证真实PD链路。
- PyPTO、Simpler 强制使用 `feat/kernel-mode-integration-test` 调试分支。

## 1. 目标和验证边界

将 DeepSeek-V4 Flash **target model 的 CSA decode attention** 接入 PyPTO，验证真实
vLLM metadata、paged cache、异步长度更新和 ACL Graph 重放之间的接口是否正确。

替换点为 `DeepseekV4Attention.forward(positions, hidden_states, llama_4_scaling)`，仅对
`compress_ratio=4` 且满足已声明支持条件的 target decode 调用启用 PTO。

```text
原生 hc_pre
    ↓
原生 input_layernorm
    ↓
attention.forward                       ← 本次替换边界
    QKV projection / 内部 normalization / RoPE
    原始 KV 写入
    主 compressor / indexer compressor / indexer
    CSA sparse attention
    inverse RoPE / output projection
    ↓
原生 hc_post
    ↓
原生 MoE / EP / EPLB
```

接口输入、输出均为 `[T, 4096]` 的 hidden states；以目标配置实际 dtype 为准，首个目标为
BF16。输入已经经过外层归一化。以下内容不进入 PTO 替换范围：

- `hc_pre`、外层 `input_layernorm`、`hc_post`。
- MoE、EP 通信、EPLB 算法。
- DSpark drafter forward；target 验证与 drafter 的 mask、query 长度不能混用。
- Prefill、SWA-only、HCA 和其他未声明支持的 attention 路径。

Q/K normalization 和 compressor normalization 属于 attention 内部，必须保留。
最终接入还要保留原生 cache 生命周期、KV connector 等必要调用约定。

本计划需要分别回答：

1. 实际长度、请求边界和页表能否以 device Tensor 正确传入 PTO？
2. 页号、slot、stride、padding 和 cache group 是否正确对应？
3. 部分接受后，下一轮是否使用了正确的 lengths、positions 和 compressor state？
4. 同一个图重放时，PTO 是否读取新内容，且没有引入额外的 device→host 同步？
5. 单层结果和状态更新是否与原生实现一致？
6. 最终 DP=EP=16、EPLB 开启时，整模型是否仍兼容？

### 1.1 验证环境标识与本机范围

本文件统一使用以下标识；阶段内的全部用例继承该阶段的环境标识：

- **【本机·单卡】**：在当前环境中使用一张卡；包括无模型权重的 probe 和完整维度的单层验证。
- **【本机·双卡】**：使用两张卡，验证两个真实 rank 的 DP metadata 协调。
- **【仅16卡环境】**：当前机器的16卡整模型阶段，须具备目标权重、设备资源及P/D服务。

| 阶段/验证内容 | 环境标识 | 本机能完成的范围 | 需留到16卡验证的内容 |
| --- | --- | --- | --- |
| P0：接口、ND配置和环境核对 | 【本机·单卡】 | 静态核对使用CPU；加载/布局检查最多使用一张卡 | 目标16卡环境的版本、量化和实际通信配置复核 |
| P1：M01～M12 metadata probe | 【本机·单卡】 | Host/device传参、页表、slot、stride、padding、最小图重放 | 16个rank的真实运行数据与协调行为 |
| P2：18组单层数值对照 | 【本机·单卡】 | 全部六档BS及128K附近长度，Native/PTO均为ND | 全部目标CSA层与整模型共同执行 |
| P3：G01～G08连续状态与graph | 【本机·单卡】 | 注入接受轨迹、原生更新链、缓存状态和跨流同步 | 真实drafter/target共同运行时的自然接受分布 |
| P4：D01～D05双rank协调 | 【本机·双卡】 | `TP=1、DP=2`，两rank不均衡负载、padding和图选择 | `DP=EP=16`的通信、图容量及EPLB联动 |
| P5：整模型功能与性能验收 | 【仅16卡环境】 | 前期整理代码、配置及结果模板；条件就绪后单独启动 | 完整模型、真实DSpark、EP/EPLB、PD链路、端到端性能 |

**本机开发验证完成条件为P0～P4通过，并交付16卡待验清单。** P5未执行不影响记录本机阶段完成，
但整体验收状态仍为“16卡待验”。单层BS=40、128K测试属于本机范围，不因最终DP=16而推迟。
本机单卡阶段按单rank验证，P4按两个真实rank验证；不在两张卡上启动或模拟16个rank作为P5结果。
EP=16及EPLB实际重平衡只计入P5通过项。

## 2. 基线、参考代码和环境记录

| 项目 | 固定基线或处理原则 |
| --- | --- |
| vLLM-Ascend | `34bb51f93724c565362f5108f5226303e1b56cad`，官方上游代码 |
| 新分支 | `nalinaly/vllm-ascend:dsv4-flash-pto`，由上述提交创建 |
| Qwen3 参考 | `093b1eb519f2a706b8b471d40cc4c66c64e68f1b`；只参考接入结构，不整批移植其修改 |
| pypto-lib | `205255b4770ee84dfa176bcbc7bbef651953c7e1` |
| 计算参考 | `models/deepseek_v4_flash_dspark/decode_csa.py::_decode_csa_tp1` |
| 参数参考 | 原工作区 `vllm-ascend/vllm_ascend/test_script/runtime/` 下的 decode 脚本 |
| 原工作区私有实现 | 不作为新分支的代码基线；不复制既有私有 CSA adapter |

参数参考脚本目前不属于新分支。后续实现应使用本文件冻结的参数创建独立验证入口，不依赖原工作区的
绝对路径、服务地址和未提交文件。

每次验证记录以下环境信息，不能只记录一个仓库 SHA：

- vLLM、vLLM-Ascend、pypto-lib、PyPTO、Simpler 的提交和实际 import 路径。
- PTOAS、PTO ISA、PyTorch、torch_npu、CANN、驱动版本。
- NPU 型号、可用显存、device ID、执行平台、PTO runtime。
- 原生和 PTO 的最终有效配置、图模式、ND weight/cache 的实际布局及量化配置。
- checkpoint 的标识和量化描述；合成权重则记录生成规则和随机种子。

Qwen3 示例提供了实例级 forward 替换、Torch custom op、fake implementation、预编译 warmup
和 graph 验证的参考。本次明确采用 ND；它的仅 BF16、无 speculative decoding、无 padding 等
其他约束不能直接用于 DSV4。仍须保留目标量化和图模式的验证要求。

### 2.1 执行约束补充（2026-09-22）

2026-09-22后续要求：最终PTO CSA为独立算子。用户要求先消除可确认的冗余适配，
剩余适配再讨论处理方式；随后已明确要求直接读写Native state并删除四次适配和两份14行窗口。
已去掉普通RoPE往返列重排、未使用的`window_swa_lens`及独立token metadata调用；
positions、普通KV页表、三组slot直接使用Native device Tensor，在设备端计算地址。
用户进一步指定先删除两次compact metadata展开，现已由消费者直接读取各自Native紧凑行；
仅保留两份每请求行偏移，由已有CSA任务在设备端计算。最新v4的两组pool按各自Native页表
读取state，原write任务在pool依赖满足后按Native slot直接写回；两份state均为InOut，
保留真实页间距、padding和原storage owner，四次适配与两份搬运窗口已删除。
实现与验证边界见日志第63～66节。若后续发现无法合理直接读写的情形，先与用户讨论。
用户要求控制测试范围：v4仅B4/B40各一次完整对比、B4一次图重放及B4 mixed10 eager/graph，
不重跑18组/100步矩阵。两份profile均确认一次完整CSA提交，单次Python封装不替代该验收。

用户明确要求：计算依据仍为指定的`decode_csa_tp1`；接入时参考已接入的
`vllm-ascend-main`与`vllm-ascend-qwen3-14b-pto`，**参考不等于照抄**。
以下对照用于发现接口约束与遗漏，不整批复制旧adapter或重新实现一套QKV算法。

**代码归属：** CSA计算与接入适配只写在`vllm-ascend-dsv4-pto`。
`pypto-lib`为只读计算参考，不在那里提取/改造入口。实际计算目录为
`vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/`，保留固定参考提交、函数来源和适配差异。
PyPTO/Simpler属于依赖工具链，不作为CSA接入的改造对象；当前保留的环境兼容差异须单列说明，
后续出现依赖问题先记录复现和必要性，不自行扩展为编译器维护工作。

| 接入问题 | 对照位置 | 本次必须重新核对的差异 |
| --- | --- | --- |
| attention实例替换、原生上下文取得 | Qwen `models/pypto_qwen3_attention.py` | DSV4签名含`llama_4_scaling`，仅C4 target decode；保留原生cache/connector约定 |
| 权重准备和ND检查 | main `_pypto_dsv4_csa/weights.py`；Qwen `_prepare_attention` | main的逐层量化拓扑与当前参考权重不同；最终以目标checkpoint加载结果为准 |
| cache物理存储和生命周期 | main `_pypto_dsv4_csa/adapter.py`、`physical_storage.py` | 五组metadata、六个逻辑cache，state padding，Indexer K/FP16 scale共享storage |
| custom op、fake、预编译 | Qwen `ops/qwen3_decode_attention.py`及`process_weights_after_loading` | 本次需声明全部实际写入，覆盖六档BS/graph bucket及当前debug runtime |
| 派发、graph和流 | main `_pypto_dsv4_csa/dispatch.py`、`backend.py`及已有graph测试 | 不沿用旧静态S=8、固定两条压缩写入或旧L1 ABI；验证本次S=6与device长度更新 |

实施顺序：

1. 在本仓库依据固定参考TP1入口提取完整attention，保留原子核调用、内部norm与任务依赖；
   外层HC/norm保留原生vLLM行为，参考仓库保持不变。
2. 显式TP1/S=6配置，核对尾tile、请求边界、压缩闭合位置及RoPE；服务导入不改`sys.argv`。
3. 按真实Native布局接入metadata/cache。原始与压缩KV直接使用原生页；
   compressor state按最新要求直接读写Native页，保留实际页间距与pool→write依赖；
   初版`8+6`行搬运窗口已在v4删除；
   Indexer直接适配共享K/FP16 scale页，不建立完整历史cache副本。
4. 初始化阶段准备ND权重、编译与warmup；按本次接口实现forward/custom op及流依赖。
5. 先做完整B4/S6/131072单层Native/PTO对照，再执行P2的18组矩阵；
   只有完整链路失败时，才对参考链中的对应阶段增加诊断。
6. 按P3、P4、P5依次验收；目标vllm_ascend W8A8权重到位后重验正式加载及目标数值，
   P5自行准备P/D配置并先启P后启D。

每项记录实际修改、适配原因、验证范围与原始结果；CPU lowering、metadata probe、
参考权重结果均不得替代完整PTO或目标模型验收。

## 3. 最终目标参数

### 3.1 用户给定的参数

| 参数 | 目标 |
| --- | --- |
| 执行阶段 | Decode |
| TP | 1 |
| DP | 16【仅16卡环境】；本机P1～P3为单rank，P4为DP=2 |
| EP | 16【仅16卡环境】；实际 EP group size 必须从运行环境核对 |
| GBS 基准 | `16 × 4 = 64`【仅16卡环境】；本机保留单rank BS扫描 |
| 单卡 BS 扫描 | `4, 8, 16, 24, 32, 40` |
| 历史上下文 | 128K；本计划暂按 `131072` 个 token 的初始历史长度处理 |
| DSpark | 出 5、验 6 |
| 平均接收步长 | 3.8；本机注入确定轨迹，真实接受统计在16卡环境验证 |
| EPLB | 开启【仅16卡环境】；本机不验证真实重平衡 |
| 对照格式 | Native/PTO 全流程 ND；适用于本机和16卡全部阶段 |

### 3.2 继承脚本的条件与必要调整

| 配置项 | 计划值或约束 | 说明 |
| --- | --- | --- |
| `max_model_len` | 1048576 | 128K 是测试历史长度，需为继续 decode 留空间 |
| `max_num_seqs` | 至少 40 | 原脚本为 32，不能覆盖 BS=40 |
| `max_num_batched_tokens` | 首轮保留 400 | 最大实际验证 token 数为 240；仍需检查图 padding 和各 buffer 的容量 |
| `block_size` | 32 | 区分 scheduler 逻辑页、压缩页和 state 页的单位 |
| `async_scheduling` | 开启 | 必须覆盖异步接受结果对 lengths 的修正 |
| Prefix caching | 开启 | 覆盖共享历史页、释放和重新分配 |
| Hybrid KV cache manager | 开启 | 使用原生分组与分配布局 |
| Speculative config | `method=dspark, num_speculative_tokens=5` | Target 的完整验证 query 为 6 |
| Drafter `enforce_eager` | 保留脚本的 `true` | 不等于 target 关闭 ACL Graph |
| Target graph | `FULL_DECODE_ONLY` | Eager 只作为分阶段调试与对照 |
| Ascend graph options | `enable_npugraph_ex=true, enable_static_kernel=true` | 最终记录实际生效值 |
| 量化 | W8A8，`quantization=ascend` | 每个模块的dtype、scale、skip list必须实查 |
| Weight layout | Native/PTO 均为 ND；`additional_config.weight_nz_mode=0` | 原脚本 `VLLM_ASCEND_ENABLE_NZ=2` 不作为本次目标；须检查加载后的真实格式 |
| Indexer cache dtype | 当前A3基线显式设 `attention_config.indexer_kv_dtype=int8` | 本机实查 `auto` 得到BF16 K，未自动采用checkpoint的INT8描述；A3原生路径写INT8 K/FP16 scale。16卡须复核实际生效值 |
| Recompute scheduler | 保留脚本的 `false` | DP 同步是否跳过还取决于通信方式等条件 |
| DSA-CP / O-proj 跨 DP TP | 本计划按未启用处理 | 若最终环境启用，需要另列通信接口验证 |

ND是本计划固定约束，适用于加载完成、warmup、eager、capture/replay和最终16卡对照。
当前基线通过 `additional_config.weight_nz_mode` 控制通用权重NZ转换；Native/PTO均须在模型加载前
将该值设为0，并合并到已有 `additional_config`，保留其中其他配置。
旧脚本中的 `VLLM_ASCEND_ENABLE_NZ=2` 应移除；若执行旧版本且仍使用该环境变量，应明确设为0，
但不能仅凭环境变量认定当前基线已关闭NZ。

加载后和warmup后检查实际格式、dtype、shape和stride；排查量化模块独立的格式转换。
权重采用用户最终确认的W8A8目标；ND不等于取消量化或所有Tensor连续，仍须保留原生paged cache
的页padding和stride，A3 DSA cache按 `PA_ND` 验证。
若目标量化模块没有可用的ND路径，记录具体模块和阻塞原因；不能静默改成NZ或全BF16后标记目标通过。
所有性能结论均为Native ND与PTO ND的对比，不将原脚本NZ结果作为本次同配置基线。

容量分为三个量，日志和接口不能混用：

- `B_actual`：实际请求数。
- `T_actual`：实际验证 token 数，均匀完整验证时为 `6 × B_actual`。
- `T_padded` / `B_capacity`：图和分配器使用的容量，可能大于实际值。

图容量必须取自原生 dispatcher 的选择结果；不能预先假设 `T_padded=240` 或一定为 256。
存在原生 dummy request 时，metadata buffer 还必须覆盖相应请求槽位。

### 3.3 正式 batch 矩阵

| 单卡 BS | `T_actual=6×BS` | DP16 均衡时 GBS |
| ---: | ---: | ---: |
| 4 | 24 | 64 |
| 8 | 48 | 128 |
| 16 | 96 | 256 |
| 24 | 144 | 384 |
| 32 | 192 | 512 |
| 40 | 240 | 640 |

前两列是【本机·单卡】P1～P3必须覆盖的本地规模；第三列仅用于【仅16卡环境】P5。
本机不需要构造第三列对应的全局并发；P4的实际总请求数取两个rank的本地BS之和。

本计划暂将 GBS=64 解释为 BS=4 的基准点，其余为独立扫描点。
如果 GBS 始终固定为 64，则较大的单卡 BS 属于不均衡 rank 分配，不能将上表最后一列用作目标。

所有正式测试点的 `T_actual` 都能被 8 整除，可能掩盖错误的 8 行处理假设；请求划分仍须按验6处理。
BS=4 的 `T_actual=24` 还要覆盖 16 行计算 tile 的尾部。

## 4. Metadata 和 cache 接口合同

### 4.1 Host/device 分工

| 字段 | 原生来源与含义 | PTO 侧要求 |
| --- | --- | --- |
| `query_start_loc` | NPU，通常为 INT32 累计请求边界 `[B+1]` | 直接接收 device Tensor；不是每请求长度数组 |
| `query_start_loc_cpu` | CPU 副本，用于调度及部分 metadata 构建 | 不代替 device 上的实际边界 |
| `seq_lens` | NPU，每请求本轮 KV 总长度 `[B]` | 与 query 边界共同决定可见范围 |
| `actual_seq_lengths_query` | 原生 CSA 中绑定到 `query_start_loc` | 保持累计边界语义 |
| `actual_seq_lengths_key` | 原生 CSA 中绑定到 `seq_lens` | 保持每请求长度语义，不做累计和 |
| `_seq_lens_cpu` 等 | CPU 上的乐观或已修正副本，具体取决于执行阶段 | 可以用于既有容量逻辑；不将其内容缓存为 PTO 的真实长度 |
| `block_table` | NPU INT32，按 cache group 提供的逻辑页到物理页映射 | 使用对应 group、实际行 stride 和有效范围 |
| `positions` | NPU；runner 路径使用 INT64 positions | 如入口需 INT32，在 device 上转换并验证范围 |
| `start_pos` | NPU，通常为 `seq_lens - diff(query_start_loc)` | 不在 host 上另算一份可能不同步的值 |
| Runner slot mapping | Device 上的 flat slot，通常 INT64 | 与 DSA backend 格式区别记录 |
| A3 DSA slot mapping | `INT32[T,2]`，`[block_id, offset]` | 需要时在 device 上转为 PTO 的 flat INT64 slot |
| `num_actual_tokens` / `num_actual_reqs` | Host 计数，原生用于切片、构图等 | 不能将首次 capture 的值固化为所有 replay 的有效性 |
| `num_compressed_tokens` | Metadata 输出/分配相关计数 | 不默认等于本轮实际产生的有效压缩条目数 |
| Shape、dtype、stride、page size | Host 可读取的 Tensor 描述信息 | 可用于分配和选择实现；与 Tensor 内容读取区别处理 |

示例：两条请求各验6，`query_start_loc=[0,6,12]`；若历史长度分别为 `P0/P1`，本轮
`seq_lens=[P0+6,P1+6]`。不能用全局 GBS、平均步长3.8或 Tensor 的 padded 行数替代这些值。

热路径不得为方便调度而新增对 NPU Tensor 的 `.item()`、`.cpu()`、`.tolist()` 或全局同步。
测试结束后的同步、结果拷回与断言允许存在，但必须在被测路径和性能计时之外。
不要求移除原生实现已有的 host 计算或事件同步；要求适配器不新增不必要的往返。

### 4.2 五组 metadata 与六个可变 cache

五组 metadata 应按原生对象的 prefix/属性查询，不依赖字典排序或固定枚举顺序：

1. 原始 SWA cache。
2. 主 compressor 的 compressed KV。
3. 主 compressor state。
4. Indexer compressed K/scale。
5. Indexer compressor state。

| 可变 Tensor | 参考用途 | A3 验证重点 |
| --- | --- | --- |
| SWA KV | 原始 KV 写入和窗口读取 | Flat slot 与 block/offset、窗口边界、无效行 |
| Compressed KV | 主压缩结果 | 压缩比4、逻辑页单位、新增页与有效条目 |
| Main compressor state | 主压缩历史状态 | 页 padding、请求归属、部分接受后的使用 |
| Indexer compressor state | Indexer 压缩历史状态 | 独立的 block table、真实物理 stride |
| Indexer K | INT8 历史索引 key | 量化规则、页映射、有效历史长度 |
| Indexer scale | A3 原生 FP16，参考 PTO 接口 FP32 | 不能直接把同一地址按 FP32 解释 |

对每个 Tensor 记录 `shape/dtype/device/stride/storage_offset/storage_bytes`、view 所属
storage、逻辑页大小和物理页步长。六个 cache 的 mutation 必须在自定义 op 与 PTO 接口中正确表达。

对测试副本也要保留原生物理布局与 alias 关系。直接 `clone()` 一个非连续逻辑 view 可能改变布局；
应按原生分配规则创建独立 storage，再初始化相同内容。禁止让 Native 与 PTO 共用同一份可变 cache。

### 4.3 参考 kernel 的适配清单

- 从 `decode_csa_tp1` 提取 attention 计算，移除 HC 参数、外层 norm 参数和 HC 输出形式。
- 统一 target 的验6约定，排查所有 `query // S`、`T=B*S`、请求 reshape 和 state 容量公式。
- 处理尾 tile；不能仅把 `DSPARK_SPEC_TOKENS` 从7改为5。
- 单独验证 query RoPE 与 compressor RoPE 的来源、位置和布局。
- 对接上述五组 metadata，而非把一个 block table 同时用于所有 cache。
- 对齐 slot 表示、indexer scale dtype、ND权重的实际布局和每个 cache 的实际 stride。
- Native/PTO加载后的权重和接口Tensor均使用ND，热路径不引入ND/NZ往返转换。
- 参考实现有显式 state slot mapping；若原生没有同名字段，应在 device 上按原生状态规则生成并验证。
- 构图前完成权重整理、编译和所需 bucket 的 warmup，不在 replay 中选择源码目录或重新编译。
- 初版可先支持均匀验6；对不等长 query、其他 TP、prefill 等情况必须明确回退或报错。
- 正式 CSA 用例要求实际执行 PTO；静默回退原生不能记为通过。

原参考入口没有独立的 `pl.Scalar` host 参数，不意味着所有 Tensor 内容都能在 Python 中读取。
若新接口需要 page stride 等标量，应逐项说明其来源、是否可变及 graph 更新方式。

## 5. Fixture、历史 cache 和接受轨迹

### 5.1 两类 fixture【本机可开发验证】

**Metadata fixture：** 不加载模型权重，使用原生 `BlockTable`、metadata builder、ForwardContext
和小型带标记值的 cache。PTO probe 输出请求归属、长度、逻辑/物理页、slot、有效掩码和读取结果。
长位置地址测试只访问 fixture 中已合法映射的页，不把小 cache 的通过结果作为128K数值验证。

**单层数值 fixture：** 保留真实 hidden/head/indexer/压缩维度，构造合法的128K历史 cache、
compressor state、ND权重与输入。先用可复现的合成数据，随后核对目标量化配置或目标层权重。
不实例化完整43层模型、MoE或 drafter；无需执行完整128K prompt prefill。

两类 fixture 均使用原生 metadata 生产逻辑。单纯手填一个字段齐全的 Python 对象只能作为局部单测，
不能替代实际 builder、原生长度修正和 device metadata executor 的验证。

### 5.2 128K 数据规模与边界【本机可开发验证】

以初始历史长度 `L=131072`、压缩比4、压缩页大小32计算：

- 每请求压缩条目数：32768。
- 每请求已满的压缩页数：1024；继续 decode 必须预留新增页。
- Indexer Top-K 候选分段为8192条，128K附近会跨过四段到五段的边界。
- 长度容量保留1M；数值测试实际只初始化目标历史和后续轨迹需要的有效 cache。

以下为单层主要数据的估算，不含 native 页 padding、权重、SWA/state、workspace、graph 和对照副本：

| 单卡 BS | BF16 compressed KV：`B×32768×512×2` | INT8 indexer K：`B×32768×128` |
| ---: | ---: | ---: |
| 4 | 128 MiB | 16 MiB |
| 8 | 256 MiB | 32 MiB |
| 16 | 512 MiB | 64 MiB |
| 24 | 768 MiB | 96 MiB |
| 32 | 1024 MiB | 128 MiB |
| 40 | 1280 MiB | 160 MiB |

每个 BS 实际运行前记录峰值显存估计及可用余量。若内存不足，优先顺序运行 Native/PTO、复用静态权重和
按 case 释放 fixture；不能无记录地缩短正式用例的128K历史。

### 5.3 接收步长3.8【本机注入轨迹；真实接受统计仅16卡】

本计划暂按 `valid_count = 1 + accepted_drafts`，平均 `valid_count=3.8` 处理。
若“3.8”仅指接受的 draft 数，则实际推进均值为4.8，需更新轨迹和性能口径。

主轨迹为 `[1,6,2,5,5]`，不同请求循环错位使用；每请求执行100步时平均推进恰好为3.8。
另设恒为1、恒为6，以及集中长串拒绝/接受的边界轨迹。相同均值不能替代这些不同状态序列。

令 `P[r,s]` 为请求 r 在第 s 轮的 query 起点，`a[r,s]` 为本轮有效推进量：

```text
query_len[r,s] = 6
positions[r,s] = P[r,s] + [0,1,2,3,4,5]
seq_lens[r,s] = P[r,s] + 6
P[r,s+1] = P[r,s] + a[r,s]
rejected[r,s] = 6 - a[r,s]
```

使用原生接受计数更新函数推进状态，并包含请求换位时的 `prev_positions` 映射。
不能每步独立重建一组“看起来正确”的 metadata，从而绕过被测的接受结果处理。

允许原生预先写入尚未接受的 speculative 数据；检查重点是它们不能错误进入下一轮可见范围，且需要时
被正确覆盖。不能凭“所有拒绝位置必须回滚为零”定义与原生不同的状态语义。

合成接受轨迹验证的是接口和状态机，不能证明真实模型的自然接收步长达到3.8。

## 6. 分阶段执行计划

### P0【本机·单卡】：冻结接口和环境，不运行完整模型

**设备：** 静态核对使用CPU，权重/布局检查最多使用一张A3；无需完整模型。

**输入：** 本文件目标参数、固定源码基线、运行环境和可用量化描述。

**工作：**

1. 完成环境 manifest，核对实际 import 路径，避免混用旧工作区模块。
2. 记录 target 与 drafter 的计划配置及本机可核对的有效配置；确认目标层 `compress_ratio=4`，
   Native/PTO的 `weight_nz_mode=0`，加载及warmup后的实际权重为ND。
3. 用原生规则推导 cache spec、分配布局和图容量，不根据 Tensor 名称猜测 dtype/stride。
4. 检查原生/PTO 函数签名、所有 cache 的 mutation 和支持范围。
5. 冻结浮点、量化 cache 和 Top-K 比较规则；尚未明确的项目列为 `PENDING`。
6. 列明128K、GBS、接收步长口径和最终16卡待验项目。

**通过条件：** 能列出每项输入的来源、含义、设备、布局和更新时机；没有未解释的类型转换或 host 同步。
P0 是静态核对通过，不等于运行验证通过。

### P1【本机·单卡】：原生 metadata → PTO probe

**设备：** 一张 A3；不加载 attention 权重。

通过计划中的 Torch custom op 和 PTO runtime 调用小核，覆盖真实参数封装与 Tensor 传递。
Probe 输出离散值，和独立 CPU 公式及原生 helper 的结果交叉核对。

| ID | 用例 | 必查结果 |
| --- | --- | --- |
| M01 | 六档 BS，各请求验6 | 累计 query 边界、请求归属和有效 token 数准确 |
| M02 | 请求历史长度各不相同 | 长度不串行，不使用全 batch 的同一个 seq_len |
| M03 | `128K−1 / 128K / 128K+1` | 页边界、压缩位置、地址计算正确 |
| M04 | 不连续物理页，五组页表各不相同 | 不混用逻辑页、物理页或 cache group |
| M05 | 请求换位、删除、新增 | State 跟随请求和页映射，不固定绑定 batch 行号 |
| M06 | `T_actual < T_padded`，slot 为无效值 | 无效行不写入真实 cache，输出处理符合约定 |
| M07 | 正确形状但有页 padding / 非紧凑 stride | 按真实物理页步长寻址 |
| M08 | INT32 block/offset → INT64 flat slot | 有效值正确；无效值保持无效，不钳制为0 |
| M09 | 异步模式中 host 乐观长度与 device 实际长度不同 | PTO 使用 device 实际值；host 容量上界仍合法 |
| M10 | 同地址 metadata 内容 A→B→A | 输出按内容变化，不复用首次读取的值 |
| M11 | 不等长 query / 混合 prefill 等不支持输入 | 明确回退或拒绝，不能错按验6执行 |
| M12 | 故意混错 group 或破坏映射的负例 | 测试能发现错误；越界负例在发射 kernel 前拦截 |

M09 不随意破坏原生 builder 必需的 CPU 容量信息；先经原生流程生成合法 metadata，再测试适配器应读取的
device 值。真正的长度修正链在P3验证。

Probe 先跑 eager，再做不依赖 attention 权重的最小 capture/replay，验证同地址内容 A→B→A。
该结果用于尽早确认参数传递和内容更新，不能替代P3的完整 CSA 图重放验证。

**通过条件：** 所有合法用例离散输出逐元素一致；无效行无误写；负例被正确识别；真实 PTO 路径执行可证实。

### P2【本机·单卡】：128K 单层 Native/PTO ND eager 数值对照

**设备：** 一张 A3 顺序执行；第二张卡不是必需条件。

**正式矩阵：** 六档 BS × 三个初始历史长度 `{131071,131072,131073}`，共18组。
首轮固定种子1024；通过后仅对 BS=4/40 的关键边界补充种子0和1。

每个 case：

1. 初始化相同的 hidden states、ND权重、metadata 和两份独立ND cache/storage。
2. 调用原生 attention forward，保存输出、必要中间值和 cache 写入结果。
3. 从相同初态调用 PTO attention forward。
4. 比较最终输出、六个 cache/state、写入位置和未写区域。
5. 失败时从 metadata、QKV/RoPE、compressor、indexer、sparse attention、O-proj 逐段定位。

至少包含：全部候选超过Top-K上限的真实长序列路径、query间可见压缩条目数不同、全局页号打乱、
跨新压缩页和 state 页更新。不得只分配512个Top-K结果对应的历史 cache 来代替32768个候选的验证。

权重测试分为可复现合成数据与目标W8A8 checkpoint两档，均采用ND。
合成权重或现有W8A8参考层通过，不能代替最终目标checkpoint验收。
只有BF16/ND对照通过时，不能将目标`quantization=ascend`的ND路径标为通过。
NZ不属于本计划的测试项或通过条件。

**通过条件：** 18组正式 case 均满足第7节规则；无静默 fallback；最终数据格式符合目标合同。

### P3【本机·单卡】：接受轨迹、连续状态和 ACL Graph 重放

**设备：** 一张 A3。先 eager 多步，再使用相同轨迹做 graph 对照。

| ID | 用例 | 规模与要求 |
| --- | --- | --- |
| G01 | 主接受轨迹 | 每个正式 BS 连续100步，平均推进3.8，跨步保留 cache/state |
| G02 | 接受边界 | 全为1、全为6、长串拒绝后接受，覆盖压缩窗口状态 |
| G03 | 同一个图内容更新 | 地址和容量不变，原地改变长度、positions、页表和slot |
| G04 | Batch 变化 | 六档 BS 之间切换；若原生选择不同 bucket，记录图重选 |
| G05 | 实际数据少于图容量 | 小 BS 放在较大的合法 bucket，检查 padding 和 dummy request |
| G06 | 请求生命周期 | 请求换位、退出、新增及合法页复用，覆盖 `prev_positions` 映射 |
| G07 | Prefix 共享 | 历史页只读共享；分歧写入按原生 copy-on-write/分配规则处理 |
| G08 | Metadata 跨流生产 | 使用原生 executor 和 external event；受控改变生产耗时以暴露缺失等待 |

执行约束：

- Warmup 和 capture 会写 cache；正式比较前按相同地址恢复全部初态，并重建相应 metadata 状态。
- Replay 前在正确的 stream/event 关系下更新输入内容，不重新绑定一个 graph 未捕获的新 Tensor。
- 捕获后不能通过每步 `.item()` 获取真实长度来决定 kernel 的数据相关工作。
- `DeviceMetadataExecutor` 的必要等待必须保留；使用其预计算 compressor 结果时尤其要核对。
- 结果比对所需同步放在计时之外；不能加入全局同步来掩盖跨流数据竞争。
- 单独记录实际 PTO kernel 执行证据；Python forward 的调用计数不能代表 replay 的执行次数。
- 保留目标 `FULL_DECODE_ONLY`、npugraph_ex 和 static-kernel 条件；降低图模式的结果只记为调试结果。
- Native/PTO保持ND，capture与replay中均不重新引入NZ权重或ND/NZ转换。

**通过条件：** Eager 与 graph 的每步输出及状态均满足合同；第100步仍一致；地址变化符合声明；
无额外 D2H 值读取、无 replay 重新编译、无因 padding 引发的误写。

### P4【本机·双卡】：双卡 DP metadata 协调

**设备：** 两张 A3，`TP=1、DP=2` 的最小验证进程组。两张卡不声明为16个 rank。

使用单层/metadata fixture，两个rank均为ND；不加载完整DeepSeek模型，不验证EP=16或EPLB重平衡。

| ID | 两个 rank 的本地 BS | 目的 |
| --- | --- | --- |
| D01 | `(4,40)` | 低负载 rank 的 padding 和图容量 |
| D02 | `(40,4)` | 交换负载后不存在 rank 固定假设 |
| D03 | `(8,24)`、`(16,32)` | 中间 bucket 协调 |
| D04 | `(0,4)`、`(0,40)` | 原生允许的空 rank/dummy 路径，无效行和同步存活 |
| D05 | 上述负载连续切换 | 图重选、metadata buffer 复用及不同接受轨迹 |

先记录 `should_skip_allreduce_across_dp_group` 的实际结果、通信方法和图模式，再确定预期 padding。
未跳过同步时，验证共同容量和本地有效行；跳过时，验证各 rank 保持本地有效数据。

最小测试模型不能因被归类为 dense 而绕过生产 MoE 所需的 DP 分支，却报告该分支已通过。
DP2 与 DP16 的通信选择可能不同，必须分别记录。P4验证的是协调机制，不是16卡性能或EPLB效果。

**通过条件：** 两个 rank 的数据不串用；padding不误写；图选择符合实际分支；空rank按原生约定工作；
Native/PTO 的本地计算结果在相同全局协调条件下保持一致。

### P5【仅16卡环境】：最终16卡整模型验收

**前提：** 有完整模型、目标W8A8 checkpoint、16卡资源；P1～P4已通过。
当前没有现成P服务，由本session准备并先启动P、再启动D；核对P/D设备分配及KV传输配置。

当前机器具备16张逻辑设备；前置验证、目标权重和P/D资源就绪后在本机启动该阶段。
先复核P0中的有效配置、ND支持和布局，再执行整模型验收。

保持 `TP=1、DP=EP=16、DSpark出5验6、EPLB开启`，Native/PTO全流程ND，
按已确认的 GBS/BS 口径执行最终矩阵。
核对每个 rank 的实际请求数，不能仅从客户端 concurrency 推断服务器 batch。

以下全部属于【仅16卡环境】，执行前结果表保持 `NOT_RUN`：

| ID | 验收内容 | 必查结果 |
| --- | --- | --- |
| F01 | 全模型ND基线与CSA接入 | 原生/PTO使用相同输入、量化、初态；所有目标CSA层接入，其余路径保持原生 |
| F02 | 16rank实际BS、GBS与graph | 先验GBS=64/单卡BS=4，再覆盖确认后的其余矩阵；复核实际通信分支和图容量 |
| F03 | PD cache转移与层级通知 | 完整链路正确；本机直接初始化cache的结果不能替代 |
| F04 | EP=16与EPLB | 专家通信、初始化及运行中重平衡与图执行、请求生命周期共同工作 |
| F05 | 真实DSpark出5验6 | 记录自然接受分布、有效推进量和模型输出，不强制注入合成接受轨迹 |
| F06 | 整模型稳定性与性能 | ND对照下的峰值显存、错误、超时、step latency和实际输出吞吐 |

P5未执行时，结论最多为“单层 CSA 接入与已覆盖的 metadata/graph 场景验证通过”，不能写成
“DP16/EP16/EPLB 场景整模型验证通过”。

## 7. 正确性判定与故障定位

| 对象 | 判定原则 |
| --- | --- |
| 请求边界、长度、页号、slot、有效掩码 | 有效位置逐元素完全一致 |
| 写入范围、padding、无关 cache 页 | 与原生行为一致；应保持不变的区域逐字节不变 |
| 输出与浮点 cache/state | 按冻结的逐 Tensor 容差比较；检查最大误差、RMSE、超差数量和位置 |
| INT8 indexer K | 先按同一量化规则要求一致；差异需单独定位量化/舍入，不能用浮点容差掩盖寻址错误 |
| Indexer Top-K | 检查范围、有效数量、分数与选择；分数并列按明确的 tie 规则处理 |
| 跨步状态 | 每步比较；不能只比较最后一帧或仅比较输出 |
| Graph replay | 与相同输入和状态的 eager/native 对照，使用相同判定标准 |

BF16输出的初始参考门槛为 `atol=1e-2、rtol=1e-2`，与参考 fixture 的量级一致。
这不是所有中间 Tensor 的通用门槛。P0/P2须结合目标量化路径，为 FP32 state、其他浮点 Tensor 和
量化 cache 明确独立标准，冻结在结果 manifest 中。未冻结的项目保持 `PENDING`，不能算通过。

近似误差、Top-K并列和量化阈值问题必须用可复现案例解释；不能在看到失败后自动放宽容差。
Top-K 顺序或选项差异只有在满足预先定义的等价条件且后续输出/状态也通过时才可接受。

失败按以下顺序定位，避免一开始调整数值容差：

1. 是否实际执行 PTO、是否传入目标层的 metadata。
2. `B_actual/T_actual/T_padded`、dtype、device、stride 和 view/alias 是否一致。
3. Query、KV长度、slot 与五组 block table 是否使用相同语义。
4. Stream/event 是否确保 metadata 已就绪，是否读到上一轮内容。
5. 权重量化、ND物理布局、RoPE、compressed state 更新是否一致。
6. 最后分析具体计算阶段的浮点和量化误差。

## 8. 性能与同步检查

正确性通过后再测性能，Native/PTO均为ND，使用相同输入规模、量化格式、cache布局与轨迹。
建议先5轮warmup、100轮计时；编译和graph捕获只进行必要次数，不为收集样本反复重启进程。

分别记录：

- 【本机·单卡】PTO kernel 的设备耗时。
- 【本机·单卡】attention wrapper 的耗时，包括其新增的 metadata 转换。
- 【本机·单卡】单层完整调用的耗时，并说明异步 metadata 生产是否包含在计时范围。
- 【本机·双卡】P4两个rank的metadata协调开销、各rank调用耗时与等待情况。
- 【仅16卡环境】最终整模型的 step latency、实际接受/输出 token 数及吞吐。

报告 median/p95、样本数、有效序列长度范围、每rank实际/填充token数及所选图容量。
核心忙碌时间、CPU提交时间、设备wall time和整模型step time不能互相替代。

性能计时不包括编译、fixture初始化、golden计算、结果拷回和逐元素断言。
DFX/swimlane 与无额外同步的正式计时分开运行。Profiler用于确认是否新增D2H、全局同步、重复编译或
未预期的中间转换；不能仅依据源代码中没有 `.item()` 就判定没有同步。

3.8是有效推进统计；每次attention仍计算6行/请求。整模型吞吐使用实际输出计数计算，不能把
`6×BS` 当成最终生成token数，也不能将单层attention耗时直接换算成整模型吞吐。

当前未给出性能改善百分比目标。先记录 Native 基线与 PTO 差异，功能通过和性能达标分别判定。

## 9. 计划中的实现文件与结果产物

完整CSA计算、Native存储适配、单层对照和完整graph脚本均已实现并运行，
当前实际入口如下，命令及覆盖范围见[验证入口说明](README.md)。
真实模型的服务forward派发已实现，入口和启用方式见[服务说明](DSV4_FLASH_CSA_SERVICE_FORWARD.md)。
正式B4/B40真实单层入口已验；最新动态版本B1图重放及多B同编译产物切换通过，见日志83；
不代表整模型/P5通过。

单/双卡开发入口限于P0～P4。P5使用独立16卡入口，启动前检查实际设备、进程组及P/D资源；
单层验证脚本不自动启动整模型。下列probe、单层、graph及DP脚本均属于开发验证范围。

```text
tests/pypto_test/
  DSV4_FLASH_CSA_VALIDATION_PLAN.md       # 当前文件
  dsv4_csa_all_groups.py                 # P1：五组metadata、负例和接受修正probe
  dsv4_csa_full_compare.py               # P2：完整单层Native/PTO对照
  dsv4_csa_continuous.py                 # P3：跨步状态和接受轨迹
  dsv4_csa_full_replay.py                # P3：完整CSA同图内容更新
  dsv4_csa_cross_stream.py               # P3：Native metadata跨流生产
  dsv4_csa_dp_metadata.py                # P4：当前覆盖DP2 metadata
  dsv4_csa_native_fixture.py             # 原生层、metadata及cache owner
  dsv4_csa_profile.py                    # 独立Native/PTO profile和PTO DFX

vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/
  decode_csa.py                         # TP1/S6完整CSA计算入口
  native_adapter.py                     # NativeCSACall，单次完整CSA提交
  native_storage.py                     # Native物理存储的零拷贝描述符
```

接入延续现有模型和权重加载流程，不新增自建的完整 DeepSeek 模型实现。
硬件正确性用例使用真实算子；纯CPU mock测试只能证明其声明覆盖的接口逻辑。

每次运行输出到独立结果目录：

| 产物 | 内容 |
| --- | --- |
| `manifest.json` | 版本、有效配置、ND实际格式、量化、设备数量、环境标识、接受步长口径、容差与随机种子 |
| `case_matrix.json` | Case ID、所需环境/卡数、实际卡数、BS、长度、模式、图容量、状态和耗时 |
| `metadata_schema.json` | 参数来源、dtype、shape、stride、单位、设备和更新阶段 |
| `acceptance_trace.json` | 每请求每步的有效推进量、请求映射和计算长度 |
| `comparison.json` | 输出、cache/state、未写区域的逐项比较结果 |
| `graph_report.json` | Capture/replay、稳定地址、实际PTO执行、bucket和同步检查 |
| `performance.json` | ND对照基线、所需/实际环境、计时范围、warmup/rounds、median/p95和各rank负载 |
| `handoff_16card.md` | 本机P0～P4结果、版本和配置、未解决问题、F01～F06待验清单及复现入口 |
| `failures/` | 首个失败的输入、metadata、必要状态、差异位置和复现信息 |

使用生成规则、seed和校验和复用大规模历史fixture；不要为每一步重复保存完整128K cache。
必要时保存一次初始snapshot及后续增量。失败材料必须足以复现，不能仅保存一个误差百分比。

结果记录至少包含：

```text
case_id, phase, status, backend, device_id
environment_scope, required_npu_count, actual_npu_count
batch_actual, requests_capacity, tokens_actual, tokens_padded
initial_context_len, current_seq_lens_range, query_lens
graph_mode, graph_bucket, dp_sync_skipped, effective_dp, effective_ep
acceptance_trace_id, acceptance_metric_definition
dtype_layout_contract, effective_weight_nz_mode, observed_weight_layout, tolerance_contract_id
output_check, cache_checks, untouched_region_check, execution_evidence
artifact_path, failure_reason
```

状态限定为 `NOT_RUN / PASS / FAIL / SKIP / BLOCKED / PENDING`。
Skip必须说明原因，不能计入通过数；P5/F01～F06未执行时记录`NOT_RUN`及实际缺少的前置条件。
旧机器的“仅16卡环境，本机两卡”只保留为历史证据，不再作为当前机器的跳过原因。
本机完成报告分别给出“P0～P4开发验证结果”和“16卡待验项”，不将P5计入本机通过率的分母，
也不把本机通过率用于表示整模型已通过验收。

## 10. 阶段出口与执行顺序

| 阶段 | 执行环境 | 阶段完成条件 | 当前状态 |
| --- | --- | --- | --- |
| P0 | 【本机·单卡】 | 环境、接口、目标参数、ND布局及比较规则已记录 | PARTIAL：CANN9.0运行时及参考ND已验；正式ModelSlim单CSA层24参数加载及ND/cache布局已过，完整目标配置复核仍待完成，见日志第78节 |
| P1 | 【本机·单卡】 | 正式metadata矩阵和负例通过，无非法访问或额外值回读 | PASS（metadata）：六档100步、graph、负例及新增QLI长度断言已通过，见记录第30节 |
| P2 | 【本机·单卡】 | 六档BS、真实128K附近长度、目标量化ND输出与状态对照通过 | PARTIAL：正式ModelSlim为2/18（B4/B40、history131071），输出、Top-K及cache/state通过，见日志第78节；参考18/18单列，不代替余下正式16组 |
| P3 | 【本机·单卡】 | 多步接受轨迹、graph内容更新、padding与跨流等待通过 | PARTIAL：正式权重B40原mixed100容量的eager0～46、B4同图A→B→A共936项通过，见第79节；正式100步和G04～G07仍待验。参考权重旧100步、G08证据保留在第60～62、71～72节，不计正式验收 |
| P4 | 【本机·双卡】 | 两rank不均衡负载和对应DP分支通过，已形成16卡移交材料 | PENDING：真实DP2 metadata已过，完整CSA/dummy/PD分支未验 |
| P5 | 【仅16卡环境】 | 16卡ND整模型功能与性能完成验收，F01～F06通过 | NOT_RUN：正式权重已到位，等待前置验证、服务派发与P/D运行资源 |

本机执行顺序为：ND配置与metadata合同 → probe → attention-only ND eager入口 → 接受轨迹与graph →
双卡DP协调 → 输出本机结果和16卡移交清单，至此结束本机开发验证。
随后在当前机器复核16卡配置，先启动P再启动D，执行P5。实现过程中保留原生对照入口和独立fixture，失败时只重跑受影响用例。

在正式执行前需落实的口径：

- 【仅16卡环境】GBS=64是BS=4基准点，还是所有场景固定的全局并发；不阻塞本机BS扫描。
- 【本机与16卡共用】接收步长3.8是否包含必出的一个token。
- 【本机与16卡共用】128K指初始历史长度、输入长度还是含输出的总长度；是否采用131072。
- 【本机与16卡共用】目标checkpoint各attention模块的真实量化、scale及ND布局；ND格式已固定。
- 【仅16卡环境】最终环境是否另行开启DSA-CP、O-proj跨DP TP或其他影响attention通信的配置。
- 【按环境分别核对】本机DP2及最终DP16是否跳过DP同步，以及各自实际选择的graph bucket。

这些条目不妨碍先按文档假设开发metadata probe，但必须在最终验收报告中明确，不能保留为隐含假设。

### 10.1 2026-09-23 剩余工作核对

六个阶段中P1已完成，P0/P2/P3/P4部分完成，P5尚未启动。按实际工作归纳如下，
各行不是等量工作，不能用已完成行数估算完成百分比。
用户指定的连续精度问题（B16/24/32 step63、B40 step46）已修复，原参考mixed100
均完成eager/graph数值复验；前三组shell收尾问题另记，见日志第71～72节。
P3其他场景及P4按用户要求暂停，未因精度修复自动恢复。

| 剩余工作 | 尚需完成的内容 |
| --- | --- |
| 服务forward接入 | 已实现真实target C4入口、Native回退、warmup/capture和KV生命周期；正式B4/B40单层服务入口已验。动态B尾块修正后的正式B1图重放及B1→2→3→4→5→40→1同产物切换通过，见日志83；整模型仍归P5 |
| P3场景覆盖 | G04～G07的BS/bucket切换、padding/dummy、请求生命周期和prefix共享；接受边界已有B4证据，其他必要覆盖仍待补齐 |
| P4完整双卡协调 | D01～D05接入完整CSA，覆盖不均衡负载、空rank、连续切换及实际图选择；已有metadata结果不替代本地计算验收 |
| 正式权重P0/P2 | 75分片及Native单CSA层加载/ND布局已验；正式矩阵已过B4/B40、history131071，剩余16组及B4/B40关键边界种子0/1待验；参考18/18不计作正式验收 |
| P5整模型 | F01～F06全部未执行；准备实际配置，先启动P再启动D，验证所有目标CSA层、DP16/EP16/EPLB、真实DSpark和PD链路 |
| 性能与同步 | 已有单层profile及泳道图；仍需稳定计时、p95、同步检查、双卡协调和整模型指标，按第8节分别记录 |

独立CSA算子、Native缓冲直读写及八次外部PTO适配调用的消除已完成，
不再列为待实现项。最新去适配版本仅执行用户指定的针对性回归，
不能将历史18组或100步证据外推为每次优化后的完整回归。
剩余项按依赖推进；正式权重已到位，后续目标数值验证使用ModelSlim加载入口。
当前正式权重B40前47步及B4图重放已通过；step46的Query、QR及SWA差异已消除，
剩余attention 35个BF16元素差异。2026-09-23用户要求暂停后续精度排查，等待下一项工作选择。

### 10.2 2026-09-23 用户指定离线 P/D 执行方式

后续整模型CSA接入与性能工作改为：P先以TP4×DP4（EP16）生成不同场景的离线KV cache，
释放16卡，再以D TP1×DP16、EP16加载同一缓存，分别运行Native与PTO。
不再以同时在线的P/D或Mooncake安装作为本轮准备工作的前置条件。
该工作独立于仍暂停的其他P3/P4与剩余精度排查，不自动将它们标为通过。

离线bank包括所有target与draft缓存、compressor state、Indexer scale和逻辑页映射。
P历史长度先覆盖255/4095/32767/131071/131072/131073；每档4份确定性token输入。
D每卡BS扫描1/4/8/16/24/32/40，均衡负载，GBS=16×BS；先eager验证实际入口。
P计算H、D提交H+1并恢复h(H)，沿用Native hybrid PD重算最后token的状态边界。

缓存产出先验证完整层覆盖和TP4复制关系，再开放D加载。Native/PTO使用相同bank和配置，
单列IO/预热/编译耗时、实际PTO命中和回退、接受率以及稳态decode性能。
离线回放不替代在线传输及完整F01～F06验收。具体入口及结果口径见
[离线P/D方案](DSV4_FLASH_CSA_OFFLINE_PD.md)，执行过程继续记入验证日志第84节及后续。

## 11. 源码依据

以下链接固定到已核对的提交；后续源码变化时重新检查对应合同。

- [DeepseekV4Attention 与外层 HC/norm 调用](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/models/deepseek_v4/model.py#L662)。
- [五组 metadata 按对象 prefix 获取](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/attention/dsa_v1.py#L1550)。
- [原生 actual sequence length 的绑定](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/attention/dsa_v1.py#L2141)。
- [Runner 的 device/host metadata 构造](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/worker/model_runner_v1.py#L3405)。
- [异步接受计数和长度修正](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/spec_decode/utils.py#L14)。
- [DP 同步与 padding](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/worker/model_runner_v1.py#L782)。
- [跳过 DP 同步的条件](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/utils.py#L1208)。
- [当前基线的weight_nz_mode配置](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/ascend_config.py#L500)。
- [权重NZ转换策略与mode=0](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/utils.py#L275)。
- [A3 slot mapping 格式](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/attention/dsa_attn_kv_plan.py#L93)。
- [Indexer cache 的 scale dtype](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/models/deepseek_v4/indexer.py#L110)。
- [Compressor metadata 的事件等待与输入](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/models/deepseek_v4/compressor.py#L177)。
- [Device metadata executor](https://github.com/vllm-project/vllm-ascend/blob/34bb51f93724c565362f5108f5226303e1b56cad/vllm_ascend/worker/device_metadata.py#L45)。
- [PTO decode_csa_tp1 函数体](https://github.com/hw-native-sys/pypto-lib/blob/205255b4770ee84dfa176bcbc7bbef651953c7e1/models/deepseek_v4_flash_dspark/decode_csa.py#L761)。
- [PTO 的 S=8 配置](https://github.com/hw-native-sys/pypto-lib/blob/205255b4770ee84dfa176bcbc7bbef651953c7e1/models/deepseek_v4_flash_dspark/config.py#L242)。
- [PTO indexer 的请求划分、可见范围与 Top-K 分段](https://github.com/hw-native-sys/pypto-lib/blob/205255b4770ee84dfa176bcbc7bbef651953c7e1/models/deepseek_v4_flash_dspark/decode_indexer.py#L189)。
- [Qwen3 forward 替换参考](https://github.com/nalinaly/vllm-ascend/blob/093b1eb519f2a706b8b471d40cc4c66c64e68f1b/vllm_ascend/models/pypto_qwen3_attention.py#L195)。
