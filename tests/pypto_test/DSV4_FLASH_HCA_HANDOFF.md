# HCA layer PyPTO 接入记录

> **性能优化的调测信息记录在 [DSV4_FLASH_HCA_OPTIMIZATION_LOG.md](DSV4_FLASH_HCA_OPTIMIZATION_LOG.md)**
> （与 CSA 侧的 `DSV4_FLASH_CSA_VALIDATION_LOG.md` 对等）。本文件只保留接入合同、
> 验收状态与剩余事项；逐轮的实测数据、口径纠正与候选取舍以该 LOG 为准。

## 性能优化的当前入口（2026-09-30 接手）

目标保持各档领先同配置Native 20%以上，不因已有候选失败而缩小目标。
后续交替推进incore与调度；128K优先、8K守护，候选按8:2衡量，最终覆盖支持档位。
先单卡，再整机token验证；耗时报告min/max/mean，并保留既有P50，不能把各pass中位数的max冒充单次重放max。

历史七档加权1.117属于第77节的源码和环境，未达目标；新轮采用CANN9.2、mode2、atomic0，
Native使用npugraph_ex dynamic=False、inplace/static开启，正式对照开SuperKernel，细化incore时关SuperKernel。

此前本节把第91节的“同步原语计数”当作已证实根因，但第92节已用实验撤回该推论。
“算子侧穷尽”“所有sync_start有害”也超出已测候选的证据范围，不能据此关闭优化方向。
保留失败候选的源码与边界，优先从最新ops-transformer的实际实现寻找数据复用或流水差距。

当前第一项是mHC post residual常驻UB：参考A3 `MhcPostKernel::ComputeCopyOutAllX`，
把每个row tile的residual加载/转FP32由16次改为4次，复用四份输入；
分工、依赖、逐输出加法顺序和BF16舍入点不变。基线与候选均已显式CPU解析/编译通过。
私有冻结包实验见[hca_residual_reuse_20260930](hca_residual_reuse_20260930/README.md)。
128K/B16与8K/B24完整状态逐bit一致，post核内mean分别下降26.01%/6.85%，已保留实现。
整层mean分别+2.72%/−0.82%，max均增加，尚未证明整层收益。
随后Q投影sync_start及与反量化联动、完整压缩块mask快路径均未受益，未接入，见LOG第100–101节。
前者泳道证明等待会转移至反量化；后者新增标量分支后attention核内也更慢。
O-B激活常驻二版已按目标函数接入，未覆盖CSA的其他任务图；128K/B16与8K/B16的O-B
核内mean下降19.37%/6.76%，但整段mean上升0.38%/1.61%，没有整段提速结论。
两档和ROW32的128K/B4完整状态逐bit一致；按用户要求保留核内收益，继续解决调度联动。
记录见LOG第102节；20%领先目标仍未达成。
长档attention的max/sum已改为同AIV的UB环，显式extract避开动态slice偏移丢失。
核内mean下降6.27%；叠加O-B后服务入口128K/B16 min/max/mean为561.50/655.50/592.21，
同轮base为585.75/638.50/600.86。mean改善但max上升，且同进程筛选方向相反，稳定整体收益未证实。
长档、组合和16384/B4短循环均逐bit通过；Q提前解析暂不接入。见LOG第103–105节。
Native跨query延续流水已接入，见LOG第106–107节：长档QK/PV保留三槽，只在worker最后一条query排空。
128K/B16服务入口min/max/mean从587.00/648.75/605.71变为558.50/634.75/583.90μs，
两种筛选方向一致；8K/B24未改分支有+7.38μs波动，长短8:2加权mean改善2.94%。
attention AIV核内mean下降7.86%，但上游起跑、O-B及mHC post也变化，不能将整段收益全部归给单task。
长短档完整状态、压缩块边界、不同query数及dummy图重放通过；B2空metadata测试误报已修复。
接下来继续核内与调度交替，参考Native多query复用，并联动观察生产者/消费者及并发任务；
不要重复把所有任务同步启动。尚无本版本新的七档Native或整机验收，不更改20%目标。

最新推进见LOG第108–111节：Q激活L1复用的三个版本与提前解析联动已完成筛选，
没有将整段负/不明收益误作成功；四块A复用和部分低max信号保留为后续联动线索。
新同卡128K/B16 Native/PTO a7a9f314 min/max/mean为533.25/562.25/544.35与
590.75/646.50/614.94μs，PTO仍慢12.97%。Native实启static/SK，PTO SK1报107017后
完整重跑SK0；两侧dynamic=False、inplace/static开启。Native自身编译图与eager有小浮点差异，
不外推为跨实现精度或token验收。目标仍为各档快20%以上，不能用旧较慢Native替代新分母。

随后已接入Q RoPE flat Gather，保持算术、worker与依赖，用整块索引替代8次逐行Gather。
Q反量化核内mean44.55→35.48μs、max48.60→36.52；长档整段mean持平、max略升，
8K/B24 mean637.78→631.03、max671.25→638.25。按规则保留核内收益；稳定整段收益未证实。
长短档、B3混合尾块与补位图3→2→1→3的完整状态精确，未新增整机或七档验收。
调度联动已完成（112节）：单独Q反量化sync虽缩短本组跨度，却增加并发gather等待；
两组同时sync整段更慢，均未接入。Native多query复用的三种方案也已完成（113节）：
M128共用压缩KV，raw仍各自M64；全GM、半UB与恢复三槽都未降低Attention核内/整段。
半UB编译容量及三槽Q/P复用L1的边界有完整记录；未修改共享工具链或扩大无收益候选测试。
生产代码保持ad0e6bbe（后续3a3ad900为实验记录）。114节已完成mHC pre混合值常驻UB、
原始BF16读取及组合：完整状态精确，但mix核内均更慢，不接入。
115节已接入4行有效分工/8行物理盒+D512，B16的mix 12→24，与12个Sinkhorn按真实时序观察；
保留D256统计次序及stage2，所有读写明确有效行，避免尾块暂存竞争。
mix核内min/max/mean37.06/38.62/37.885→25.16/28.80/26.548（mean−29.93%）。
128K/B16整段548.50/597.25/569.23→556.00/589.25/571.18；
8K/B24为592.00/650.00/613.40→602.50/631.75/615.88，尚未证明稳定整段收益。
长短完整状态、B3两行尾及补位3→2→1→3通过；无新七档/Native/模型验收。
下一轮调度检查mix与Sinkhorn的sync_start/提前解析联动，不批量全开，也不缩小20%目标。

116–117节已完成该轮：mix+Sinkhorn同步独立ABBA有收益，但与widen候选叠加后
长mean只降1.35、短增4.20μs，暂不接入同步，样本max下降留作削峰线索。
已接入widen物理M8/有效M4、K512 stage4、固定K512统计顺序，显式有效写回去掉单尾块缓冲。
128K/B16 min/max/mean为576.75/624.50/590.25→555.25/603.75/568.65μs；
8K/B24为610.00/674.00/621.875→620.75/643.75/631.25，短档mean增加9.375，8:2总体改善2.626%。
widen核内25.55→26.30并未变快，收益伴随下游Q反量化/gather/attention时序变化，不能错误归因扩核。
两档完整状态精确，B3两行尾及补位3→2→1→3通过；生产CPU编译/load通过。
未新增七档Native/模型验收，下一步研究Q_B与反量化的Cube/Vector生产者消费者流水，目标不变。

118–119节完成Q_B双槽MIX及分组发布诊断，私有实验见[hca_qb_mixed_20260930](hca_qb_mixed_20260930/README.md)。
MIX20使压缩gather组跨度33.46→106.40μs并推迟Attention；MIX24同步长短均无整体收益。
MIX16长档mean少4.50、max少11.50仅是线索，尚未接入，不称稳定改善或EP16尾延迟已解决。
必须联动看正在优化的组、并发gather、消费者启动和整段min/max/mean；不要累加独立优化收益。
分组版还发现一个实现边界：私有Q根Tensor manual_dep=True后，外层reshape默认将新视图恢复为false，
四个InOut依旧串行。四组/两组外层reshape的计时不能作为独立发布收益；检查生成提交参数和真实泳道。
算子侧直接分配二维Q的`streamed4_flat`保持物理字节和四条直接TaskId依赖，避开写入前reshape。
长档四类状态、自身图重放、保护区通过，min/max/mean为544.00/583.00/553.15→538.00/564.75/550.325μs，
第120节短档也通过：8K/B24 min/max/mean为596.75/634.50/611.20→600.50/626.50/607.975μs；
长短8:2 mean约−0.514%，样本max均下降。
每档五个ABBA小组只有三组改善，长P50仅少0.50，候选保留继续联动，尚未替换生产。
DFX确认反量化组真实重叠，但有一组Cube跨度117.80、其他约66–68μs，核内mean仍约60。
下一轮重点解决生产者晚起跑并观察并发gather，可在消除reshape假依赖后再试两组发布，
不能复用之前带假依赖的两组结论。生产保持9015d45a，未改共享工具链或CSA，无新Native/模型验收。

第121节已接入四组/总16 Cube的Q发布，每组4 Cube→12 AIV；二维私有Q保留manual_dep，
四条TaskId直接接到Attention。128K/B16 min/max/mean为585.25/611.00/593.80→563.75/579.00/571.475，
8K/B24为608.75/662.50/626.075→610.75/641.75/627.825μs，长短8:2约−2.952%，max均下降。
DFX的Q单worker核内更慢、并发组亦变化，按整段收益保留，不归因成Q核内提速。
ND保持原公共Q投影，mode0/mode2均CPU编译/load通过；两档完整状态精确，B3尾块/补位图及ND B4通过。
生产有效函数与已测ready_v2 AST一致，未改变共享CSA/Native/工具链；详情见LOG121和实验README。
第122节已接入mHC post门控转置，保留BF16舍入与所有加法顺序。
在Q分组上测组合增量，长mean591.55→576.55、短630.20→613.775 μs，8:2约−2.550%，
max均下降；不能与第121节独立窗口收益相加。核内post mean28.8325→26.4096，−8.403%。
两档四类状态逐bit、B3尾块/补位图及生产CPU编译/load通过；未改共享CSA/Native/工具链。
第123节最新同卡Native对照：128K/B16 Native min/max/mean=531.25/595.25/551.431，
PTO=554.50/628.25/578.014 μs，mean仍慢4.82%。Native实际static/SK、dynamic=False、inplace开启，
PTO SK0。无新七档/模型验收，比Native快20%以上目标未达成。
下一项核实Native O-B整宽量化与PTO分组量化的差异，待核实，不预设收益。

## 目标与分支

- 分支：`dsv4-flash-hca-pto-v0.25.1rc1`。
- 工作目录：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-hca-pto-0251rc1`。
- 从 CSA 分支已提交的 `65500b91` 创建独立 worktree，继承 v0.25.1rc1 baseline 和 CSA 公共实现。
- HCA 参考：`pypto-lib/models/deepseek_v4_flash_dspark/decode_hca.py` 的 `decode_hca_tp1`，
  以及 `decode_compressor_ratio128.py`、`decode_sparse_attn_hca.py`。
- 正式权重：`/data/model/DeepSeek-V4-Flash-0731-w8a8`，首个 HCA 是第 3 层（下标从 0 开始）。

## 接入边界

最终为一次 PTO 调用完成 mHC pre、input RMSNorm、QKV/RoPE、C128 compressor、HCA attention、
O projection 和 mHC post；FFN 保持 Native。HCA 没有 Indexer，不迁移 CSA 的 Indexer 工作区。
公共 mHC/QKV/O projection 优先复用本仓库已有实现，HCA 特有逻辑放在
`vllm_ascend/ops/pypto/deepseek_v4_flash_hca/`。

与 CSA 相同，首批支持 A3、TP1、DSpark S6、动态 batch，后续使用既有 P TP4×DP4 离线 KV，
验证 D TP1×DP/EP16。上游 HCA 参考的接口不直接等同于 Native 接口：

| 入参 | Native 接入约定 |
| --- | --- |
| hidden / mHC 输出 | BF16，`[B*6, 4, 4096]` |
| norm 权重 | BF16；不在初始化时永久转 FP32 |
| position | 原始 INT64 device tensor |
| state slot | 原始 INT32 `[B*6, 2]` 页号、页内偏移 |
| state | A3 block_size=32 配置下，每页 8 行、每行 1024 个 FP32 元素 |
| compressed KV | Native BF16，每页 32 行、每行 512 个元素 |
| compressed RoPE / slot | Native `compressor_metadata` 返回的紧凑行，不展开为每 token 缓冲 |
| 页表、query bounds、seq_lens | 原始 device tensor；核内计算地址与有效请求条件 |

Host 仅使用已有配置、shape/stride 和调度信息做入口判定、零拷贝描述符绑定。
不要增加 state 搬运窗口，也不要修改 PyPTO、Simpler、pypto-lib 依赖仓库。

## 执行顺序与验收

2026-09-28 起步时用户明确：先修精度再优化性能，精度以相同输入/解码设置的最终输出 token 一致验收，
不要求中间张量逐 bit 移植。max_abs、RMSE 和 cache/state 差异保留作定位数据，不单独判 token 精度失败。
写保护、有限值、PTO 实际执行和图重放功能检查继续保留。当时先做真实模型 token 对照；
以下是历史接入顺序，当前已按用户的新目标恢复性能优化，以本文件顶部和优化LOG为准。

1. C128 子链：正式第 3 层权重，验证动态 B1/B4、跨 128 边界、零压缩输出、dummy、
   反序与滚动 state 页表；同时比较 state/压缩 KV 和完整分配的写保护。
2. HCA attention：直接读取 Native 原始 KV 页表、seq_lens 和压缩 KV 页表，验证因果窗口、
   压缩块可见性与 inverse RoPE；不重新增加外部窗口转换。
3. 整层 root：复用公共 mHC/QKV/O projection，完成 lowering、设备编译、单层 Native 对照。
4. 服务 opt-in 入口：保留 Native fallback，接好 KV connector、dummy、图捕获/重放；
   未通过整层门禁前，不对外注册一个只会 fallback 的 HCA 模型入口。
5. 连续 decode、请求生命周期与 BS 切换，再做 TP1×DP/EP16 整模型对照和 profiling。

性能调优在功能门禁之后。测试只做当前改动所需的用例，不重复既有 CSA 七档测试。
所有 NPU 执行经 `task-submit`，过程报告留在本目录下；大文件与编译产物本地保留。

## 2026-09-28 起步记录

- 已创建独立分支/worktree，原 CSA 目录的未提交改动未迁入。
- 已实现 C128 投影、128 行 softmax pooling、RMSNorm/RoPE、直接 state/KV 写回。
  先读取历史并叠加本步投影，再提交 state，避免滚动页提前覆盖历史。
- 已新增 `dsv4_hca_compressor.py`，使用正式 compressor 权重和 Native compressor 作对照。
  首次 lowering 发现测试入口遗漏 `task_dummy(deps=[])` 的必需参数，已修正。
- CPU lowering 已通过，结果：`results/hca_compressor_20260928/lowering/report.json`。
  首轮还修正了新代码中的模块属性调用、临时变量类型重用和 Tensor/Tile 混用，未修改编译器。
- Native 的 `num_compressed_tokens` 是输出容量，不是本步有效压缩行数。
  测试已采用 builder 的 `min(num_tokens, num_tokens // 128 + num_reqs)` 公式，
  消费有效 slot 并保留无效行；“零压缩输出”指有效行数为零。
- 正式文件中第 3 层 compressor 的 wkv、wgate、APE、norm 均以 FP32 存储。
  测试按 Native BF16 参数的加载规则转换：wkv/wgate/norm 为 BF16，APE 为 FP32；
  这与将已经加载的 Native BF16 norm 改成 FP32 是两件事，后者不采用。
- 首个排队任务 `task_20260928_130005_19593033480` 在启动前撤销，用于修正上述测试容量。
  设备首轮 `task_20260928_130120_2012190221` 在编译阶段失败，未执行 PTO 计算。
  原因是新实现的 RMS 单行归约产生了不合法的 `[1,1]` FP32 列主序 tile，
  以及在 inline 返回表达式里直接创建 dummy。已改为 UB 内 8 行归约后只发布一行、
  先绑定 dummy TaskId 再返回；未修改依赖库。
- 修正后的完整设备代码已在 CPU 上编译通过，证据：
  `results/hca_compressor_20260928/build_v2/report.json`。
  设备复跑任务：`task_20260928_130512_220087618629`，结果目录
  `results/hca_compressor_20260928/device_v2/`。
- 对照沿用当前 CSA 单层诊断的零容差统计，数值差异单独记录；非有限值或写保护失败记 FAIL。
  正常运行只记 MEASURED，不将“无越界写”当作数值通过或整层通过。
- C128 子链任务已完成（退出码 0），四个场景均无非有限值，Native/PTO 写保护全部通过：

  | 场景 | state 最大绝对差 | 压缩 KV 差异元素 |
  | --- | ---: | ---: |
  | B1，history=124 | 2.38419e-6 | 1，最大差 3.05176e-5 |
  | B1，history=128，无有效压缩行 | 2.14577e-6 | 0 |
  | B4，history=0/122/127/255 | 3.57628e-6 | 0 |
  | B4，实际 B3，dummy position=999999 | 3.81470e-6 | 0 |

  这些是合成输入和历史下的首轮差异，不作为连续轨迹或整层数值验收结论。
- 已移入 HCA attention，保持上游 TP1 的 raw/compressed 两分支流水；删除外部 SWA
  indices/lens 入参，核内直接按 Native 页表聚合原始 KV、计算有效范围。
  RoPE 直接消费 Native 交错 FP32 数据；修复 B1 的有效 token 尾块覆盖。
- 已串起 `decode_hca.py` 整层 root，公共 mHC/QKV/O projection 复用 CSA 性能版。
  完整设备代码 CPU 编译通过：`results/hca_layer_20260928/build_v1/report.json`。
- 权重准备复用 CSA 的公共函数，只将 compressor 宽度改为按 C4/C128 选择；Indexer 参数
  仅在 C4 分支绑定。缓存描述符复用 `physical_pages` / `table_storage`，不复制 Native 数据。
- 正式第 3 层首轮 `task_20260928_131605_257972321342` 被队列自动附加的 `--device 0`
  参数挡在启动前；测试已接受逻辑卡号，后续提交显式给出 `--device 0`，物理卡仍由队列管理。
- B1/history124 整层：`task_20260928_131716_26051483190`；
  B4/history32764 整层：`task_20260928_132015_267219922943`。两轮均退出 0，
  Native/PTO 的写保护和只读 metadata 检查通过，输出无非有限值。
  两轮输出最大绝对差均为 0.03125，RMSE 分别为 0.00211580 和 0.00167286。
  这两轮继承了当时默认的 `atomic_add=1`，不作为固定归约基线。
- 服务入口已接入：`PyptoHCADeepseekV4ForCausalLM` 只替换 C128 attention 半层，
  C4/SWA/FFN 仍走 Native；注册 `vllm::dsv4_hca_forward`，
  共用 CSA 的入口判据、按步 metadata 缓存和零拷贝描述符，并保留 KV connector 的等待/发布顺序。
  runner 复用原有两处挂点：加载后准备权重、图重放档位判定。
- 首轮服务/图检查 `task_20260928_132150_27044162541` 发现相同输入的输出变化，
  只影响 B1 最后两行，cache/state 相同。复核发现 HCA 复用性能版计算，但环境仍使
  公共投影采用 `atomic_add=1`。未调整阈值，改用 CSA 性能基线的固定归约后复验。
- `atomic_add=0` 的 B1 服务/图检查 `task_20260928_132443_276257314513` 已完成：
  直接调用与服务 custom op、服务 eager 与图 replay 的输出及三份完整 allocation 均逐 bit 相同；
  全部写保护通过。对 Native 的输出最大绝对差 0.015625，RMSE 0.00211179，
  仍不声明数值验收通过。证据：`results/hca_layer_20260928/b1_service_graph_v2/report.json`。
  HCA 首版的配置和算子注册入口均明确要求 `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0`。
- 固定归约 B4/history32764 的服务/图任务：`task_20260928_132703_280344327829`，
  已完成，退出码 0。直接调用与服务 custom op、服务 eager 与图 replay 的输出及三份完整
  allocation 均逐 bit 相同，Native/PTO/graph 三轮各 19 项保护区及只读 metadata 检查全部通过。
  对 Native 输出最大绝对差 0.03125、RMSE 0.00167138，未宣称精度通过。
  结果目录：`results/hca_layer_20260928/b4_service_graph_v2/`。

CPU lowering：

```bash
source ../env-dsv4-0251rc1.sh
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
python tests/pypto_test/dsv4_hca_compressor.py --lower-only \
  --output tests/pypto_test/results/hca_compressor_20260928/lowering
python tests/pypto_test/dsv4_hca_compressor.py --build-only \
  --output tests/pypto_test/results/hca_compressor_20260928/build_v2
```

设备对照：

```bash
task-submit --device auto --max-time 900 \
  'bash tests/pypto_test/run_hca_compressor.sh tests/pypto_test/results/hca_compressor_20260928/device_v2'
```

测试临时复用 CSA 工作目录内已验证的 Native 扩展和 CANN 自定义算子包；
Python/PTO 源码由 `dsv4_csa_env.activate()` 指向当前 HCA worktree。

整层/服务检查：

```bash
task-submit --device auto --max-time 900 \
  'bash tests/pypto_test/run_hca_single_layer.sh tests/pypto_test/results/hca_layer_20260928/b1_service_graph_v2 --batch 1 --history 124 --service-graph --device 0'
```

模型选择使用 `hf_overrides={"architectures": ["PyptoHCADeepseekV4ForCausalLM"]}`。
初始化前设置 `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0`，NZ 环境与 `weight_nz_mode` 一致。
服务入口已做单层真实 metadata/custom op 检查，尚未启动整模型服务或做 DP/EP16 验收。

## 2026-09-28：单卡七档初步性能对比

按用户要求补齐 128K/B4、B8、B16 与 8K/B16、B24、B32、B40，全部 S6。
七档各自一张卡，同一进程内 Native/PTO 交替计时；任务均退出 0。
正式第 3 层权重，合成输入及历史，NZ mode 2、Native deterministic level 0、PTO atomic0。
每侧 5 次预热、20 次正式图重放，图外设备事件计时，初态恢复与编译不计入。
范围为完整 attention 半层（mHC pre/norm/HCA/mHC post），包含本层 compact metadata 生成，
PTO 走真实服务 custom op。当前 Native wo_a 仍受 CANN 9.0 缺少 NZ 算子的兼容路径限制。

| 历史 / Batch | Native P50 μs | PTO P50 μs | PTO 延迟变化 |
| --- | ---: | ---: | ---: |
| 128K / B4 | 427.66 | 470.93 | +10.12% |
| 128K / B8 | 551.33 | 575.59 | +4.40% |
| 128K / B16 | 693.89 | 713.06 | +2.76% |
| 8K / B16 | 565.25 | 615.31 | +8.86% |
| 8K / B24 | 717.56 | 704.92 | −1.76% |
| 8K / B32 | 777.73 | 795.60 | +2.30% |
| 8K / B40 | 869.16 | 929.59 | +6.95% |

仅 8K/B24 在本轮中位数上略快，不能宣称稳定收益；其余六档仍慢。
每档 88 项保护区/只读 metadata/服务与直接调用/两侧图重放一致性全部通过。
Native/PTO 输出仍有差异：128K/B4 max_abs=0.015625，其余为 0.03125，零容差均未通过。
SWA 和 state 的原始字节也有差异，不能将图重放一致性写成 Native 数值通过。
本轮 S6 不跨 C128 组边界，compressed cache 不新增写入，不覆盖新压缩行的数值验证。

完整 P95、数值差异、task 与复现命令见
[七档报告](results/hca_performance_20260928/README.md)。
新增计时工具 `dsv4_hca_performance.py`；单层入口新增计时/NZ/确定性参数，未改生产算子。

## 2026-09-28：首轮 token 故障与修正

全卡资源等待期间，先用 TP1×DP/EP8 验证 8K/B16、每请求 256 token。
两侧均为正式权重、同一离线 KV、NZ2/det0、PTO atomic0、FULL_DECODE_ONLY；
EP8 显存预算为 95%，默认 EP16 仍为 90%。

修正前 v2：32768 个 token 中 32640 个不同，首 token 相同，PTO 随后持续输出 0。
发现 HCA 压缩 KV gather 会读取有效长度之后的未初始化行，零概率不能隔离 NaN。
只在本仓库 HCA 内核加入 `seq_lens // 128` 读取上限，部分页用动态 `valid_shape`，
私有片内未读位置保持零；不清空 Native cache、不增加转换，也未修改依赖仓库。

整层设备编译通过。H8192/H8320 两项单卡无效 KV 毒化检查通过有限值和写保护，
Native/PTO max_abs 均为 0.03125；H8192 输出与既有有限填充快照逐 bit 相同。
这些单卡用例首次执行时已包含修正，不冒充旧实现的设备反例。

修正后 v3 `task_20260928_161034_88223618843` **PASS**：32768 个输出 token 全部一致，
DSpark 统计也一致，无缺失结果。8 个 rank 各有 44 次真实 B16 图调用，
全部 20 个 HCA 层均有匹配的 PTO 捕获证据。该轮只重跑 PTO，Native 复用 v2 相同配置结果。
本结论限于该 EP8 场景，不代表 EP16、七档 token 或全部生命周期通过。
[命令、过程与证据](results/hca_token_20260928/README.md)。

## 验收结论（2026-09-29 定稿，CANN 9.2.0-beta.2，上线 decode 口径）

**验收档位为六档**：128K×B4/B8/B16/B24 + 8K×B24/B32。目标是每档 PTO P50 ≤ 同卡
Native P50 × 0.80。**未达标。**

下表取两轮同口径数据，用来同时给出数值与它的复现性：

| 档位 | 第一轮 | 第二轮 | 差 | 距目标 0.800 |
| --- | ---: | ---: | ---: | ---: |
| 128K B4 | 0.999 | 1.011 | +0.011 | +0.199 |
| 128K B8 | 0.932 | 0.944 | +0.013 | +0.132 |
| 128K B16 | 0.961 | 0.957 | −0.004 | +0.157 |
| 128K B24 | 0.925 | 0.921 | −0.004 | +0.121 |
| 8K B24 | 0.956 | 0.964 | +0.007 | +0.156 |
| 8K B32 | 0.974 | 0.978 | +0.004 | +0.174 |
| **六档均值** | **0.958** | **0.963** | — | **+0.158** |

**验收比值的复现性约 ±0.013（六档逐档差的最大值）**，而距目标的差距是 0.158，
**为测量不确定度的 12 倍**——因此"未达标"这个判断不受测量精度影响，是确定的结论。
绝对量上六档距 0.80×Native 差 69～156 μs。功能门禁（对优化前基线的逐 bit 对照）各档全 PASS。

### 档位表的两次调整（用户定档）

1. 新增 **128K/B24**。该档位此前的显存缺口随 `wo_a` 布局修正而消失
   （PTO 曾差 Native 1.63 GiB，与 recast 的 1.603 GiB 吻合），单卡已跑通。
2. 退役 **8K/B16**。
3. 退役 **8K/B40**：上线模板 `--max-num-seqs 32` 是一步最多调度的请求数，而 B40 是
   40 个请求（40×6 = 240 token），超出该上限，这种形状生产里不会出现；
   `--max-num-batched-tokens 400` 不构成约束，卡住的是请求数。它也恰好是最差的一档
   （1.005～1.026）。**因此 8K/B32（192 token）就是上线能达到的最大一步**，
   是长尾档位的代表。若将来 `max_num_seqs` 提高到 40 以上，B40 才重新有意义。

注意单层 bench 仍把 `max_num_seqs` 设为 `max(40, batch)`；B40 退役后各档的 batch
均不超过 32，该表达式对六档等价于 40，与上线的 32 仍有偏离但不再造出超容量形状。

性能根因与已穷尽的算子侧手段见
[优化日志第 0 节索引](DSV4_FLASH_HCA_OPTIMIZATION_LOG.md)。要点：资源不是瓶颈
（AIC 核时间 ÷ 24 = 383 μs，整层 625 μs，目标 538 μs）；唯一有强证据的性能改进是
`wo_a` 布局跟随 Native 实测存法（proj_a 核时间 −46%、端到端 128K/B16 −41 μs）；
其余十余个候选或被否定或落在 ±13 μs 噪声底内。

## 剩余事项

1. **优化版整机 token 验证已完成（2026-09-29）**：`task_20260929_070907_235294817658`，
   16 卡 TP1×DP/EP16、**128K/B24**、`gpu_memory_utilization=0.97`，PTO 与 Native 同档期
   各跑一遍（同时补上了 CANN 9.2 口径的 Native 对照）。生产算子为 v17+v21+v22，
   此前从未整机验证过。

   | 项 | 结果 |
   | --- | --- |
   | `token_status` | **PASS** |
   | token 不一致 | **0 / 98304**（16 rank × 24 请求 × 256 token） |
   | 比对 rank 数 | 16 |

   证据：`results/hca_token_20260929/h131072_b24_ep16_v1/token_comparison.json`。

   **一项未解释的现象，不计入通过**：`spec_decode_mismatched_cases = 16`，即 16 个 rank 的
   DSpark 计数全部不一致。Native 每个 rank 恒为 1032 drafts / 5160 draft tokens，
   PTO 为 1017～1030 drafts 且逐 rank 不同；**两侧接受率都是 100%**
   （`num_accepted_tokens` = 5 × `num_drafts`），说明 draft 质量相同，差的只是调用次数。
   两侧日志里抢占/重算均为 0。推测是 `--async-scheduling` 下步的组成随时序变化
   （PTO 更快，批次凑法不同），但**未经证实**——要证伪需让 Native 自己重跑一次、
   看它的 1032 是否可复现。在此之前不要把该现象说成无害。

2. DP/EP16 正式精度基线 `task_20260928_162843_15654084376` 已 PASS：8K/B16，
   65536 个 token 全部一致，DSpark 统计一致，16 个 rank 各 20 层 PTO 捕获与 44 次真实 B16 图调用。
   证据 `results/hca_token_20260928/h8192_b16_ep16_v1`；这轮不含后续性能优化。
   长上下文与其余 batch 的整模型 token 覆盖仍需补齐。
3. 固定归约下的连续 decode 状态轨迹，B1～B64 的动态 BS 切换、整个算子的 padding/dummy、
   请求生命周期与 prefix 共享。当前 dummy 证据仅覆盖 compressor 子链。

   **2026-09-29 查清了这一项的现状与做法**：CSA 侧早已有完整的 padding 图检查
   `dsv4_csa_single_layer.py: check_padding_graph()`，由 CSA 脚本的 `--padding-graph`
   调用；它捕获满档图后，按 `active` = 满档 → 满档−1 → 1 → 满档 逐次用 Native builder
   在**图外原地**更新输入（`seq_lens[active:]` 归零、`slot_mapping[active*6:]` 填 −1、
   `block_table[active:]` 归零，并以 `num_reqs_actual=active` 重建 metadata），
   再重放同一张图，要求有效请求的输出与**满档 eager 结果的前 active*6 行逐 bit 相同**、
   补位 cache/state 保持初态、写保护只落在有效 slot 内。这同时覆盖了动态 BS 切换与
   padding/dummy 两项。

   **HCA 没有接这个检查**：`dsv4_hca_single_layer.py` 只 import 了
   `guard_checks / make_fixture / make_layer / restore`。所需构件都已具备——HCA 脚本
   自己捕获 NPUGraph，`NativeHCACall` 已接 `compact_metadata`，oracle 用的
   `_compute_compressor_metadata` 就在 Native 的 `dsa_v1.py` 上（HCA 与 CSA 同一个类）。

   **唯一的设计决定**：compact metadata 的生产者放在图内还是图外。CSA 的 `run()` 把
   `_compute_compressor_metadata` 一并捕获进图，所以重放时它会按更新后的 metadata 重算；
   而 HCA 的单层脚本目前在图外算一次 `fixture["compact"]["compressed"]` 再传给
   `NativeHCACall`，重放时会是旧值。要让 padding 测试成立，必须改成与 CSA 同构
   （把 compact 生产者纳入捕获区），或在重放前原地更新那几个 compact 张量。
   落地时请按 HCA 的组（swa / compressed / state，无 indexer）与单一输出 `x_out` 改写，
   不要直接复用 CSA 的函数体。

   **2026-09-29 已完成动态 BS 切换与整算子 padding/dummy 两条**：新增
   `dsv4_hca_padding.py`，HCA 单层脚本加 `--padding-graph`（要求 batch ≥ 2，
   且必须与计时/泳道分开进程）。判据比 CSA 版更直接，用字节级掩码构造期望值：
   期望的整份 cache/state = 初态，只在有效 slot 覆盖的字节上换成满档结果，
   因此一次比对同时表达"补位请求一个字节都不许写"和"有效请求写出的内容必须与满档逐 bit 一致"；
   输出侧要求有效请求的行与满档 eager 结果的前 `active*6` 行逐 bit 相同。

   实测（CANN 9.2，nz_mode=2，deterministic 0，生产算子 v17+v21+v22）：

   | 档位 | 满档 | active 序列 | 结果 |
   | --- | ---: | --- | --- |
   | h124 / B4 | 4 | 4 → 3 → 1 → 4 | 输出、三份 allocation、compact metadata、写保护全 PASS |
   | h8190 / B8 | 8 | 8 → 7 → 1 → 8 | 同上 |

   证据：`results/hca_padding_20260929/h124_b4_v2/`、`h8190_b8_v2/`。
   设计决定已验证：**compact metadata 的生产者放在捕获区内可行**，重放时会按更新后的
   metadata 重算，不需要改成图外原地更新。
   踩过的坑：`make_call` 里不能调 `prepare_weights()`——它内部 `scale()` 做
   `bool(count_nonzero(offset).cpu())` 是同步 D2H，捕获期间会被拒
   （`error 107030` / `Not_Supported(EE1016): Stream during the capture stage is not supported`）。
   权重准备必须提到捕获之前做一次并闭包引用。

   **2026-09-29 连续 decode 状态轨迹已通过**：新增 `dsv4_hca_trajectory.py` 与
   `--trajectory-steps N`。每步 `positions += 6`、`seq_lens += 6`，用 fixture 保存的
   `BlockTable` 重算 `slot_mapping`，再用 Native builder 原地重建 metadata 并校验地址不变；
   两侧从同一初态出发、每步看到完全相同的输入（同 seed 生成的 hidden + 同一套推进后的
   metadata），各自在自己的 cache 上累积。

   **范围**：这是同输入下的 cache/state 轨迹一致性，**不是 token 轨迹验收**——真实轨迹
   第 k 步的输入取决于第 k−1 步的输出，那需要整模型（剩余事项第 1 项）。

   | 档位 | 步数 | 输出 max_abs | growth_ratio | 输出 RMSE 首→末 | 压缩 cache 累积误配 |
   | --- | ---: | --- | ---: | --- | --- |
   | h124 / B4 | 48 | 0.0156～0.0313 | 2.0 | 0.0020152 → 0.0022025 | 11 → 459 |
   | h8190 / B4 | 24 | 0.0156～0.0313 | 1.0 | 0.0018963 → 0.0020428 | 7 → 215 |

   判定依据：输出 max_abs 全程钉在 BF16 的一到两个 ULP，48 步不增长（`growth_ratio = 2.0`
   只是一个指数档）；RMSE 48 步只涨 9.3%，不是指数累积；压缩 cache 的累积误配随压缩事件
   线性增长（每次事件约 +200，48 步跨 3 次事件到 459），无跳变；每步写保护 PASS、
   无非有限值；滑窗与滚动 state 页在 48 步内已多次回绕。
   证据：`results/hca_trajectory_20260929/h124_b4_s48_wide/`、`h8190_b4_s24_wide/`。

   **必读的 harness 前提（踩过三次，都是搬用既有构造时没核对它的前提）**：

   1. **页表宽度必须按轨迹终点定**。`make_fixture` 按 `history` 算列数，SWA 在
      history=124 下只有 6 列、覆盖 position ≤ 191；轨迹推进会让 `pos // block_size`
      越界读到错误物理页，表现为"发散"——越界版 48 步的输出 max_abs 涨到 **1.414**、
      `growth_ratio` 90.5，压缩 cache 误配出现 11→1091→3079 的阶跃，
      极易被误读成压缩器的数值累积。修法是传
      `table_history = history + 6 * steps`（该参数本是为 metadata A→B→A 加的）。
   2. `make_call` 内不能调 `prepare_weights()`：内部 `scale()` 做
      `bool(count_nonzero(offset).cpu())` 是同步 D2H，图捕获期间会被拒
      （`error 107030` / `Not_Supported(EE1016)`）。权重准备要提到捕获之前。
   3. 轨迹每步推进后必须重取 `fixture["readonly"]` 快照：只读检查要验证的是
      "本步内算子没有改动 metadata"，而不是"metadata 等于第 0 步"。

   **2026-09-29 请求生命周期与 prefix 共享已通过**：新增 `dsv4_hca_lifecycle.py`
   与 `--lifecycle`（需 batch ≥ 2，单独进程）。两个阶段各有精确判据，都不需要第二份 fixture。

   **阶段一 · 页复用**：把请求 0 重置为 history 0 但保留其原物理页，并把这些页里超出新
   有效范围的行填 NaN——算子若读陈旧行，NaN 会传播出来，污染自身就是 oracle。
   **同时用 Native 在同一份污染快照上跑一次作对照**，判据是"Native 有限而 PTO 非有限"才算
   PTO 缺陷。

   | 档位 | Native 非有限 | PTO 非有限 | 判定 |
   | --- | ---: | ---: | --- |
   | h124 / B4 | 0 | 0 | PTO 不读陈旧行，通过 |
   | h8190 / B4 | 98304 | 98304 | 两侧相同 → **前提不成立**：生产路径依赖"分配新请求时清页"，本阶段不作 PTO 判据 |

   后者若没有 Native 对照就会被误报成 PTO 缺陷（98304 正好是请求 0 的 6×4×4096）。
   两档的"其余请求输出与基准逐 bit 相同"与写保护均 PASS。

   **阶段二 · prefix 共享**：先把请求 0 的页内容复制进请求 1 的页（内容相同、各自独占）
   跑一次记下输出，再把请求 1 的**只读前缀列**指向请求 0 的同一批物理页再跑一次，
   要求输出**逐 bit 相同**。若存在任何假定物理页独占的读写路径（就地改写、按页号去重、
   写回不检查共享），输出必然变化。实测两档均 PASS：
   h124 共享 swa 3 / compressed 0 / state 15 列，h8190 共享 swa 3 / compressed 1 / state 15 列。
   证据：`results/hca_lifecycle_20260929/h124_b4_v3/`、`h8190_b4_v3/`。

   可共享列的判法必须**从 `slot_mapping` 反查本步实际写入的物理页**，页表里命中这些页的列
   一律不可共享。不能按几何推算：`state` 的 view 是 `[pages, page_elements]`，
   `views[0].shape[1]` 是页内元素数（1024）而不是行数，按它算会把整行都当成可共享、
   让两个请求写同一物理行，表现为"prefix 共享改变了输出"——那是测试设定错误，不是算子缺陷。

   **第 2 项至此全部完成**：动态 BS 切换、整算子 padding/dummy、连续 decode 状态轨迹、
   请求生命周期与 prefix 共享。
4. 单层独立的 metadata A→B→A 地址固定重放**已完成**（2026-09-28 23:0x，CANN 9.2.0-beta.2）：
   `results/hca_optimization_20260928/metadata_replay_cann92/`。B4、history A=124 / B=8190、
   统一页表宽度、B 的页表行反序；54 个 metadata 张量叶子中 23 个在 A/B 间指针与内容都不同。
   A→B→A 三步每步图重放与 eager 逐 bit 相同、写保护按目标状态自身 slot 计算且 PASS、
   输出无非有限值；B 另与在其自身张量上的直接调用逐 bit 相同，第二次 A 与第一次 A 逐 bit 相同。
   脚本 `dsv4_hca_metadata_replay.py` + `run_hca_metadata_replay.sh`。
   每步都从同一初态开始，因此它不覆盖连续多步 decode 轨迹（仍属第 2 项）。
5. 用户随后要求开始性能优化，目标七档全面超越 Native 20%，并明确先单卡再整机。
   已冻结含 KV 精度修正的基线，首项候选将 raw KV 逐行读取改为页内连续搬运；CPU 编译通过。
   单卡长短 B16 的计时/泳道已排队，未预记收益；单卡通过之前不提交优化版整机任务。
   [优化记录](results/hca_optimization_20260928/README.md)。

大文件 `states.pt` 和 `build/` 留在本地；报告与日志位于上述相对目录，不计算 hash。


## 2026-09-28：性能候选继续按单卡先行

优化台账见 [HCA 性能优化记录](results/hca_optimization_20260928/README.md)。
页内 raw KV 搬运与 PV N128 已通过长短 B16 旧版精确对照；O 反量化+mHC post 合并已修正
首版列提取错误，B1/H0、H124 及长短 B16 精确对照通过。合并收尾改为 16 行、48 Vector 块后，
同卡 ABBA 的长短 B16 PTO P50 分别下降 3.06%/3.32%，同期 Native 下降 1.05%/2.30%，
仅作小幅收益候选，尚未七档通过。统一 softmax/PV 暂无收益，生产 attention 已恢复。
Q_B 整组启动短档慢 1.01%、长档快 0.52%，不保留；压缩整组启动对照因卡 9 初始化超时不完整。
短历史 attention 流水融合与 mHC pre 消费者融合两个独立候选已通过 CPU 完整编译，
仅在测试快照中，正排队做单卡旧 PTO 精确对照/计时；保持两路 softmax 与 mHC 归约顺序。
生成代码已确认 mHC 候选去掉独立归约与 pre/post 门控任务，设备结果未出，不预记性能。
所有优化版整机任务仍未提交，七档单卡门禁优先；20% 目标尚未达到。


## 2026-09-28 19:19：等待单卡资源的恢复点

连续三轮目标工作均核实到同一外部资源限制：所有 16 张 NPU 被其他整机作业占用，
本会话的两个单卡任务仍为 pending。CPU 完整编译与候选源码复核已完成，
下一步需要设备结果，不能据 CPU 编译继续宣称数值或性能通过。
目标暂记为资源受阻；不取消队列作业，也不改动其他会话任务。

恢复时先查询以下现有句柄，不因等待时间长而重复提交：

| 候选 | 任务句柄 | 结果目录（相对 results/hca_optimization_20260928） |
| --- | --- | --- |
| 短历史 attention 融合 | `task_20260928_190150_994132305` | `short_fused_h8192_b16` |
| mHC pre 消费者融合 | `task_20260928_190950_35744614087` | `hcpre_fused_h8192_b16` |

另有 `source_short_pipeline` 双槽流水候选，`short_pipeline_build/report.json` 完整 CPU
编译通过，尚未提交 NPU。根据第一版融合的精确对照和泳道再决定下一项设备工作。
生产实现保持 `source_opost16`，未混入上述三个候选。七档单卡及优化版整机尚未验收；
全面领先 Native 20% 的目标未达到，恢复后仍按单卡先行流程推进。


## 2026-09-28 19:50：单卡工作已恢复，覆盖此前等待检查点

19:35 左右资源释放后，短历史融合、双槽流水、mHC pre 对齐修正均完成 B16/H8192
单卡旧 PTO 精确对照，输出和三份完整 cache/state 全部相同，服务图和保护区通过。
其同轮 Native/PTO P50 分别为 592.23/581.25、579.59/579.19、566.85/595.45 μs。
这些独立轮次不能证明候选间性能提升，也没有达到领先 Native 20%。

mHC pre 首版 UB 未对齐故障已在候选修正；只结束了本会话卡 13 的失败挂起作业，没有重置卡。
长历史统一流水首版输出门禁失败，已按生成代码修复 sink 与循环最大值的别名覆盖，
修正版 CPU 编译 PASS，待单卡结果。生产树仍保持 source_opost16。

恢复时查询：

- 长历史 sink 修正版：`task_20260928_194732_15942363707`，`unified_sink_h131072_b16`。
- 短历史+mHC pre 同卡组合 ABBA：`task_20260928_194836_16190322987`，`short_hcpre_abba_h8192_b16`。
- O post 残差 Tile 复用候选：`source_opost_reuse`，CPU `opost_reuse_build`，尚未设备测试。

所有路径位于 `results/hca_optimization_20260928`。详见该目录 README，含失败证据和限制。
先完成有收益候选的边界与七档单卡，再做优化版整机 token/性能；整机尚未提交。


## 2026-09-28 20:05：短档组合通过边界，七档单卡排队

`source_short_hcpre`（短历史 attention 第一版融合 + mHC pre 对齐修正）的 B16/H8192
同卡 ABBA 完成：旧 PTO→候选 P50 609.01→568.80 μs，下降 6.60%；Native 同期下降 0.52%。
B1/H0、H124 的精确对照、毒化、服务图和保护区通过。组合尚比同轮 Native 慢约 2.14%。
七档单卡任务、源码与命令见 `results/hca_optimization_20260928/short_hcpre_seven/tasks.json`，
不要重复提交。所有计时都在输出及状态门禁之后，尚未提交优化版整机。

另外两个独立候选正在定位功能问题，不得混入七档组合：

- 长历史统一流水：sink 修正后仍有明显差异，attention packed 诊断也失败；
  压缩 PV 诊断 `task_20260928_200304_201669631260`，`cmp_debug_h131072_b16`。
- Q_B 与归一化/RoPE 融合：每八行首行相同，其余不同，连续 RoPE Tile 未解决；
  Q 诊断 `task_20260928_200227_20078135798`，`q_debug_h8192_b16`。

O post 残差 Tile 复用功能通过但没有性能收益，不扩测、不合入。
生产树保持 source_opost16；完整失败证据、结果路径及限制见优化记录 README 最新章节。


### 20:08：Q_B gather 行偏移修正与当前等待点

只读复核 `rope_prepare` 确认其输出是每行 0..63 的列索引，原 Tensor gather 按行处理；
新候选误将其直接传给展平索引语义的 Tile gather，漏加 `行号×64`，因此每八行只对第一行。
`source_qb_fused_rows` 在连续 64 列 RoPE Tile 上构造完整展平索引，
`qb_fused_rows_build/report.json` CPU 完整编译 PASS。
单卡整层精确对照和计时为 `task_20260928_200633_210859118643`，
目录 `qb_fused_rows_h8192_b16`，尚未设备验收，不能预记问题已经全部解决。
先前 Q 中间量诊断 `task_20260928_200227_20078135798` 仍 pending 时已取消，
避免同一缺陷再跑两轮诊断；其源码留存。压缩 PV 诊断与七档单卡任务仍保留。

当前 16 张卡由其他整机作业占用，本会话等待既有单卡队列，不干预其他任务。
下轮先查 `short_hcpre_seven/tasks.json` 的七个句柄、压缩 PV 诊断
`task_20260928_200304_201669631260` 和上述 Q_B 修正版；不要重复提交。
生产实现仍未引入这些待验证候选，目标尚未完成。


## 2026-09-28 20:25：继续优化，长档归约结构修正

长档融合找到具体 UB 复用问题：selected_m/m_after 与后续 PV Tile 同在偏移 94464，
循环尾回写最大值时读到 PV 内容。`source_unified_attention_long` 简化长档归约分支后
CPU 编译通过，next_m 与 PV 地址分开；地址证据 `unified_loop_storage.json`。
新设备任务 `task_20260928_202310_259903926056`，`unified_long_h131072_b16`，尚未通过。
此源码仅用于长档候选，不替换短历史路径，生产树仍未变化。

旧压缩 PV 诊断 `task_20260928_200304_201669631260` 在 pending 状态取消。
恢复时保留查询的任务为：七档 `short_hcpre_seven/tasks.json`、Q_B 行索引修正
`task_20260928_200633_210859118643`、上述长档修正。
七档汇总命令为 `python tests/pypto_test/summarize_hca_seven.py \
  tests/pypto_test/results/hca_optimization_20260928/short_hcpre_seven`。
所有相对结果路径位于 `tests/pypto_test/results/hca_optimization_20260928`。

## 2026-09-28：HCA 独立分支提交范围

本次将现有 `dsv4-flash-hca-pto-v0.25.1rc1` 分支推送到
`nalinaly/vllm-ascend`，不另外创建日期分支。该分支继承 CSA 的 `65500b91`，
HCA 相关公共函数调整也保留在本分支；之后合并前仍需补必要的 CSA 回归。

提交的生产算子对应 `source_opost16`：包含有效压缩 KV 读取边界修正、
raw KV 页内连续搬运、PV N128 双累加和 O 反量化与 mHC post 融合。
短历史 attention 与 mHC pre 组合、Q_B 融合、长历史统一流水仍是本地实验候选，
没有因为发布分支而替换生产实现。

验证脚本、交接记录及小型 JSON/Markdown 报告进入 Git。报告保留失败和待验证记录；
首轮七档性能报告早于 KV 精度修正，不能作为当前版本的验收结果。
8K/B16、DP/EP16 的 65536 个输出 token 一致结论来自性能优化前基线；
当前性能改动已完成的单卡验证范围见优化记录，尚无优化版整机验收结论。
本次仅整理提交，不新测、不计算文件 hash。

以下路径相对当前工作目录，大快照、编译产物和实验源码继续保留在本机：

| 本地路径 | 内容与用途 |
| --- | --- |
| `tests/pypto_test/results/hca_compressor_20260928/` | C128 子链状态快照、编译产物和原始运行记录 |
| `tests/pypto_test/results/hca_layer_20260928/` | 整层及服务图快照、编译产物 |
| `tests/pypto_test/results/hca_performance_20260928/` | 首轮七档原始 profiling、泳道、完整状态和编译产物 |
| `tests/pypto_test/results/hca_token_20260928/` | EP8/EP16 逐 rank 原始输出、毒化检查状态和编译产物 |
| `tests/pypto_test/results/hca_optimization_20260928/` | `source_*` 冻结源码、各轮原始计时/泳道、状态和编译产物 |

正式权重仍使用 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，不进入 Git。
恢复实验时使用优化记录中的已有任务句柄与源码路径，仍按先单卡、再整机的顺序推进。


## 2026-09-28 21:55：wo_a 改 ND、冷 L2 归因与 kernel 模式开销

**口径与范围更新**

- 分支已摘入 CSA 的 `1f7bad79`（wo_a 恒为 ND）与 `715e504d`（ring arena 可配置）。
  mode=2 下 Native 的三维 wo_a 仍是 ND，HCA 此前按 NZ 声明会另存每层 64 MiB 私有副本，
  且违反 CSA §144 的同口径要求。HCA 自有 `decode_hca.py`、`o_proj_hc_post.py` 已改为
  `WO_A_WEIGHT_LAYOUT`（工作区未提交）。新的公平基线为 `source_prod_woa`，旧 NZ wo_a 的
  七档数字仅作历史参考。
- 用户明确：HCA 范围内的调度与 incore task 都在优化范围内；本仓 `pypto/`、`simpler/`
  代码也在范围内（取代本文开头"不修改依赖仓库"的旧约束）。NZ 权重相关做法先对照
  `DSV4_FLASH_CSA_VALIDATION_LOG.md` 中 CSA 已有结论；DLPack 别名方案暂不做。

**已完成的单卡结论**（全部对旧 PTO 快照逐 bit 一致，详见优化记录 21:00～21:25 各节）

- 短档组合七档（NZ wo_a 口径）与 Native 大致持平；长档统一流水修正版精确通过。
- 冷 L2 泳道（`--swimlane-cold-l2`）与"Native 之后"泳道（`--swimlane-after-native`）
  已加入测试入口；正式交替计时处于两者之间。
- 候选取舍：Q_B N128 流水、proj_b/proj_a 加深流水、proj_a N256、wo_a Vector 预热均无收益或
  退化，不合入；widen+RMS 合并（CSA d1f170ff）、QR 输入/gamma UB 复用（CSA 56186de9）、
  KV/compressor ND 权重预热组成 `source_combo_v8`，同卡 ABBA 相对 v4 扣除 Native 漂移约
  −1.5%～−2.5%，仍未达到 20% 目标。

**kernel 模式固定开销探针**（`dsv4_pto_launch_floor.py`，同一 custom op + NPUGraph + 图外 Event）

| 形态 | P50 μs |
| --- | ---: |
| 1 任务 × 1 块 | 79.7 |
| 32 串联单块任务 | 151.6 |
| 8 任务 × 48 块、无依赖 | 171.9 |
| 75 任务 × 8 块、无依赖 / 串联 | 476.2 / 473.2 |

ring task window 1024～16384、heap 64～256 MiB 均不改变约 80 μs 的单次固定开销。
HCA 整层约 76 次提交、50 个 worker 任务、650 个块，与探针同一量级；8K/B16 的约 576 μs
可近似拆为固定约 80、启动约 17、关键链阶段边界约 85、核内工作约 394 μs。
固定开销的具体归属（AICPU init/握手/收尾或 AICore 侧）尚未实测拆分，已向用户确认方向，
不据此修改运行时。下一步先在 HCA 算子侧减少任务数与关键链阶段边界。

恢复时：候选源码与结果均在 `results/hca_optimization_20260928/`（`source_combo_v1`～`v9`、
`abba_*`、`launch_floor/`）；统一入口 `run_hca_candidate_case.sh`（精确门禁+计时、冷泳道、
Native 后泳道）与 `run_hca_abba.sh`。七档单卡及优化版整机仍未验收，20% 目标未达到。


## 2026-09-28 23:00：工具链切到 CANN 9.2.0-beta.2，验收分母需重取

另一个会话在 **21:59:13** 把公共入口 `env-dsv4-0251rc1.sh` 改为 source
`cann-9.2.0-beta.2/set_env.sh`（原 `/usr/local/Ascend/cann-9.0.0`），并清理了旧
CANN 路径混用、修复 profiler 属主检查；Native 自定义算子包仍是当前 release 版本。
`run_hca_single_layer.sh` 会 source 这个入口，因此**任务按启动时刻继承版本**：
21:59 之前的取证（七档基线、`source_prod_woa` 对照、v1–v9 候选、`incore_v8_*`、
kernel 模式 launch floor 探针、已交付的 `hca_h131072_b16_profiles/`）都是 9.0.0；
之后排队的都是 9.2.0-beta.2。核对方式是看结果目录 `build_output`／日志里的 CANN 路径，
不能看当前 shell 的 `ASCEND_HOME_PATH`——已在运行的 shell 仍是旧值。

同一份 v8 代码换版本后：8K Native/PTO 551.6/597.5 → 558.3/597.5，
128K 684.7/685.8 → 703.4/708.4 μs。两侧都略慢，PTO/Native 比值几乎不动
（8K 1.083→1.070，128K 1.002→1.007），**七档 ≤0.8×Native 的目标在 9.2 上一样远**。

对验收的影响，按"同一次对照内部版本是共同项"划分：

- 保留：每个 ABBA 的相对结论。v10（Q_B 按 K256 双缓冲、M128 覆盖有效行）与
  v11（把 `hca_raw_valid`、`hca_inverse_rope_sign` 推迟到 Q_B 窗口，KV 预热 24→8 块）
  各自逐 bit 相同且都有收益，已合成 `source_combo_v12`（两者改动文件不相交，逐文件校验来源）。
- 重取：**七档同卡 Native P50 是验收分母，必须在 9.2 上重测**；incore 逐功能对照表
  是"哪个 task 落后 Native"的判据，9.2 可能改了 MatMul/TransposeBatchMatMul 实现，
  Q_B 与 O_A 这两个落后项要在 9.2 上复核后再继续投入。

本轮在 9.2 上排队的 12 个任务：v12 长短 ABBA
（`task_20260928_225814_24993186253`、`_249937130041`）、
七档 `source_prod_woa` vs v12（`seven_cann92_v12/tasks.json` 记 7 个句柄）、
v12 长短 incore 取证（`task_20260928_225859_25295299731`、`_252961228505`）、
metadata A→B→A 重放（`task_20260928_225934_25611505991`，对应剩余事项第 3 项）。

新增 `submit_hca_seven_tier.sh`（2026-09-29 已更名为 `submit_hca_tiers.sh`、改为六档）：各档各占一张卡并行提交。注意本机 `task-submit`
只接受一整条命令字符串，按 argv 传会被当成自身选项，任务立刻 `exit=2`、
日志里命令显示为 `bash --device N`（v12 首次提交踩到，已重投）。


## 2026-09-28 23:30：O 投影瓶颈定位与计时口径纠正

用户要求参考 CSA 算子按 shape／档位分别选核内 tiling。查证后确认 CSA 验证日志第 175 节
v3 的 O projection 自适应分档（O-A N128/N256、O-B M32/M96/M128、大档位 O-B 权重完整 K
常驻、不重排 Native 权重）在 HCA 侧已经是 `decode_o_proj_tp1` 的逐字拷贝，CSA 唯一没有
覆盖的维度是 proj_a 的行块 M。

按此排的两个单变量候选：v13（proj_a 行块按档位贴合）**无效，已丢弃**——PyPTO 的 cube
本来就按 `valid_shape` 只算有效行；v14（proj_a 列块 N128→N256）跨度没变，但 incore 证实
核时间从 2624 降到 2048 μs，折算 24 个 AIC 为 85.3 μs，已追平 Native 的 86～87 μs，
只是块数 64→32 使填充率 89%→67%，把收益吃掉。瓶颈因此定位为 **ND 的 wo_a 按列切片时的
跨步读**：N128 每次仅 256 字节连续、跨步 2048 字节，达成约 519 GB/s；N256 为 778 GB/s。
后续 v15/v16 用 N256 + 3 行块凑满 96 块 / 4 波，并把 wo_a 的 `CachePolicy.BYPASS` 改回
默认以承接行块间的重复读（v16 保留 BYPASS 作对照）。

**计时口径纠正，影响此前所有"基线→候选"差值的读法：**
`run_hca_reference_pair.sh`（七档使用）先跑基线、后跑候选两个独立进程，候选总在更热的卡上
测量，因此该差值系统性偏向候选，不能当作收益；两次七档之间也不可比（同一份 Native 代码
在 8K/B16 上一次 583.9、一次 544.8，差 39 μs）。七档里可信的只有同一轮内测出的
PTO/Native 比值。只有 ABBA 的 A→B→B→A 能抵消单向漂移，并且要再扣掉 Native 在基线轮与
候选轮之间的漂移量（Native 四轮代码相同，其变化量就是漂移）。按这个口径，v12 相对 v8 的
净收益是 8K −12.2 μs、128K −16.1 μs（约 2%），此前只给比值的说法偏乐观。

当前最佳仍是 `source_combo_v12`（= v8 + v10 Q_B 分块 + v11 准备任务延后），七档逐 bit
门禁全 PASS，同轮 PTO/Native 比值 0.941～1.072、均值约 1.008，距 0.80× 仍差 83～190 μs。
即使把 kernel 模式约 80 μs 的公共启动开销整个减掉也只到 0.80～0.93×，缺口主要在串行关键
链本身：O 投影 156、attention 141、Q_B 54 μs 三项占一半以上。


## 2026-09-28 23:45：wo_a 布局必须跟随 Native；新增 128K/B24 档位

用户要求**性能测试优先 vllm_ascend 的 nz_mode=2**，并询问是否可以开启 128K/B24。
两件事汇到同一个根因上。

**根因。** 候选快照用 `--operator-source` 重指 `vllm_ascend.ops.pypto.__path__`，连带冻结了
共享的 `deepseek_v4_flash_dspark/nz_mode.py`，里面是 `WO_A_WEIGHT_LAYOUT = None`（写死 ND）。
`wo_a` 是三维分组权重，Native 侧存 ND 还是 NZ 取决于当前 CANN 的 `npu_format_cast`
支不支持三维 BF16：9.0.0 不支持（ND），9.2.0 支持（NZ）。在 9.2 上写死 ND 会让
`native_adapter.root_weight` cast 出私有副本，每层 78.18 MiB、21 层合计 **1.603 GiB**。
生产版已改为 `_native_keeps_3d_bf16_as_nz()` 导入期探测，并在不匹配时打
`PTO_CSA_WEIGHT_RECAST` 告警（此前完全静默）。

**影响一：128K/B24 的决定性前提。** PTO 此前在 128K/B24 上差 Native 1.63 GiB，
与这 1.603 GiB 几乎完全吻合。所以布局修正不是优化项，是该档位能不能跑的前提。

**影响二：作废了本会话两轮 proj_a 结论。** 我在 9.2 上量到的 proj_a 一直是 recast 副本，
"ND 列切片跨步读 519 GB/s"对副本成立，但生产路径上 NZ 列切片本身连续。据此排的
v13（行块贴合）、v14（列块 N256）、v15/v16（N256+3 行块±L2）全部否掉，v15/v16 反而差
39～52 μs。**教训与既有约束"对照 Native 路径、不要自己发明判据"一致：布局不是可选常量。**

**v17 = v12 + 生产版共享文件**（只刷新 1f7bad79 / 715e504d / 1a88c7b4 动过的 `ops/pypto`
文件），唯一实质变量是 wo_a 布局跟随探测。ABBA 净收益：8K/B16 −6.98、128K/B16 −41.07、
128K/B24 −15.26 μs。proj_a 核时间从 2624 降到 1408 μs（−46%），整个 O 投影组由落后
+23.5 μs 转为 8K −1.9、128K −18.0（领先 Native）。整层 DFX 跨度 8K 563.1→498.6 μs。
所有 v17 运行的 `PTO_CSA_WEIGHT_RECAST` 告警为 0。

**八档验收表（CANN 9.2，逐 bit 门禁全 PASS）**：128K B4/B8/B16/B24 = 0.993 / 0.930 /
0.946 / 0.943；8K B16/B24/B32/B40 = 0.983 / 0.961 / 0.965 / 1.000。比值均值 0.965
（v12 时 1.008），八档全部 ≤1.0，距 0.80× 仍差 70～165 μs。
**128K/B24 单卡已跑通并纳入验收表**；按用户要求，整机验证通过后再去掉 8K/B16 的 case。

### 下一步（整机 128K/B24 的前置条件）

1. 按 **HCA 自己的形状**取一遍 ring 下限：`additional_config={"pto_csa_ring_config":
   {"heap_mb": [...], "task_window": ...}}` 可配 arena 尺寸（默认 4×256 MiB heap +
   16384 深 task window = 1.343 GiB）。**不要照抄 CSA 的 `[256,128,256,32]` + 4096**，
   那是 CSA 形状上实测的；给小了不是变慢，是 dispatch 期 Task Allocator Deadlock
   整进程退出。做法：故意把某条 ring 调小，读
   `<output>/ascend/debug/device-N/device-*.log` 里的
   `Heap ring N: used=… / Requested: … bytes`，加余量后使用。
   HCA 分支有两个 `pypto.torch.init` 调用点（`prepare_csa_model` 与
   `PyptoHCADeepseekV4ForCausalLM.process_weights_after_loading`），两处都已接上。
2. 16 卡 128K/B24 整机 token 验证，用 HCA 自己的 ring 配置 + `gpu_memory_utilization=0.97`
   （CSA 侧在 9.2 上连跑三次 24/24 并发、0 抢占）。
3. **9.2 口径的 Native B24 基线尚不存在**（现有 22.90 GiB / 88 727 us 是 9.0 的），
   整机档期要同时跑 Native 才能给出可比对照。
