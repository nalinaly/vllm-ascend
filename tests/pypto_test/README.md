> **2026-09-23 基线已迁至官方 v0.25.1rc1。** 当前入口为[基线迁移说明](BASELINE_MIGRATION_V0251RC1.md)。
> 下方历史 PASS 对应旧基线；本次 release 尚未进行真机数值验收。历史脚本需按新接口适配。

# DeepSeek-V4 Flash CSA验证入口

先读仓库根目录 [DSV4_FLASH_CSA_HANDOFF.md](../../DSV4_FLASH_CSA_HANDOFF.md)。
本目录包含截至2026-09-22的验证代码和证据；Native存储适配已进入完整单层对照，服务forward派发待接入。
TP1/S=6计算入口已位于本仓库`vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/`，
当前最新证据为`results/cann90_20260921/adapter_redundancy_v4/`：两组compressor直接读写
Native物理state，四次state gather/commit和两份14行搬运窗口已删除。
Native state及各自页表/slot通过device Tensor传入，实际页间距由零拷贝view保留；
原pool读取历史，原write任务在pool依赖满足后直接写Native有效行。
B4/history131071、B40/history131073完整比较各18项通过；B4同图A→B→A共90项通过；
B4 mixed10的eager10+graph10共370项通过，跨步cache/state保留并使用Native接受修正。
两档输出、Top-K和分数与v3逐bit相同，state及页表的四项零拷贝断言通过。
两份graph profiler均确认1次PTO提交（1个Simpler AICPU入口和1个AICore kernel_mode入口），
NativeCSACall只调用一次完整CSA，外部PTO适配调用已全部消除，详见日志第66节。
Native metadata生产和ExternalEvent等待继续由原调用方管理；服务forward派发仍待接入。

此前v3已删除两次compact展开调用，消费者直接读取Native紧凑slot/cos/sin；
六份token级slot/频率展开和两份内部signed-sin缓冲删除，已有CSA任务计算两份INT32[B]偏移。
v3的B4/B40完整比较和B4图重放通过，两档输出/Top-K/分数与v2逐bit相同，profiler为5次提交。

此前v2已删除独立token metadata调用，positions、三组slot和普通KV页表直接使用Native
device Tensor；B4/B40完整比较与B4图重放通过，输出/Top-K/分数与v1逐bit相同，profiler为7次提交。

此前`adapter_redundancy_v1/`去掉普通RoPE往返重排及未使用的滑窗长度，
参考18组矩阵共324项通过，输出、Top-K和分数与`reference_matrix_v5`逐bit相同。
v1的B4连续eager100+graph100及六项G08通过；B8按用户控制测试范围的要求主动停止，
只记录已完成eager100+graph91步。B16/B40复现原有输出失败，已观察步的指纹均与旧版一致。
这些18组和100步轨迹证据属于v1及之前版本；v2/v3各三项、v4四项针对性验证，不外推为完整回归。
pypto-lib仅作只读参考。
CPU检查入口为`dsv4_csa_reference_lower.py --output-dir <结果目录>`，使用下述环境运行。
Native/PTO对比profile及PTO泳道图见`results/cann90_20260921/csa_profile_v1/README.md`。
Indexer页加载优化后的图与前后比较见
[`indexer_page_load_v1/README.md`](results/cann90_20260921/indexer_page_load_v1/README.md)：
B40 score leaf从28.17ms降至5.70ms，完整CSA三次profile中位数从31.15ms降至13.52ms；
B4/B40共36项检查通过，输出/Top-K/scores与优化前逐bit相同。
B40/S6/history131073、同一logical device8、各3次单层ACL Graph采样已完成；
设备跨度中位数为Native2473.26微秒、PTO31145.62微秒（含Native metadata生产）。
两份profile关闭DFX；泳道图另开进程采集一次eager CSA，含真实核函数名和依赖边。
`native_adapters`当前仅检查Native普通RoPE和state/页表/共享Indexer存储的零拷贝描述符；
旧版state窗口写回结果属于历史证据，当前state读写和保护区由完整CSA用例验证。
`full_compare`执行参考权重的完整Native/PTO单层对照；正式目标权重仍待验，P2整体保持PENDING。
`full_replay`是完整计算链的A→B→A图重放诊断，B4/B8/B40真机通过，含Native外部事件与完整存储对照。
`continuous`执行保留cache/state的eager及graph同轨迹对照；当前B4/B8 mixed100及B4
all1/all6/reject_then_accept均各eager100+graph100通过，每项3700检查。
B16/24/32在step63、B40在step46仅输出超差，Top-K、cache/state与保护检查通过；继续定位稀疏attention。
`--diagnostic-step`、`--query-diagnostic-step`及`--output-diagnostic-step`复用生产函数采集对应边界。
复现必须保留原`--steps`：该值影响缓存容量和初始化，25步配置的PASS不能替代100步。
`dsv4_csa_cross_stream.py`复用完整重放测试，对Native三种生产stage分别增加设备计算延迟；
B4/B40六组全部通过，最大延迟submit返回时生产仍未完成，见`full_cross_stream_v4/`。
其余连续轨迹及G04～G07等覆盖待验，P3整体仍未通过，详见日志第60～62节。
正式目标为`/data/model/DeepSeek-V4-Flash-0731-w8a8`（75分片，下载中），
下载完整后须使用正式ModelSlim加载路径；不能套用旧参考loader。

当前已迁至CANN9.0、16卡机器，使用工作区`../env.sh`及Python3.10环境。
新结果在`results/cann90_20260921/`，进展与失败原因见验证记录第17节起。
当前构建与兼容选择见[CANN9.0环境记录](handoff/ENVIRONMENT_CANN90.md)。
本机硬件任务必须通过`task-submit`；下文旧机器的直接Python命令仅保留为参数参考。
当前入口为`run_csa_validation.sh`，从队列的`TASK_DEVICE`取得设备号，例如：

```bash
source ../env.sh
task-submit --device auto --max-time 600 \
  "bash $PWD/tests/pypto_test/run_csa_validation.sh native_layout $PWD/tests/pypto_test/results/cann90_20260921/layout_run /data/model/dsv4-flash-0731-dspark-w8a8"
```

命令返回任务ID；用`task-submit --status <ID>`查看状态，`--log <ID>`查看日志。
异步提交不因客户端等待超时取消排队。其他单卡阶段使用`native_ops`、`metadata`或
`native_forward`，需先完成配套custom包安装。双卡阶段使用`--device auto --device-num 2`
及`dp_metadata`；入口通过`ASCEND_RT_VISIBLE_DEVICES`将获分配的两张卡映射到逻辑0、1。
`run_kernel_runtime_validation.sh`以同样方式验证调试分支的Torch eager/graph/退出清理。

## 1. 文件与实际覆盖范围

| 文件 | 作用 |
| --- | --- |
| `dsv4_csa_scatter_offset.py` | Native负slot、非零offset、保护区回归；`scatter_offset`阶段 |
| `dsv4_csa_precision_kernels.py` | 完整链失败后的QR/Indexer诊断边界，复用同一参考计算函数 |
| `dsv4_csa_qr_boundary.py` | `qr_boundary`轻量诊断：已捕获Native QA进入同一生产QR归一化函数 |
| `dsv4_csa_env.py` | 测试进程的源码选择、editable hook、隔离C++扩展加载 |
| `dsv4_csa_inventory.py` | 环境与checkpoint headers初始盘点；其早期输出不是最终通过状态 |
| `dsv4_csa_native_fixture.py` | 真实配置、TP/DP进程组、一层原生attention及五组cache owner |
| `dsv4_csa_native_layout.py` | 真实权重加载/哈希、原生布局及ND检查 |
| `dsv4_csa_contract.py` | 支持条件、容量和metadata拒绝用例 |
| `dsv4_csa_metadata_kernel.py` | PTO device metadata读取及C4紧凑slot展开 |
| `dsv4_csa_transport_smoke.py` | 合成metadata的最小eager/graph/sentinel验证 |
| `dsv4_csa_shared_storage_kernel.py` / `dsv4_csa_shared_storage_smoke.py` | 共享K/FP16 scale、offset、padding探针 |
| `dsv4_csa_native_ops_smoke.py` | 原生metadata算子实调 |
| `dsv4_csa_native_builder_smoke.py` | 原生SWA builder |
| `dsv4_csa_native_transport.py` | 单SWA组Native→PTO transport早期矩阵 |
| `dsv4_csa_all_groups.py` | P1五组完整metadata矩阵、负例、graph及100步接受修正probe |
| `dsv4_csa_dp_metadata.py` | 两个真实DP rank的同步、dispatcher和非空rank五组probe |
| `dsv4_csa_native_forward.py` | 参考权重Native完整CSA预执行；QLI修复后B4/B40输出有限 |
| `dsv4_csa_native_adapters.py` | Native普通RoPE布局及state/页表/共享Indexer存储零拷贝描述符；实际读写由完整链验证 |
| `dsv4_csa_full_compare.py` | 完整参考TP1/S6链路的Native/PTO输出、cache/state及Top-K对照 |
| `dsv4_csa_full_replay.py` | 完整链的同图A→B→A；Native外部metadata事件、内容更新与eager/graph逐字节对照；`--profile`仅采样最后一次重放 |
| `dsv4_csa_profile.py` / `run_csa_profiles.sh` | 同卡、独立进程的Native/PTO完整单层ACL Graph profile及一次独立eager DFX窗口 |
| `compare_dsv4_csa_profiles.py` | CPU核对同条件/输入hash/输出、图执行次数和逐次设备跨度；用实际编译kernel_config补全泳道核函数名称 |
| `dsv4_csa_continuous.py` | G01/G02完整CSA接受轨迹；保留跨步cache/state，逐步Native对照及eager/graph存储指纹 |
| `dsv4_csa_compressor_boundary.py` | 连续轨迹失败后的Indexer压缩诊断，复用生产project/pool/write函数 |
| `dsv4_csa_query_boundary.py` | 连续轨迹失败后的QA、QR、Indexer query诊断，复用生产函数并保存Native边界 |
| `dsv4_csa_weights_boundary.py` | 复用生产head weights/系数函数，核对六档形状并可采集Native MatMul profiler |
| `dsv4_csa_output_boundary.py` | 捕获Native投影输入并送入同一生产O投影函数，区分投影与上游误差 |
| [PTO_ISA_A3_TDIV_HIGH_PRECISION_REPRO.md](PTO_ISA_A3_TDIV_HIGH_PRECISION_REPRO.md) | A3高精度TDIV支持范围、组件归因、Native标量路径对比及独立复现 |
| `repro_tdiv_high_precision.py` / `run_tdiv_high_precision_repro.sh` | 不依赖模型的默认TDIV、高精度选项和设备标量除法对照，4096个输入已真机复现 |
| `build_native_cann92.py` | 隔离ABI实验的构建脚本，历史参考，不是默认部署方案 |

## 2. 运行前提

以下是现有脚本真实使用的命令形式，不是自动配置新机器的安装器。
先按 [环境与构建记录](handoff/ENVIRONMENT_AND_BUILD.md) 准备PTO工具链、模型Python、固定vLLM和配套扩展。
测试loader默认寻找 `.cache/csa/native-install`；原始机器该目录不随Git传输。
若新机器采用已有安装扩展，先修改测试loader的选择并记录实际路径。

所有命令在仓库根目录运行。本机历史环境可概括为：

```bash
source /mnt/workspace/inductor/pto_eager/env.sh
source /home/developer/Ascend/cann-9.2.0/opp/vendors/custom_transformer/bin/set_env.bash
source .cache/csa/csa-native-ops-install/vendors/csa_validation_transformer/bin/set_env.bash

CSA_MODEL_PY=/mnt/workspace/inductor/vllm-ascend/.venv/bin/python
CSA_CHECKPOINT=/usr/.devenv/models/DeepSeek-V4-Flash-W8A8
CSA_OUT=tests/pypto_test/results/new_environment
```

上述三个shell变量仅用于替换下文命令参数；新机器使用其实际值。
本机还在每次启动前加过以下前缀解决aarch64 static TLS问题；新环境没有该错误时无需套用：

```bash
LD_PRELOAD=/home/developer/.local/lib/python3.11/site-packages/scikit_learn.libs/libgomp-a49a47f9.so.1.0.0${LD_PRELOAD:+:$LD_PRELOAD} \
  "$CSA_MODEL_PY" tests/pypto_test/dsv4_csa_native_layout.py \
  --checkpoint "$CSA_CHECKPOINT" --device 0 --output-dir "$CSA_OUT/layout"
```

## 3. 已有入口的复验命令

五组metadata和100步轨迹，不读取checkpoint权重payload，只使用模型配置和真实原生owner：

```bash
"$CSA_MODEL_PY" tests/pypto_test/dsv4_csa_all_groups.py \
  --checkpoint "$CSA_CHECKPOINT" --device 0 \
  --batches 4 8 16 24 32 40 --steps 100 --output-dir "$CSA_OUT/metadata_trace"
```

两卡协调入口由父进程启动两个worker，**使用逻辑卡0和1**，不启动整模型或16个rank：

```bash
"$CSA_MODEL_PY" tests/pypto_test/dsv4_csa_dp_metadata.py \
  --checkpoint "$CSA_CHECKPOINT" --output-dir "$CSA_OUT/dp2_metadata"
```

读取真实第2层W8A8权重、检查ND和layout：

```bash
"$CSA_MODEL_PY" tests/pypto_test/dsv4_csa_native_layout.py \
  --checkpoint "$CSA_CHECKPOINT" --device 0 --output-dir "$CSA_OUT/layout"
```

原生完整attention预执行，完整128K历史，未包含PTO对照：

```bash
"$CSA_MODEL_PY" tests/pypto_test/dsv4_csa_native_forward.py \
  --checkpoint "$CSA_CHECKPOINT" --device 0 --batch 4 --history 131072 \
  --seed 1024 --output-dir "$CSA_OUT/native_forward"
```

旧机器的Compressor stride失败已越过；当前CANN9.0发现的Native非有限值已定位并修复为
QLI metadata长度缓存未刷新。B4/131072、B40/131073完整Native参考fixture输出全部有限，
见验证日志第29～30节；尚无完整Native/PTO数值对照结果。
必要时在`native_forward`参数后加`--diagnostics`，输出`native_stage_trace.json`。
该诊断会同步和回读阶段输出，不用于性能测量或异步流验收。
`make_attention()`和`load_layer_weights()`当前专为本机cann_recipe导出的compressed-tensors
参考checkpoint构造。用户将提供vllm_ascend W8A8目标权重；到位后核对其描述并使用正式加载路径，
不能只替换checkpoint路径便认定等价，也不能将当前scale适配作为格式转换方案。

## 4. 结果解释

- 旧机器移交状态为 [handoff_state.json](results/local_20260921/handoff_state.json)；
  当前状态以[验证计划](DSV4_FLASH_CSA_VALIDATION_PLAN.md)及[实际记录](DSV4_FLASH_CSA_VALIDATION_LOG.md)为准。
- 文件夹v1/v2等为真实历史实验，不删除失败，也不把早期失败误认为后续仍失败。
- `native_groups_v4` / `native_trace_v2` 汇总中有历史字段
  `full_P1_status=PENDING_REJECTION_TESTS_AND_FORMAL_CASE_MAPPING`，当时脚本漏更新这个描述。
  实际negative_cases和正式映射已通过；新脚本字段已修正，但不重写旧JSON证据。
- 100步trace每步重置sentinel，仅验metadata；完整CSA跨步状态还没有结果。
- DP2报告中的world=HCCL、CPU group=Gloo是真实进程组；probe graph与模型graph分别记录。
- `native_forward_custom_v3` 是交接时最新原生失败；没有完整CSA性能结果。

## 5. 后续开发

按 [NEXT_STEPS_16CARD.md](handoff/NEXT_STEPS_16CARD.md) 完成P0/P2/P3/P4剩余项及P5。
主计划第9节列出的生产adapter、单层compare和完整replay文件是待实施设计，不能当成已存在功能。
继续维护专用 [验证记录](DSV4_FLASH_CSA_VALIDATION_LOG.md)，新环境使用新结果目录。

下载后可在仓库根目录核对交接文件字节内容：

```bash
sha256sum -c tests/pypto_test/handoff/SHA256SUMS
```

校验清单固定于本次交接，后续开发改动对应文件后不再要求与旧哈希相等。
原始日志、快照、patch和同事HTML按字节归档，保留其中原有拼写和换行，不作为拼写修复对象。
