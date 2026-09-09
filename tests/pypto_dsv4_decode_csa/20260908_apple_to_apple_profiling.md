# vLLM-Ascend CSA 同口径复现与 profiling

更新时间：2026-09-08 10:50:00

## 范围与交付约束

只检查 `torch.ops.vllm.dsa_forward` 的 native 与 PyPTO L1 替换，不运行或修改
recipes/SGLang。目标为 A3、TP1、B4/S8、C8191、TRB ACLGraph，完整 CSA，
输入/输出 `[32, 4096]` BF16，六类 InOut cache；HC-pre、输入 RMSNorm、HC-post
在比较边界外。本次先采集证据，不优化生产算法。

- 正式延迟：关闭 profiler/DFX，同一设备、同一进程、同一 caller stream 的 ABBA；
  每路 20 warmup、100 sample，每 20 enqueue 同步，至少三个 fresh process。
- 公平性：同 seed 的非零输入与权重、等价但地址独立的六类 cache、相同 metadata；
  两路均 ACLGraph replay。native 保留生产 multistream overlap。
- 校验：采样前、采样后均比较输出及所有 cache；编译、prepare、capture、数据准备、
  验证和 profiler 时间不计入正式延迟。
- profiling：独立诊断运行；native 必须有 device kernel 记录，不能用 host enqueue
  代替。PTO 必须标明泳道来自 L1 replay 还是 L2 诊断；二者不能混称。
- 交付原始 profiler/Perfetto 数据、可读时间线/泳道、汇总及复现命令。
  每张图保留微秒单位、真实空隙、采集模式和输入基线，不拼接为虚假的连续执行。

## 初始核验

1. 工作树：vLLM-Ascend `17574788`，PyPTO `3872b35d`，simpler `06c8b296`。
   grok 新增的 standalone runner、测试与 9 月 7 日结果未跟踪；原样保留。
2. 9 月 5 日报告写明 PTO 配对差中位数慢 11.130/15.004/18.949 us。
   “PTO 快约 13 us”对应 9 月 3 日历史结果，不能归到 9 月 5 日复测。
3. `a3_child_swimlane.py` 明确使用 L2、CPU 全零 fixture，并声明绝对时长
   不是 L1 性能。因此旧泳道不能冒充上述 L1 ACLGraph 样本。
4. 设备初查：逻辑 device0/1 对应 card7/chip0/1，AICore 均为 0%，
   npu-smi 未列出运行进程。此检查不能证明跨容器独占。
5. `task-submit` 不在 PATH。按现有流程串行运行并记录为 unlocked；
   不宣称获得硬件队列锁，也不做设备 reset。

## 可视化约定

主交付是 profiler 文件与泳道图，不是业务仪表盘。native 图展示 device kernel
的实际起止与多 stream 重叠；PTO 图展示可核实的 core/task 起止。优先保留原始
Perfetto 时间线，另导出静态预览。蓝/金两色加中性色，并以 lane 和文字区分类型。
若只能获得另一执行模式的诊断图，必须独立命名并写明不可用于严格延迟对比。

## 执行记录

### 09:49–09:51：三个独立进程复跑完成

原样运行 Grok 的 standalone 入口，`TASK_QUEUE_ENABLE=1`，逻辑 device0。
每进程每路 20 warmup、100 sample，ABBA，每 20 enqueue 同步。
下列都是关闭 profiler 的 device event span，单位 us，没有剔除长尾。

| 进程 | native p50 / p90 / p99 | PTO p50 / p90 / p99 | PTO−native p50 |
| --- | --- | --- | --- |
| run1 | 720.350 / 727.890 / 731.348 | 733.650 / 744.566 / 760.227 | +13.300 |
| run2 | 723.070 / 729.980 / 734.649 | 738.720 / 749.582 / 767.009 | +15.650 |
| run3 | 729.220 / 736.810 / 742.008 | 732.560 / 744.154 / 752.986 | +3.340 |

原始数据：`results/20260908_apple_to_apple/run{1,2,3}/result.json`，
各目录 `run.log` 为完整执行日志。三轮退出码均 0，采样前后输出和六类 cache
均通过现有校验；输出 max abs error 均 `0.000244140625`。
run1 PTO 第 71 个正式样本为 `1244.020 us`，原样保留。
run1 退出后两设备 AICore 均 0%，未发现退出后持续占用。

需要保留的两个口径限制：

- 此 fixture 使用真实生产实现和张量布局，但权重为合成随机值，历史 KV/cache
  初始大部分为零；C8191 是位置/长度配置，不是从真实 checkpoint 跑完 8191 token。
  它可以复现这份微基准，不能直接代表真实请求、模型问答精度或 serving 吞吐。
- 原入口浮点默认容差是 `atol=0.1, rtol=0.1`，只报“通过”不充分。
  本次记录实际误差，并在独立 profiling 运行收紧到 `atol=0.001, rtol=0.01`。
  未通过放宽容差来获得上述结果。

架构检查：skill 的预检脚本固定 card0，本机只暴露 card7，脚本失败；随后使用
`npu-smi info -t board -i 7 -c 0` 和 CANN 对应配置核实 `Ascend910_9362` /
`Ascend910_93`，确认 A3/a2a3。环境检查只有旧 commit pin 不一致；当前三仓
版本已在上文固定，没有为让检查变绿而改写 pin。

### 09:51：独立 profiling 启动；L1 子任务泳道限制已定位到代码

`profile1` 使用同一配置、输入 seed、设备和两个 ACLGraph，单独运行 3 个
diagnostic ABBA round，不把带 profiler 的结果混入上述三轮正式延迟。

PTO 子任务泳道不是漏传一个参数：当前 `L1Config` 没有 DFX 配置，且 simpler
`src/common/platform/onboard/host/device_runner_base.cpp:296` 在 L1 初始化时
明确拒绝 `config.diagnostics_any()`。L2 使用的 collector 初始化/启动/导出
没有接入 L1 生命周期。先检查 CANN 实际能记录的 L1 device 时间线，不能把
旧的 L2 全零数据转贴成对应泳道。

### 09:52–09:55：CANN 导出失败定位

`profile1` 的 NPU 执行和采样前后收紧容差校验已完成，但在导出检查处退出 1：
`kernel_details.csv` 缺失，`trace_view.json` 只有 215124 字节且 JSON 未闭合。
`analyse.done` 存在并不能证明导出成功；没有把该文件当成有效 profile。

手动运行同一 `PROF_*` 的 `msprof --export=on --type=text`，实际错误为
`ImportError: libsqlite3.so: cannot open shared object file`。`ldd msprof_analysis.so`
也明确显示该依赖 `not found`。msprof 的子进程导出失败，外层却退出 0；
torch_npu 的 `CANNExportParser` 仅检查退出码并捕获了 stdout/stderr，因而先报
export 成功，随后 timeline/relation/kernel parser 连锁失败。

系统已有 `libsqlite3.so.0.8.6`，缺少该解析器需要的无版本别名。仅在
`/tmp/csa-profiler-libs.N1JP0P/` 创建 `libsqlite3.so` 链接，进程内追加
`LD_LIBRARY_PATH` 后 `ldd` 依赖解析正常。没有修改系统库、CANN 安装或 benchmark
源码。失败产物保留；`profile2` fresh process 重新采集同配置的 3 个 ABBA round。

### 09:56–10:03：device profile 导出成功与逐条核验

`profile2` 退出码 0，采样前、采样后、profiling 后均通过收紧容差校验。
输出 max abs error 都为 `0.000244140625`，mean abs error 为 `1.362406743e-5`。
原始 `trace_view.json` 有 2143 个事件，`kernel_details.csv` 有 240 行 kernel：

| 路径 | Device Model ID | replay 数 | 每 replay kernel | 总 kernel 行 |
| --- | --- | --- | --- | --- |
| native | 49 | 6 | 38 | 228 |
| PTO L1 | 48 | 6 | 2（AICPU + AICore 父 kernel） | 12 |

离线审计发现 CSV 的 `Model ID` 对自定义 kernel 为 `4294967295`：包括 native
的 6 条 Triton RMS 和 PTO 的全部 12 条 kernel。若直接按 CSV Model ID 过滤，
会漏 native 算子并丢掉 PTO。实际 device trace 保留了正确 Model ID；本次用
**精确 device 起始时间 + kernel 名称**逐条一对一关联，并校验 duration 相等。
CSV 原始列不改，追加 `Resolved Device Model ID`、`Replay`、`Replay Kernel` 和
`Relative Start(us)`。不存在用 host enqueue 时间窗猜测 device 归属。

每路 6 个 MODEL_EXECUTE / MODEL_WAIT_COMPLETE 窗口均配对，检查窗口不交叠、
所属事件恰好落入一个窗口、kernel 无重复或遗漏。两路拆分 trace 保留原始 ts、dur、
stream 和 device 控制事件；只导出所选 device model，不制造缺失的 host flow。

仍有 torch_npu 的 `acl to npu flow` / step-range 关联告警，不能宣称 host→device
关联或 step 汇总完整。它不影响已独立核验的上述 240 行 device kernel；本次交付
的细分依据是 device model 与真实起止，不是缺失的 framework 归因。

静态预览固定选各路第 3/6 个 replay（第二个 ABBA round 的第一次调用），不是
挑最快样本；两图时间轴均从本次 MODEL_EXECUTE 起点开始，使用相同 0–850 us
横轴，保留真实空隙与重叠。PNG 已实际打开检查。native 的 38 个算子有两个计算
stream，本 profile 的 interval overlap 为 118.40–122.22 us。PTO 父 kernel 的
AICPU/AICore 重叠不能解释为子任务并行度，暂不能据此归因内部调度损失。

### 10:05–10:07：独立 fixture 身份核验

新增 `audit_fixture_identity.py`，仅在原构造函数返回时和计时前计算逻辑字节
SHA256，不改变 benchmark、production op、capture 或 ABBA 回调。独立 fresh
process `identity` 退出 0：

- 两路 30 个权重/buffer 张量的 shape、dtype、stride 和逻辑字节哈希完全相同。
- 六类 cache + block tables + slot mapping + positions 等共 21 个张量完全相同。
- 六类可变 cache 地址互不别名；原 runner 还校验 input/output/cache 的八组地址独立。
- `[32,4096]` BF16 hidden input 共 131072 个非零元素，两路与 resident source
  逐字节相同，SHA256 为 `90a51da67e11954da5b58e62912361819215e2ccfaf1bea823347e5dd24936f9`。
- 采样前后收紧容差校验均通过。该诊断进程的计时不加入正式三轮统计。

### 10:10：离线统计与交付校验

`audit_profile_artifacts.py` 从原始 600 个正式 device 样本独立重算 p50/p90/p99，
与 runner 保存值逐项一致；warmup 不混入、无非正样本、无重复 sample key、
没有删除长尾。11 个 Host 测试通过（14 条既有 torch deprecation warning），
本次新增 Python 文件的 Ruff 检查通过。三仓生产源码未改，TRB host/AICPU/AICore
binary SHA256 与 9 月 5 日记录一致，已写入 software manifest。

ABBA 的顺序效应需单独保留：native 每轮 A1/A2 的 p50 分别为
`702.70/725.64`、`705.62/727.65`、`712.51/734.53 us`。除首轮/同步边界外，
A1 接在上轮 native 后，A2 接在 PTO 后。这里只证明顺序相关，尚未识别是 cache、
调度或其他机制；因此三轮总体 p50 的差值不能包装成固定的“慢 17 us”。

### 10:14：最终只读检查

离线审计本已通过 nbformat 结构校验，并由独立 kernel 从头到尾执行完全部 4 个
代码单元，无异常；不是未执行模板。notebook 依赖与 kernel 安装在
`/tmp/csa-notebook-tools.jrknMV/`，没有更新项目既有依赖或用户级 kernel 配置。
PyPTO 与 simpler 工作树均干净，vLLM-Ascend 本次只新增诊断脚本、测试、文档
和结果。两设备 Health OK、AICore 0%，npu-smi 无运行进程；未做 reset。

## 当前交付与未完成项

文件总目录：[`results/20260908_apple_to_apple/`](results/20260908_apple_to_apple/)。

- [native device trace](results/20260908_apple_to_apple/artifacts/native_device_trace.json)、
  [逐 kernel CSV](results/20260908_apple_to_apple/artifacts/native_kernel_details.csv)、
  [预览图](results/20260908_apple_to_apple/artifacts/native_device_preview.png)。
- [PTO L1 父 kernel device trace](results/20260908_apple_to_apple/artifacts/pypto_device_trace.json)、
  [CSV](results/20260908_apple_to_apple/artifacts/pypto_kernel_details.csv)、
  [预览图](results/20260908_apple_to_apple/artifacts/pypto_device_preview.png)。
- [正式延迟汇总](results/20260908_apple_to_apple/artifacts/formal_latency_summary.csv)、
  [全部正式样本](results/20260908_apple_to_apple/artifacts/formal_device_samples.csv)、
  [输入身份核验](results/20260908_apple_to_apple/identity/fixture_identity.json)、
  [数据与代码哈希清单](results/20260908_apple_to_apple/artifacts/audit_manifest.json)、
  [软件基线](results/20260908_apple_to_apple/artifacts/software_manifest.json)。
- [离线审计本](results/20260908_apple_to_apple/reproduction_audit.ipynb)：重算样本、
  验证源文件 SHA、输入身份、12 个 replay 覆盖与区间计算闭合。

评估：**性能复现和 native device profile 可带上述微基准限制分享；用户要求的
PTO L1 逐 core 子任务泳道尚未完成，不能把整个交付标为完成。**
已询问是否允许在独立诊断分支接入 L1 collector 生命周期；在获得方向前不改变
simpler/PyPTO runtime，不使用 L2 全零泳道代替，不合并旧结果。

## 复现命令

### 10:35–10:43：获准补齐 L1 子任务采集

独立 worktree：`/mnt/workspace/inductor/simpler-l1-swimlane`，分支
`diagnostics/l1-csa-swimlane-20260908`。原 runtime 工作树、正式性能结果不变。

仅接入现有 level-2 collector：prepare 固定设备缓冲地址，外部同步后按单次 replay
清空/导出记录，close 先停止 collector 再释放内存。设备端重置每次调用的计数，
并复用没有任务的 core 留下的空缓冲，避免重复 replay 的计数混入和缓冲泄漏。
此处是诊断支持，尚在编译/测试，不能称已成功采图。

按用户最新要求，停止额外数据质量流程，不再生成或核对文件/张量摘要。
后续只记录必要改动、运行命令、算子结果和 profiling 文件。

### 10:47–10:50：小算子采集通过，转入 CSA

独立 runtime 编译通过，`test_l1_chip_worker.py` 8 项通过。
device0 上板前已显示 Warning；device1 为 OK，后续两路统一切到 device1。
`task-submit` 仍不存在，串行 unlocked 运行，不做设备 reset。
架构预检脚本仍因固定 card0 失败；实际 card7/chip1 为 Ascend910 / 9362（A3）。

`l1_child_swimlane --mode smoke --device 1` 完成 3 次 add ACLGraph replay，
输入逐次变化，三个输出正确，三个独立子任务文件均导出，退出后 device1 健康。
原始文件在 `results/20260908_apple_to_apple/l1_smoke1/replay{1,2,3}/`。

CSA 首次启动被原 benchmark 的 device0 硬编码在 Host 拦下，未执行 CSA；
诊断包装器临时覆盖该设备限制，两路均传入 device1，生产 benchmark 不改。
第二次启动目录为 `l1_csa2`。诊断采用单 replay 外部同步、单窗口导出，
这些带 DFX 的时间不作为正式延迟结果。

在 `vllm-ascend` 仓运行，先激活已有项目环境：

```bash
source .venv/bin/activate
source /mnt/workspace/inductor/pto/pypto/.claude/skills/testing/load-env.sh
unset PTOAS_ROOT
export PATH=/mnt/workspace/inductor/pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas:$PATH
export LD_LIBRARY_PATH=/mnt/workspace/inductor/toolchains/gcc15/lib:${LD_LIBRARY_PATH:-}
```

正式复跑（每次使用新的结果路径和独立进程；三次参数不变）：

```bash
TASK_QUEUE_ENABLE=1 python -m tests.pypto_dsv4_decode_csa.standalone_csa_perf_case \
  --device 0 --runtime tensormap_and_ringbuffer --mode aclgraph \
  --start-position 8191 --seed 20260902 --warmups 20 --samples 100 \
  --enqueue-batch-size 20 --master-port 29841 --result-json /tmp/csa-fresh-result.json
```

独立 profiling（仅该诊断进程增加 sqlite 解析库路径；本次实际使用下列临时目录）：

```bash
LD_LIBRARY_PATH=/tmp/csa-profiler-libs.N1JP0P:${LD_LIBRARY_PATH:-} TASK_QUEUE_ENABLE=1 \
  python -m tests.pypto_dsv4_decode_csa.a3_single_op_benchmark \
  --device 0 --runtime tensormap_and_ringbuffer --mode aclgraph \
  --start-position 8191 --seed 20260902 --warmups 20 --samples 100 \
  --enqueue-batch-size 20 --master-port 29852 --atol 0.001 --rtol 0.01 \
  --profile-directory /tmp/csa-fresh-profile --profile-rounds 3
```

若临时 sqlite 目录已清理，先以 `mktemp -d` 建新目录，在其中把
`libsqlite3.so` 链接到已确认存在的 `/lib/aarch64-linux-gnu/libsqlite3.so.0.8.6`，
再更新该进程的库路径；无需覆盖系统文件。

离线重建拆分 trace、CSV、预览和审计清单：

```bash
python -m tests.pypto_dsv4_decode_csa.audit_profile_artifacts \
  --root tests/pypto_dsv4_decode_csa/results/20260908_apple_to_apple \
  --profile tests/pypto_dsv4_decode_csa/results/20260908_apple_to_apple/profile2/raw
```
