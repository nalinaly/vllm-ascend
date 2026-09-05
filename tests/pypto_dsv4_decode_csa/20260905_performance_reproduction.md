# vLLM-Ascend CSA 性能复现与设备残留状态排查

更新时间：2026-09-05 14:09:20

## 1. 范围与当前状态

本阶段先复现已有 native production / PyPTO CSA 单算子性能比较，暂不修改
CSA 优化代码。沿用前期记录，所有新增时间不带时区。用户另要求排查并在可能时
修复历史异常退出后 AICore 利用率长期 100% 的问题。

上一阶段 A2/A3 runtime post-close worker-retirement 修复及中文版已提交
[simpler PR #2133](https://github.com/hw-native-sys/simpler/pull/2133)。已提交不等于
已合入主仓，也不等于所有环境的二进制均已更新。本阶段必须独立核对实际加载路径。

三轮复测完成，采样前后精度全部通过，但未复现此前“PyPTO 小幅超过 native”的结论：
本次 PyPTO p50/p90/p99 的配对差中位数分别慢 11.130 / 15.004 / 18.949 us。
device0、device1 均在修复版 runtime 的正常完整执行/退出后恢复 0% 利用率，未执行 reset。
第 2 节历史数字与第 6 节本次复测分开记录。

## 2. 固定比较口径与历史结果

- 入口：`a3_single_op_benchmark.py`。
- A3、TP1、ratio-4、B4/S8、start-position 8191，完整 index score 路径。
- `tensormap_and_ringbuffer`、ACLGraph、固定 captured binding；
  `TASK_QUEUE_ENABLE=1`；native multistream overlap 保持开启。
- 3 个独立 Python 进程；每进程 20 次 warmup、100 次正式 sample，
  同卡、同 caller stream ABBA，每 20 次 enqueue 批量同步。
- 采样前后都检查非零 output、六类 mutable state 与 indexer 联合量化契约。
- 主指标为稳态 `device_span`，编译、prepare、capture、首次 replay、warmup
  和 validation-only copy 不计入。Host enqueue 单列，不与重叠的 device span 相加。
- 不把本项结果外推为整网吞吐、HBG 性能或 B8/B12/B16 性能。

历史证据：`results/20260903_device0_trb_steady_state_performance.json`。

| 进程 | native p50/p90/p99（us） | PyPTO p50/p90/p99（us） |
| --- | --- | --- |
| 1 | 727.460 / 734.704 / 737.949 | 705.870 / 718.912 / 730.880 |
| 2 | 724.700 / 732.510 / 736.001 | 711.400 / 724.110 / 732.607 |
| 3 | 712.550 / 718.186 / 724.581 | 700.280 / 709.086 / 717.236 |

每进程先算 `Dq = native_q - PyPTO_q`，再取三个进程 Dq 的中位数：
p50/p90/p99 分别为 +13.300 / +9.100 / +7.068 us。正值表示 PyPTO 更快。
不混合三个进程的原始样本，也不以两个跨进程中位数相减代替配对差。

## 3. 设备只读检查

### 2026-09-05 13:50—13:54

`npu-smi info -m` 确认映射如下；复位命令中的 NPU/card ID 不能误当逻辑 device ID。

| 逻辑 device | NPU/card | chip | 物理设备号 | 当前状态 |
| --- | --- | --- | --- | --- |
| 0 | 7 | 0 | 14 | AICore 100%，HBM 3462 MiB，无列出的设备进程 |
| 1 | 7 | 1 | 15 | AICore 100%，HBM 2874 MiB，无列出的设备进程 |

`npu-smi info -t proc-mem -i 7 -c 0/1` 均返回 `No process in device`。
容器进程列表没有运行中的模型/算子测试 Python 进程。以上不保证能够看到所有
宿主机/其他容器的进程，不能据此擅自复位整个物理板卡。

细分 `usages` 查询：device0 AIV 80%、AICube 100%，device1 AIV 0%、AICube 94%；
两者 AICPU、HBM bandwidth、NPU Utilization 均为 0%。这些计数与正常持续计算不一致，
但单凭这些现象不能区分 runtime 残留任务、驱动状态未回收或利用率统计残留。

单独 `health` 查询还发现 device0 的 `81078603`，描述为 port link up-to-down；
device1 为 OK。暂未建立该端口告警与本地计算阻塞的因果关系，不将它当作 CSA 根因。

### 设备级恢复权限/接口边界

只调用帮助，没有执行 reset：

1. 普通用户 `npu-smi set -h`：要求 root。
2. `sudo -n npu-smi set -h`：sudo 环境缺少驱动动态库。
3. 仅为此帮助命令显式提供 root 所有的 driver/common、driver/driver 库路径后：
   `This command cannot be executed on a VM or container.`

因此当前容器不能通过官方 npu-smi 管理接口复位设备。不修改工具、不绕过容器限制、
不重启宿主机、不复位另一块设备。下一步先用有时限的独立最小运算进程确认计算进展，
需要设备级恢复时交由宿主机执行，并先确认本型号的复位影响范围。

## 4. 软件/源码基线核对

2026-09-05 13:52，按 README 使用 vLLM-Ascend `.venv`、固定 PTOAS 和 GCC15
runtime 路径运行只读 `check_environment --device 0`。

- Python 3.11；torch 2.12.0+cpu；torch-npu 2.12.0+git5462a1b。
- PyPTO 0.2.1、simpler 0.1.0；PyPTO/simpler 实际 Python 源路径仍指向
  `/mnt/workspace/inductor/pto/pypto` 及其 `runtime`。
- PyPTO HEAD `9cece0b730a96fe1a52c2637537132f524ffe1ea`，
  simpler HEAD `b6f905f63277597bd2547d672fd9d57b6013fca9`；后者有上阶段 runtime 修复的
  工作树改动，不能仅用 HEAD 声称与旧性能测量时完全相同。
- 唯一失败项是检查器固定的旧 vLLM-Ascend commit。当前 HEAD 为
  `01861ad0baced3dfc829e825f1ed16616706171a`，检查器仍要求 `e7cb166...`。
  审计 `e7cb166..HEAD` 后，提交差异属于测试、文档、profile 归档和 lint 配置，
  没有生产算法修改；检查器仍原样保留 `ok=false`，不偷偷更新历史 pin。
- benchmark entry、`benchmark.py`、`native_fixture.py` 的 SHA256 与历史 JSON
  的 current/支持文件指纹一致。生产实现指纹也已复核为历史 current 值
  `5d3726a7dd2c7c36b81720380a03e5be1053515ee0d18e2fdc3b513b4dcfb4b3`。
  正确计算方法是按文件路径排序后生成 `sha256sum` 清单，再对清单哈希；
  不能按每行开头的 hash 排序，否则只是生成另一种清单，并不证明源码变化。
- 用户已有 Qwen3、worker、DSA 接入以及未跟踪 CSA 文件全部保留；本阶段未修改算法实现。

### 2026-09-05 13:55—13:59：最小运算、Host 测试与实际 runtime

- device0 最小运算进程约 6 秒完成，退出码 0。FP16 16×16 ones 初始化、
  vector add、Cube matmul、各阶段 device synchronize 和数值断言全部通过。
  日志：`/tmp/csa_20260905_device0_progress_probe.log`。45 秒外部 timeout 未触发。
- 进程退出后 `usages` 仍为 AICore 100%、AIV 75%、AICube 95%，没有设备进程。
  因此“利用率 100%”不等于“不能执行新任务”。仍不能仅凭小矩阵成功，断言没有
  残留状态或它对大任务/性能毫无影响。
- CSA Host suite：531 passed，14 个 torch deprecation warning，70.86 秒。
  日志：`/tmp/csa_20260905_host_tests.log`。
- `onboard-arch-precheck` 原脚本硬编码 card0，而容器只暴露 card7，因此原命令未能
  检测芯片。按该 skill 的相同检测方法查询 card7/chip0，得到 `Ascend910_9362`，
  CANN 对应 ini 的 `Short_SoC_version=Ascend910_93`，确认是 A3 / a2a3，非 A5。
- `task-submit` 不在 PATH。本次 NPU 调用顺序执行，记录为 unlocked；不能据此
  声称获得与 CI 设备锁等价的隔离保障。
- 实际 `_task_interface` 来自 vLLM-Ascend 本地 venv，build commit 为 `b6f905f...`；
  `simpler_setup.environment.PROJECT_ROOT` 指向当前 PyPTO/runtime 源树，非 PR 独立 worktree。
- 该树 TRB `libaicpu_kernel.so` 含 `platform_finish_aicore_exit` 和
  `platform_publish_aicore_post_close_release` 符号，AICore 源码也有 `ld_dev` 等待及 DSB。
  本次运行使用上阶段本地修复产物，不是重新运行旧退出协议。
- TRB 三份预构建产物 SHA256：
    - AICPU：`02bb8fc062b9ab3a959873588c79e5010e135131eeee35aeb533a51a397c65dc`
    - Host：`1bb66068ac4093a618c35ecad23c57af1e1436bd9263dbe4c7566698e0f27d75`
    - AICore：`e285ae31cf3790272a49830b92dc83231506817b13c642f9f330a4955f259067`

### 2026-09-05 13:59:18：启动第一个 fresh-process 正式口径运行

命令使用第 2 节全部参数，另设 240 秒进程时限（仅用于异常保护，不进入样本）。
设备日志通过 `ASCEND_PROCESS_LOG_PATH` 定向至本轮目录，不去其他任务的共享日志中查找。

```bash
TASK_QUEUE_ENABLE=1 python -u -m tests.pypto_dsv4_decode_csa.a3_single_op_benchmark \
  --runtime tensormap_and_ringbuffer --mode aclgraph --device 0 --batch 4 \
  --start-position 8191 --warmups 20 --samples 100 --enqueue-batch-size 20
```

原始输出：`results/20260905_device0_reproduction/run1.log`。

## 5. 执行完成情况与后续分析边界

1. 已完成 Host 单元测试、源码差异审计及实际 runtime 构建产物指纹记录。
2. 已完成 device0 最小运算和完整运行，以及 device1 独立恢复/功能验证；不需要宿主机 reset。
3. 已完成三个 fresh process 的正式 ABBA 口径比较，采样前后正确性均通过。
4. 已保存原始日志、完整原始样本 JSON、逐进程统计和配对差，见第 6 节。

后续优化需要先解释为什么算法源码未变，但 PyPTO 稳态时延相较旧记录上升。
本次实际 runtime 含上阶段退出握手修复，与历史测量环境不同，因此优先分析 runtime
调度/关闭尾部的开销是合理方向。但本轮没有重跑旧退出协议，也没有做只改变退出协议的
受控 A/B；不能把全部性能差异直接定性为该修复导致，更不能通过删除正确性所需的等待
来“恢复性能”。旧的 100% 读数也不是全部差异的解释，恢复后的 run2/run3 依然较慢。

## 6. 本次实际结果

单位均为 us；下表是本次新运行，非复制历史结果。

| 进程 | native p50/p90/p99 | PyPTO p50/p90/p99 | D（native - PyPTO） | 采样前后正确性 |
| --- | --- | --- | --- | --- |
| run1 | 723.960 / 731.816 / 738.329 | 735.090 / 745.074 / 757.279 | -11.130 / -13.258 / -18.949 | 均通过 |
| run2 | 727.010 / 733.676 / 744.606 | 732.940 / 748.680 / 758.835 | -5.930 / -15.004 / -14.229 | 均通过 |
| run3 | 719.790 / 726.390 / 729.015 | 737.600 / 752.128 / 759.265 | -17.810 / -25.738 / -30.250 | 均通过 |

三轮 output 最大绝对误差均为 0.000244140625，均值误差约 1.3624e-5，
与历史证据一致；六类 state 及 indexer 联合量化检查通过。

| 指标 | D 的三轮中位数（us） | 每轮 PyPTO 相对 native 变慢比例的中位数 |
| --- | --- | --- |
| p50 | -11.130 | 1.5374% |
| p90 | -15.004 | 2.0450% |
| p99 | -18.949 | 2.5665% |

本次三轮、三个分位的 D 均为负，未达到既定的 `D50 > 0, D90 >= 0, D99 >= 0` 性能门槛。
这是“正确性通过、领先目标未复现”，不是执行失败或精度失败。
双方 Host enqueue p50 约 18—19 us，不能把与 device 重叠的 Host 时间相加解释总延迟。

产物位于 `results/20260905_device0_reproduction/`：

- `run1.log` / `run2.log` / `run3.log`：原始进程输出，退出码均为 0。
- `run1.json` / `run2.json` / `run3.json`：从原始输出提取的完整 benchmark JSON，
  包含每后端每进程 100 个正式样本、20 个 warmup 样本，以及完整精度门禁。
- `summary.json`：协议、源码/二进制指纹、逐进程量化结果、配对差和边界说明。
- 已独立按原始 sample 数组重新计算线性插值 p50/p90/p99，确认与 runner 的统计字段一致；
  没有把 warmup 或跨进程样本混入计算。

### device0 利用率恢复的实测顺序

1. 运行前连续查询为 AICore 100%，无设备进程。
2. native 最小运算成功后，仍显示 AICore 100%。
3. 第一轮完整 native/PyPTO CSA eager warmup、capture、replay、teardown 成功后，
   `Aicore/Aivector/Aicube Usage Rate` 全部变为 0%，`proc-mem` 仍为无进程。
4. 第二轮完整运行成功后再次查询，三个利用率仍为 0%。

没有执行设备 reset、写驱动寄存器、修改/重装驱动，也没有新增 runtime 源码改动。
恢复发生于包含修复版 runtime 的完整运行/正常退出之后，这是时间顺序证据；
尚未对硬件/驱动内部状态做观测，不能进一步声称精确到哪一个寄存器或计数器的恢复机制
已经得到证明。run2 在恢复后依然稍慢，说明不能把本轮性能差异全部归咎于初始 100% 读数。

### 2026-09-05 14:05:56：device1 也已恢复

device1 测试前仍为 AICore 100%、AICube 94%、无设备进程。按同一 SoC 方法核对架构后，
单独运行已有 `subprocess_runner --runtime tensormap_and_ringbuffer --device 1 --batch 4`。
未与 device0 正式性能采样并发，也未修改 runner 的设备限制。

- eager 成功，output 是原预分配张量。
- 4 次不同输入值的 ACLGraph replay 成功，最大误差 0。
- main/inner 状态确实写入，四类 page-padding canary 无破坏。
- 正常 `graph.reset()` 和 `pypto.l1.shutdown(device=1)` 后进程退出码 0。
- 随后 AICore/AIV/AICube 三个利用率均变为 0%，无设备进程。

这是 real-shape **zero-weight 功能/恢复探针**，不是 device1 非零精度或性能验收。
记录位于 `results/20260905_device1_recovery/smoke.log` 与 `smoke_result.json`。
两张卡的现有异常读数均已消除；本阶段未新增任何“强制清零”或驱动修改代码。
