# Native / PTO ACLNN：保留静态编译的 SK 开关对照

2026-10-31 汇总已有四组真实 16-worker 结果，没有新增 NPU 测试。
PTO 开 SK 的 target 最慢 rank mean 为 **52.538645 ms**，比同配置 Native
低 **11.93%（7.114669 ms）**，距离 48 ms 目标仍差 **4.538645 ms**。

## 固定口径

- CANN 9.2 beta2，DP=EP=16、TP=1、每卡 B16；同一冻结源码
  `results/model_decode_source_sk_ab_v1`、权重和固定请求。
- 128K 公共前缀加问题后缀，实际历史长度 131104～131346；四组各 rank 的请求键、
  长度逐项一致，未修改业务输入追逐性能目标。
- 四组都开启 static kernel、npugraph_ex、inplace_pass；dynamic=False、NZ2、
  event mode=0、TASK_QUEUE_ENABLE=1、PER_STREAM_QUEUE=0；保留共享专家多流。
- 预热后每 rank 取 10 个满批 B16/T96 target `_model_forward` 设备事件样本。
  不计初始化、draft、接受拒绝；不剔除长尾。
- **mean 是最慢 rank 的 10 步均值；min/P95/max 取全部 160 个样本。**
  不将不同 rank 的第 N 个样本假定为同一个全局 step。
- 三步 profiling 独立于稳态事件计时，采集耗时不替换下表数据。

## Target 网络耗时

单位：ms。

| 后端 | SK | min | 最慢 rank mean | P95 | max | 最慢 rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Native | 关 | 63.689419 | 65.862919 | 66.274803 | 66.458702 | 4 |
| Native | 开 | 58.765598 | 59.653314 | 59.952301 | 60.118221 | 1 |
| PTO ACLNN | 关 | 55.432720 | 56.186922 | 56.702862 | 56.836342 | 5 |
| PTO ACLNN | 开 | 51.849541 | 52.538645 | 53.123741 | 53.440102 | 13 |

| 对照 | 最慢 rank mean 减少 | 耗时降幅 |
| --- | ---: | ---: |
| Native：SK 关 → 开 | 6.209605 ms | 9.43% |
| PTO：SK 关 → 开 | 3.648278 ms | 6.49% |
| SK 关：Native → PTO | 9.675996 ms | 14.69% |
| SK 开：Native → PTO | 7.114669 ms | 11.93% |

两侧开 SK 都受益；本轮 Native 的 SK 收益更大，因此开 SK 后 PTO 相对 Native
的绝对优势从 9.676 ms 缩小到 7.115 ms。该差值是整网实测结果，不能直接归因于
某个 CSA 核或某段 Host 开销。四组是本机同配置对照，不与客户约 50 ms 的记录
混算收益，也不据此宣称达到 48 ms。

## 执行和输出核对

四组队列任务均 exit=0。所有 worker 的记录均确认 static=True、inplace=True 和
预期的 SK 开关；rank0 日志有真实静态编译过程及
`OFFLINE_STATIC_KERNEL ... module_loaded=True installed_packages=4`。
PTO 两组的禁止 Python PTO/Simpler 执行检查通过。

离线解析四组已有 rank0 三步 profile：SK 关时 Native/PTO 每步 SK 任务数都是 0；
SK 开时分别为 582/332。PTO SK 关的每步能直接看到 21 个 CSA ACLNN 核；
SK 开后名称只显示融合组首尾，不能把一个带 CSA 名称的 SK 解释成只融合了一层。
这里确认开关实际生效，不以任务数量证明全部算子的融合范围。

Native 的 SK0/SK1 之间、PTO 的 SK0/SK1 之间，各自的 256 个请求全部输出 token、
稳态输出和 16 个 rank 的 DSpark 统计均一致。此结论只针对同后端切换 SK，
不表示 Native 与 PTO 的复杂问题输出已经一致。

## 证据

任务 ID、固定协议见 [sk_ab_validation.json](sk_ab_validation.json)。
完整逐 rank 样本和比较结果见
[target_timing.json](results/sk_ab_summary/target_timing.json)，
三步实际执行核对见 [graph_coverage.json](results/sk_ab_summary/graph_coverage.json)。
精简、可提交的结果见 [sk_ab_results.json](sk_ab_results.json)。

原始目录为 `results/native_sk0_ab_v1`、`results/native_sk1_ab_v1`、
`results/pto_sk0_ab_v1`、`results/pto_sk1_ab_v1`；每个目录包含 16 份
`rankN.performance.json`、worker 日志及独立三步原始 profile。
本轮只导出了各组 rank0 的 `trace/rank0/*/ASCEND_PROFILER_OUTPUT/trace_view.json`。
原始结果目录是本地产物，不随 Git 提交。
