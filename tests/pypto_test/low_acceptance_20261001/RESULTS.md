# 七组CSA/HCA组合：完整混合接受精度与跨运行对照

2026-10-01维护决定：按用户要求[封存CSA精度版](../archive/csa_precision_20261001/README.md)，
性能版成为默认及唯一维护路径。下面七组数据、配置和未通过项保持原样；自身复现一致不等于Native/PTO精度已通过。

两档中，Native与Native2、PTO CSA+HCA与PTO2各自的输出token、接受统计及逐轮事件完全一致；Native与PTO之间的差异仍在。

已修复离线测试connector将同步KV加载误报为异步加载的问题，原五组采用相同修复后的冻结源码完整执行。
随后补充全Native第二遍（Native2），本轮再补充性能CSA＋PTO HCA第二遍（PTO2）；旧组结果直接复用。
两档七组均执行完整。跨运行复现与相对Native精度分开判断：各PTO组合相对Native仍未通过精度验收。
问题机制、CPU回归和原失败组合恢复证据见[修复记录](sync_load_fix.md)。

固定TP1、DP=EP16；128K/B24使用24条不同请求，8K/B40使用40条不同请求，每条生成192个token。
所有实现恢复同一份Native导出的target/draft cache/state，temperature=0，DSpark提出5个草稿。
本次重复使用原全部请求，不使用筛选后的easy_repeat B9。平均接受数不含目标模型追加token；
0～5为每轮实际接受个数。完整配置见[README](README.md)、[source.json](source.json)。

## Native与PTO的第二遍复现

| 档位 | 对比 | 首次平均接受/轮 | 第二遍平均接受/轮 | 不同token位置 | 统计不同副本请求 | 逐轮事件不同副本请求 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B24 | Native / Native2 | 3.825182 | 3.825182 | 0/73,728 | 0/384 | 0/384 |
| 128K/B24 | 性能CSA＋PTO HCA / PTO2 | 3.970053 | 3.970053 | 0/73,728 | 0/384 | 0/384 |
| 8K/B40 | Native / Native2 | 3.726161 | 3.726161 | 0/122,880 | 0/640 | 0/640 |
| 8K/B40 | 性能CSA＋PTO HCA / PTO2 | 3.648795 | 3.648795 | 0/122,880 | 0/640 | 0/640 |

两次同实现的全部16rank启动参数与worker配置均一致，包括event模式；仅排除每次生成的通信ID和结果目录。
自动KV容量仍按原显存利用率生成，实际容量变化记录于七组JSON的native_repeat_conditions/pto_repeat_conditions，
不因容量数值变化而修改原参数，也不声称实际调度轨迹完全相同。
[Native2任务](native_repeat_tasks.json)、[PTO2任务](pto_repeat_tasks.json)。

本轮没有观察到Native或PTO联合路径的跨运行token/接受事件波动，Native2与PTO2之间的差异也和两组首次对照相同。
这支持“同一实现能复现、实现之间仍有差异”，但不能仅据此判定PTO存在功能bug或把差异认定为可接受的数值取舍。
本轮未检查所有浮点中间结果；首分歧前的同输入定位仍需继续。

## 逐请求输出与DSpark差异

“输出不同请求”按独立题目计数：该题任意DP的192个输出token与对应Native不一致即计入。
“统计不同”比较草稿轮数、提出数、接受数与0～5直方图；“事件不同”还比较逐轮顺序。
16个DP是相同题目的副本，不能算成额外独立题目。

### 128K/B24

| 组合 | 平均接受个数/轮 | 输出不同请求 | 统计不同请求 | 事件不同请求 | 最早分歧token（从1计） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native CSA + Native HCA | 3.825182 | 0/24 | 0/24 | 0/24 | — |
| Native2（全Native同配置第二遍） | 3.825182 | 0/24 | 0/24 | 0/24 | — |
| CSA精度版 + Native HCA | 3.884089 | 18/24 | 21/24 | 21/24 | 3 |
| CSA性能版 + Native HCA | 3.695960 | 15/24 | 20/24 | 21/24 | 3 |
| CSA性能版 + PTO HCA | 3.970053 | 19/24 | 20/24 | 21/24 | 3 |
| PTO2（CSA性能版 + PTO HCA第二遍） | 3.970053 | 19/24 | 20/24 | 21/24 | 3 |
| Native CSA + PTO HCA | 3.755235 | 18/24 | 21/24 | 21/24 | 3 |

| 组合 | 不同token位置/全部位置（16卡） | 输出不同副本请求 | DSpark统计不同副本请求 |
| --- | ---: | ---: | ---: |
| Native2（全Native同配置第二遍） | 0/73,728 | 0/384 | 0/384 |
| CSA精度版 + Native HCA | 35,888/73,728 | 288/384 | 336/384 |
| CSA性能版 + Native HCA | 28,240/73,728 | 240/384 | 320/384 |
| CSA性能版 + PTO HCA | 31,952/73,728 | 304/384 | 320/384 |
| PTO2（CSA性能版 + PTO HCA第二遍） | 31,952/73,728 | 304/384 | 320/384 |
| Native CSA + PTO HCA | 35,778/73,728 | 252/384 | 330/384 |

[七组汇总](seven_way_128k_b24.json)、[全部16rank逐请求明细](seven_way_128k_b24.full.json.gz)。

### 8K/B40

| 组合 | 平均接受个数/轮 | 输出不同请求 | 统计不同请求 | 事件不同请求 | 最早分歧token（从1计） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native CSA + Native HCA | 3.726161 | 0/40 | 0/40 | 0/40 | — |
| Native2（全Native同配置第二遍） | 3.726161 | 0/40 | 0/40 | 0/40 | — |
| CSA精度版 + Native HCA | 3.653221 | 33/40 | 31/40 | 32/40 | 2 |
| CSA性能版 + Native HCA | 3.665461 | 29/40 | 31/40 | 31/40 | 2 |
| CSA性能版 + PTO HCA | 3.648795 | 30/40 | 32/40 | 33/40 | 2 |
| PTO2（CSA性能版 + PTO HCA第二遍） | 3.648795 | 30/40 | 32/40 | 33/40 | 2 |
| Native CSA + PTO HCA | 3.702128 | 29/40 | 31/40 | 31/40 | 2 |

| 组合 | 不同token位置/全部位置（16卡） | 输出不同副本请求 | DSpark统计不同副本请求 |
| --- | ---: | ---: | ---: |
| Native2（全Native同配置第二遍） | 0/122,880 | 0/640 | 0/640 |
| CSA精度版 + Native HCA | 59,328/122,880 | 528/640 | 496/640 |
| CSA性能版 + Native HCA | 59,712/122,880 | 464/640 | 496/640 |
| CSA性能版 + PTO HCA | 61,744/122,880 | 480/640 | 512/640 |
| PTO2（CSA性能版 + PTO HCA第二遍） | 61,744/122,880 | 480/640 | 512/640 |
| Native CSA + PTO HCA | 58,144/122,880 | 464/640 | 496/640 |

[七组汇总](seven_way_8k_b40.json)、[全部16rank逐请求明细](seven_way_8k_b40.full.json.gz)。

首个输出分歧之后，各实现的后续上下文已经不同；不能把后续token位置差异当成同输入逐元素浮点误差。

## 执行完整性、抢占与跨DP一致性

各组均要求16个DP各自完整输出全部请求，不能用部分rank补齐整模型结论。跨DP表比较rank1～15与rank0。

| 档位 | 组合 | 完成副本请求 | 抢占次数 | 跨DP输出不同 | 跨DP接受事件不同 |
| --- | --- | ---: | ---: | ---: | ---: |
| 128K/B24 | Native CSA + Native HCA | 384/384 | 0 | 0/360 | 0/360 |
| 128K/B24 | Native2（全Native同配置第二遍） | 384/384 | 0 | 0/360 | 0/360 |
| 128K/B24 | CSA精度版 + Native HCA | 384/384 | 0 | 0/360 | 0/360 |
| 128K/B24 | CSA性能版 + Native HCA | 384/384 | 0 | 0/360 | 0/360 |
| 128K/B24 | CSA性能版 + PTO HCA | 384/384 | 0 | 0/360 | 0/360 |
| 128K/B24 | PTO2（CSA性能版 + PTO HCA第二遍） | 384/384 | 0 | 0/360 | 0/360 |
| 128K/B24 | Native CSA + PTO HCA | 384/384 | 431 | 137/360 | 163/360 |
| 8K/B40 | Native CSA + Native HCA | 640/640 | 0 | 0/600 | 0/600 |
| 8K/B40 | Native2（全Native同配置第二遍） | 640/640 | 0 | 0/600 | 0/600 |
| 8K/B40 | CSA精度版 + Native HCA | 640/640 | 0 | 0/600 | 0/600 |
| 8K/B40 | CSA性能版 + Native HCA | 640/640 | 0 | 0/600 | 0/600 |
| 8K/B40 | CSA性能版 + PTO HCA | 640/640 | 0 | 0/600 | 0/600 |
| 8K/B40 | PTO2（CSA性能版 + PTO HCA第二遍） | 640/640 | 0 | 0/600 | 0/600 |
| 8K/B40 | Native CSA + PTO HCA | 640/640 | 0 | 0/600 | 0/600 |

上表只比较同次执行的跨DP副本；跨运行Native2/PTO2结果已在报告开头独立列出。

## HCA长档容量诊断（与主矩阵分开）

主矩阵128K/B24的Native CSA＋PTO HCA发生431次抢占；同次跨DP输出不同137/360、
接受事件不同163/360。它已完整结束，但不能隐去重算条件后把这些差异全归因于浮点算术。

额外任务只在冻结测试入口将vLLM已有的`watermark`从0改为0.01，入场时保留1%的KV块余量。
算子、bank与采样不改；该诊断不替换主矩阵中的某一行，也不修改生产默认值。

| 条件 | 抢占次数 | 跨DP输出不同 | 跨DP事件不同 | 对Native输出不同题目 | 对Native统计不同题目 | 平均接受/轮 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 主矩阵watermark=0 | 431 | 137/360 | 163/360 | 18/24 | 21/24 | 3.755235 |
| 额外诊断watermark=0.01 | 0 | 0/360 | 0/360 | 16/24 | 19/24 | 3.735444 |

预留余量后，抢占和同次跨DP差异同时消失，说明容量/入场条件是需要控制的因素。
与Native的token及接受统计仍不一致，不能把消除抢占当作精度修复。
这项对照改变了请求入场、抢占及重算条件。结果用于检查容量因素，不能等同于保持相同调度的算术A/B，
也不以同次跨DP一致冒充跨运行确定性。后续同输入定位应先固定这些执行条件。
[完整诊断](hca_watermark_001_128k.json)、[仅测试入口的差异](watermark_diagnostic.patch)。

## 接受边界覆盖

| 档位 | 组合 | 接受0 | 1 | 2 | 3 | 4 | 5 | 均值3～4且六种边界均覆盖 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B24 | Native CSA + Native HCA | 928 | 1328 | 1344 | 1360 | 1360 | 9056 | 通过 |
| 128K/B24 | Native2（全Native同配置第二遍） | 928 | 1328 | 1344 | 1360 | 1360 | 9056 | 通过 |
| 128K/B24 | CSA精度版 + Native HCA | 976 | 1120 | 1360 | 1152 | 1200 | 9376 | 通过 |
| 128K/B24 | CSA性能版 + Native HCA | 1232 | 1520 | 1456 | 1280 | 1488 | 8864 | 通过 |
| 128K/B24 | CSA性能版 + PTO HCA | 704 | 1056 | 1392 | 1200 | 1088 | 9520 | 通过 |
| 128K/B24 | PTO2（CSA性能版 + PTO HCA第二遍） | 704 | 1056 | 1392 | 1200 | 1088 | 9520 | 通过 |
| 128K/B24 | Native CSA + PTO HCA | 1038 | 1410 | 1376 | 1402 | 1558 | 8737 | 通过 |
| 8K/B40 | Native CSA + Native HCA | 1632 | 2560 | 2832 | 2192 | 2064 | 14896 | 通过 |
| 8K/B40 | Native2（全Native同配置第二遍） | 1632 | 2560 | 2832 | 2192 | 2064 | 14896 | 通过 |
| 8K/B40 | CSA精度版 + Native HCA | 2016 | 2448 | 3216 | 2032 | 2208 | 14656 | 通过 |
| 8K/B40 | CSA性能版 + Native HCA | 1888 | 2368 | 3280 | 2288 | 2096 | 14624 | 通过 |
| 8K/B40 | CSA性能版 + PTO HCA | 1808 | 2384 | 3520 | 2320 | 2112 | 14416 | 通过 |
| 8K/B40 | PTO2（CSA性能版 + PTO HCA第二遍） | 1808 | 2384 | 3520 | 2320 | 2112 | 14416 | 通过 |
| 8K/B40 | Native CSA + PTO HCA | 1616 | 2384 | 3392 | 2272 | 1824 | 14832 | 通过 |

边界覆盖合格不代表实现精度通过；不为个别实现另换题目或强制改变接受结果。

## 只替换一个算子时的直接比较

| 档位 | 左侧 | 右侧 | 输出不同题目 | 接受事件不同题目 |
| --- | --- | --- | ---: | ---: |
| 128K/B24 | CSA精度版 + Native HCA | CSA性能版 + Native HCA | 16/24 | 20/24 |
| 128K/B24 | CSA性能版 + Native HCA | CSA性能版 + PTO HCA | 19/24 | 20/24 |
| 128K/B24 | CSA性能版 + PTO HCA | Native CSA + PTO HCA | 20/24 | 22/24 |
| 8K/B40 | CSA精度版 + Native HCA | CSA性能版 + Native HCA | 30/40 | 34/40 |
| 8K/B40 | CSA性能版 + Native HCA | CSA性能版 + PTO HCA | 32/40 | 34/40 |
| 8K/B40 | CSA性能版 + PTO HCA | Native CSA + PTO HCA | 32/40 | 33/40 |

七组JSON保留全部21种两两比较，包括Native2和PTO2。不能仅按输出不同题目数给浮点算法精度排序。

## 对照条件和结论边界

- CANN9.2.0-beta.2、TMR、NZ2、atomic0、确定性level1、HCCL确定性；AIV保留，EPLB关闭。
- npugraph_ex和static kernel均有实际配置/安装证据。使用当前vLLM整模型编译入口，生产inplace_pass=False；不用于单算子SK性能结论。
- 21个CSA与20个HCA的runtime描述符、PTO捕获及实际图重放逐rank核实；每请求计数与原生Prometheus计数核对。
- Native进程event=0，PTO初始化event=1；其余受检查显式worker配置相同。尚未以相同event模式隔离因果。
- 自动KV容量没有强制相等。按启动日志记录的容量见[kv_capacity_128k.json](kv_capacity_128k.json)；不同入场批次仍可能影响结果。
- B24/B40是提交的独立请求数；FULL144/240含padding，不能据此宣称全程B条请求同时活跃。
- 本轮只比较整模型token/接受事件，未采七组同输入逐层hidden/logits或cache浮点位差。不能把所有差异归因于CSA/HCA算术，也不能先接受为量化权衡。
- 修复前的两次未完成、跨DP差异和旧比较保留在Git及验证日志§511/§512，不能继续当成当前修复版本结论。

## 可复用的首分歧样例

128K/B24中，两种CSA配Native HCA均无抢占、同次跨DP一致：

| 请求 | 精度CSA相对Native | 性能CSA相对Native |
| --- | --- | --- |
| h131072_q08 | 输出相同，接受事件不同 | 第3个输出token分歧 |
| h131072_q18 | 第3个输出token分歧 | 输出相同，接受事件不同 |

这些样例可用于下一步固定前缀的定位，不用不同题目数给两版浮点精度排序。

## 后续定位

从首个输出分歧前的相同token前缀出发，固定同一步输入与调度，先查整数metadata、保护区和cache/state，
再比较hidden/logits并隔离规约、量化与Top-K差异。在逐token和DSpark未对齐前，不标为精度验收通过。

## 第二遍复现命令

两次新增PTO任务仍使用原run_mixed.sh，configuration=csa_performance_pto_hca，输出分别写入pto_repeat/{128k,8k}/pto2；原始结果不覆盖。
七组CPU汇总沿用five_way.py，增加两个可选结果目录：

```bash
PYTHONPATH=tests/pypto_test python tests/pypto_test/low_acceptance_20261001/five_way.py \
  --root tests/pypto_test/results/low_acceptance_20261001/five_way_sync/128k \
  --bank tests/pypto_test/results/low_acceptance_20261001/mixed128k_fixed --batch 24 \
  --native-repeat tests/pypto_test/results/low_acceptance_20261001/native_repeat/128k/native2 \
  --pto-repeat tests/pypto_test/results/low_acceptance_20261001/pto_repeat/128k/pto2 \
  --compact-details --output /new/seven_way_128k_b24.json
```

8K替换为8k、mixed8k_fixed和batch=40。整体精度FAIL会返回exit1；重复运行分别看native_repeat_status/pto_repeat_status以及对应conditions。
