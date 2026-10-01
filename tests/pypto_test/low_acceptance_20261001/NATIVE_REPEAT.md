# Native第二遍复测与六组对照

两档Native与native2的逐token、接受统计及逐轮事件均完全一致。

原五组结果复用同步加载修复后的完整矩阵；本轮仅新增128K/B24和8K/B40两个16卡Native任务。
使用原全部64条请求，不使用筛选后的easy_repeat B9。每条生成192 token，初始target/draft cache/state来自原固定bank。

| 档位 | Native平均接受/轮 | Native2平均接受/轮 | 不同token位置 | 统计不同副本请求 | 逐轮事件不同副本请求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B24 | 3.825182 | 3.825182 | 0/73,728 | 0/384 | 0/384 |
| 8K/B40 | 3.726161 | 3.726161 | 0/122,880 | 0/640 | 0/640 |

## 六组结果

下表以第一次Native为参考，按独立请求计数：任一DP副本不同即计入。16个DP副本不算作额外独立题目。

### 128K/B24

| 组合 | 平均接受/轮 | 输出不同请求 | 统计不同请求 | 逐轮事件不同请求 |
| --- | ---: | ---: | ---: | ---: |
| Native CSA + Native HCA | 3.825182 | 0/24 | 0/24 | 0/24 |
| Native2（全Native同配置第二遍） | 3.825182 | 0/24 | 0/24 | 0/24 |
| CSA精度版 + Native HCA | 3.884089 | 18/24 | 21/24 | 21/24 |
| CSA性能版 + Native HCA | 3.695960 | 15/24 | 20/24 | 21/24 |
| CSA性能版 + PTO HCA | 3.970053 | 19/24 | 20/24 | 21/24 |
| Native CSA + PTO HCA | 3.755235 | 18/24 | 21/24 | 21/24 |

[六组汇总](six_way_128k_b24.json)、[全部16rank两两差异](six_way_128k_b24.full.json.gz)。

### 8K/B40

| 组合 | 平均接受/轮 | 输出不同请求 | 统计不同请求 | 逐轮事件不同请求 |
| --- | ---: | ---: | ---: | ---: |
| Native CSA + Native HCA | 3.726161 | 0/40 | 0/40 | 0/40 |
| Native2（全Native同配置第二遍） | 3.726161 | 0/40 | 0/40 | 0/40 |
| CSA精度版 + Native HCA | 3.653221 | 33/40 | 31/40 | 32/40 |
| CSA性能版 + Native HCA | 3.665461 | 29/40 | 31/40 | 31/40 |
| CSA性能版 + PTO HCA | 3.648795 | 30/40 | 32/40 | 33/40 |
| Native CSA + PTO HCA | 3.702128 | 29/40 | 31/40 | 31/40 |

[六组汇总](six_way_8k_b40.json)、[全部16rank两两差异](six_way_8k_b40.full.json.gz)。

## 运行条件与检查

同一source_v9_sync_load冻结源码、vLLM提交752a3a504485790a2e8491cacbb35c137339ad34、CANN9.2.0-beta.2、正式0731-w8a8权重。
TP1/DP=EP16、seed=1024、temperature=0、DSpark出5、NZ2、确定性level1、HCCL确定性开启、EPLB关闭、watermark=0。
长档显存利用率0.97、图档位144；短档0.95、图档位6/240。npugraph_ex/static kernel和原生产编译入口保持不变。
Native两次event均为0；PTO原结果event为1，仍保留该已知集成路径差异。atomic0只控制PTO CSA，不表示全Native没有atomic。
两次Native的所有rank启动日志参数和实际worker配置均核对；仅忽略随机通信ID及观察结果目录。

| 档位 | 配置检查 | Native抢占 | Native2抢占 | Native2跨DP输出差异 | Native2跨DP事件差异 | 自动KV token容量范围（第一次→第二次） |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 128K/B24 | PASS | 0 | 0 | 0/360 | 0/360 | 2,726,503～2,727,767 → 2,726,601～2,727,865 |
| 8K/B40 | PASS | 0 | 0 | 0/600 | 0/600 | 627,238～627,569 → 627,262～627,592 |

自动KV容量按同一显存利用率策略生成，可能随启动可用内存轻微变化；不为追求容量数字相等而修改原参数。
完整逐rank容量记录见JSON。不声称所有执行调度轨迹、logits或浮点中间结果逐bit一致。

## 结论边界

本轮原完整低接受率用例的Native跨运行结果稳定，不能用已观测的Native自身波动解释Native/PTO的现有差距。
这不证明PTO必然存在功能bug，也不证明差异只是正常数值取舍；首次分歧前的同输入检查仍然需要。
Native2与各PTO组合的差异和第一次Native完全相同。两档六组主矩阵整体仍未通过精度验收。

这是一次新增重复运行，未测所有浮点中间结果，也不能证明任意未来运行都必然确定。
生产算子、采样与接受逻辑未修改，没有新增单卡/性能测试或重新采prefill。

## 复现

使用[原run_mixed.sh入口](run_mixed.sh)，configuration仍为native，输出放入新的native2目录；
两档均经统一队列分配16卡。原五组无需重新执行，完成后仅在CPU合并：

```bash
PYTHONPATH=tests/pypto_test python tests/pypto_test/low_acceptance_20261001/five_way.py \
  --root tests/pypto_test/results/low_acceptance_20261001/five_way_sync/128k \
  --bank tests/pypto_test/results/low_acceptance_20261001/mixed128k_fixed --batch 24 \
  --native-repeat tests/pypto_test/results/low_acceptance_20261001/native_repeat/128k/native2 \
  --compact-details --output /new/six_way_128k_b24.json
```

8K对应替换为8k、mixed8k_fixed、batch=40。完整六组因原PTO差异返回FAIL/exit1，
Native复现结论单独看native_repeat_status与native_repeat_conditions，不将整体FAIL误认为任务执行失败。

[任务和固定配置](native_repeat_tasks.json)、[原五组报告及容量诊断](RESULTS.md)。
