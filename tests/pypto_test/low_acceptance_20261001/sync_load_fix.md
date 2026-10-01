# 离线同步KV加载声明修复

## 问题与定位

128K/B24的“CSA性能版＋Native HCA”先后两次未完成：首次仅10个DP得到完整输出，
增加客户端保活后仍有DP 1/3/5/7/9/11/15未完成，300秒后退出。
这属于执行功能问题，不能当成浮点误差或用部分DP填补精度结果。

检查第二轮日志发现：这些DP均已加载全部24条请求的cache，却没有请求完成记录。
诊断任务`task_20261001_141029_197387125273`增加CPU侧调度进度记录后完整执行，
并记录到容量边界：偶数DP第3步可推进23条请求；奇数DP只推进1条，第4步即被抢占，
之后重新加载该请求。未改算子的一次跑通不能证明原问题已修复。
原始进度在`../results/low_acceptance_20261001/hang_diagnosis_v2/progress_rank*.jsonl`。

具体发现是离线connector接口声明与实现不符：

- `start_load_kv()`在当前forward前同步完成文件读取、H2D及`torch.npu.synchronize()`。
- `get_num_new_matched_tokens()`原来返回`(history, True)`，将它声明成跨调度步的异步加载。
- 异步入场先分配prefix，暂不分配lookahead。容量临界时多个prefix占满块池，
  接收完成的请求还需要decode/草稿空间才能进入RUNNING，而等待请求无法释放这些块。
- 客户端延长等待不能解除这种零进展状态。调度器的接收完成通知即便没有丢失也可复现。

vLLM `KVConnectorBase_V1.get_num_new_matched_tokens()`明确将第二个返回值定义为
“是否在scheduler steps之间异步加载”；其同步文件示例`ExampleConnector`返回False。
本次修复按现有实现使用同步声明，不修改生产CSA、HCA、vLLM调度器或cache布局。

## 修改

[connector.py](../offline_pd/connector.py)返回`(history, False)`，让调度器在入场时
一起考虑prefix、当前待计算token和speculative lookahead；删除不再适用的异步接收完成通知。
文件加载仍在forward之前同步完成，固定bank内容、权重和采样规则保持原口径。
[scheduler.py](scheduler.py)的逐请求完成记录增加`preemptions`，用于检查容量抢占。

## CPU回归

[test_offline_connector_cpu.py](../test_offline_connector_cpu.py)使用当前真实AsyncScheduler
和KV块管理器；只创建本地微型模型配置，不加载权重、NPU或PyPTO。
两个32-token请求分别恢复31-token前缀，采用DSpark的5个lookahead槽位，块池仅有两个可用块。

| 场景 | 行为 |
| --- | --- |
| 旧异步声明 | 两个prefix占满块池；给出全部接收完成通知后，连续三步仍为0 scheduled tokens |
| 修复后同步声明 | 第一条获得1-token首步及6-token后续步；第一条结束后第二条正常入场 |

两项测试通过。命令：

```bash
VLLM_PLUGINS='' PYTHONPATH=/path/to/pinned/vllm:tests/pypto_test \
  python -m pytest -q --noconftest tests/pypto_test/test_offline_connector_cpu.py
```

## 整模型复测

冻结源码为`../results/low_acceptance_20261001/source_v9_sync_load`，移除了临时逐步诊断钩子。
先复测原失败的128K/B24性能CSA＋Native HCA，再用相同修复刷新两档五组对照。
原始结果在`../results/low_acceptance_20261001/five_way_sync/`，任务ID见该目录的`tasks.json`。
整模型完成情况和最终精度差异以[RESULTS.md](RESULTS.md)的更新记录为准。

首个修复后任务`task_20261001_142251_220221019171`正常结束（exit=0，5分46秒）。
16个DP各完成24条不同请求，每条192个token，共384条、73,728个输出位置；
全部请求的抢占次数均为0，同次跨DP的token和接受事件差异均为0。
平均每轮接受3.695960个，实际接受0～5均覆盖，逐请求计数与原生统计一致，
21个CSA的PTO捕获与20个Native HCA实际选择核实通过。
这是功能恢复及覆盖证据，还不是与Native数值一致的结论；[恢复记录](sync_load_recovery_128k_b24.json)。

CPU用例证明了容量停滞机制；原两次失败没有逐步状态记录，不能声称已还原当时每个DP的全部状态。
修复后须以16个DP完整结束及逐请求记录为验收，不以一次无异常启动为通过。
