# 混合请求的DSpark接受边界基准

此前128K/B16两版49,152个输出token相同，但DSpark接受率100%，只能作为冒烟证据。
本轮改用同批不同请求，覆盖拒绝、部分接受、全部接受以及请求间不同的回退位置。
生产CSA/HCA算子不改，算子版本为`84dc9a3f`，测试起点为`7e57a5b4`。
当前入口包含`976ebd77`的同步KV加载声明修复；旧异步声明版本的结果不能混入当前矩阵。

## 最新验收口径

- 用例池至少40条不同请求，填满128K/B24、8K/B40；不能用两个问题重复复制成整批。
- 每轮提出5个草稿，记录实际接受长度0/1/2/3/4/5的事件直方图；主批须覆盖全部六种长度。
- 主基准整批按草稿轮次加权，平均接受数目标约3.8，允许3～4。**不是每条请求都限制在3～4**。
- 其他batch_size从同一固定池取不同请求，平均接受数允许1～5，不强行重配到3～4；边界覆盖逐档报告。
- 对每条请求保存逐轮提出数、接受数、输出token；逐请求统计合计必须与vLLM原有Prometheus计数一致。
- 逐token、逐请求DSpark统计、实际PTO图覆盖均须验收；100%接受率不能替代这套基准。
  图覆盖计数是补位后的B×6档位，不将它冒充同一步实际活跃B条请求或满并发性能验收。

“平均接受3.8个”不含目标模型追加的1个token，不等于3.8%的接受率。主要报告
`accepted / draft_rounds`；另报`accepted / proposed`。如果某轮实际提出不足5个，后者不能直接拿前者除以5。
这里覆盖的是每轮接受长度边界，不声称存在“整个请求每轮都恰好接受0个”的自然输入。

## 用例与固定状态

[prepare_mixed.py](prepare_mixed.py)产生40条不同问题，初始配比为16个Python轨迹、12个状态机、
6个账本计算、6个简单重复任务；参数不同，按固定顺序交错，不修改草稿、采样或接受规则。
题目用于检测两版的一致性，本轮不验收模型的算术解题能力。

分别复用已审计的8192/131072-token前缀，然后追加正式DeepSeek V4 chat编码的问题。
所以开始生成的上下文为8K/128K加数十至数百个问题token，不冒充精确8192/131072长度。
[8K题目记录](mixed8k_cases.json)、[128K题目记录](mixed128k_cases.json)包含完整题目和实际后缀长度。

固定bank由Native TP1/DP=EP16真实prefill后缀、保存全部target与draft cache/state。
性能版和精度版恢复同一份bank，只重算最后一个prompt token。保存前缀时沿用既有逻辑屏蔽未来行，
导出后只做一次边界/结构/覆盖检查；不增加hash和全量位差扫描。

文件加载在当前forward之前同步完成，connector向调度器声明同步加载，
入场时一并考虑prefix、当前计算token和DSpark lookahead空间。
旧版误报异步加载可在容量临界点占满prefix块而无法继续decode；详见[修复与CPU复现](sync_load_fix.md)。
逐请求完成记录含`preemptions`；汇总报告同时披露容量抢占，不能将其隐去后只谈浮点精度。

## 配置与执行

正式DeepSeek-V4-Flash-0731-w8a8，TP1/DP=EP16，每请求生成192个token，temperature=0。
当前正式矩阵包含以下五种组合，均用相同输入与固定cache/state：

1. Native CSA + Native HCA。
2. CSA精度版 + Native HCA。
3. CSA性能版 + Native HCA。
4. CSA性能版 + PTO HCA。
5. Native CSA + PTO HCA。

CANN9.2、TMR、NZ2、atomic0、确定性level1、
HCCL确定性，AIV保留，EPLB关闭。图模式为FULL_DECODE_ONLY，static kernel必须有实际安装证据。
B24长档显存利用率0.97、capture_sizes=[144]；B40短档0.95、capture_sizes=[6,240]。
五组主矩阵统一使用默认`watermark=0`；额外的HCA容量诊断使用`watermark=0.01`，单独记录，
不替换主矩阵中的某一组，也不据此改变生产配置。
使用当前vLLM整模型编译入口（生产配置`inplace_pass=False`），不是单CSA的SK1性能基线。
本轮只作精度验收，不给出性能结论。

[scheduler.py](scheduler.py)继承原生AsyncScheduler，仅观察原生统计回调；原始调度、采样和接受逻辑继续执行。
每个请求结束只写一次CPU JSONL，不增加NPU同步。
各客户端完成生成后先用文件标记等齐16个DP，再结束观察和关闭EngineCore；
等待期间EngineCore继续参与其他DP的EP通信，不在worker内调用阻塞屏障。
该处理覆盖低接受率导致的各DP结束轮数不同，避免先结束的进程破坏其他rank的收尾。
此观测用来诊断正确性，不用来宣称性能收益。
[mixed.py](mixed.py)提交不同请求并核对逐请求统计；[five_way.py](five_way.py)比较五组及全部两两组合。
逐层读取实际runtime选择，并核对捕获层数，防止配置标签与实际运行路径不一致。
Native默认event模式为0，PyPTO初始化切到1；此差异单独报告，其他worker配置须一致。
本轮验收实际集成路径，尚未通过相同event模式对照排除该差异的影响。

```bash
# 使用完整冻结源码和新结果目录；16卡任务经过统一队列。
# 五种configuration名称见five_way.py的CONFIGURATIONS。
/data/server-toolkits/pto-task/app/task-submit --device auto --device-num 16 --max-time 2400 \
  'bash /path/to/repo/tests/pypto_test/low_acceptance_20261001/run_mixed.sh /new/result/native /frozen/source /fixed/bank 40 native'
# 五组结束后CPU汇总，无需再次占卡：
PYTHONPATH=tests/pypto_test python tests/pypto_test/low_acceptance_20261001/five_way.py \
  --root /new/result --bank /fixed/bank --batch 40 --compact-details --output /new/result/comparison.json
```

修复前第4组曾复用既有结果，其余四组补跑两档；其中128K性能CSA+Native HCA两次未完成。
客户端保活没有解除停滞，不能将提前退出作为唯一根因。
当前已修复离线connector的同步加载声明，见[定位与回归](sync_load_fix.md)；
两档五组已用同一修复入口全部重新完成，未混用修复前后的结果。
两档分别提交24/40条不同请求，每卡使用相同题目顺序。全量输出为384/640条请求、
73,728/122,880个token；16个DP副本不是384/640种不同题目。
图档位包含padding，不能据此宣称全程有B条请求同时活跃。

## 结果与范围

两档五组执行全部完成，8个PTO组合/档位的精度比较未通过；正式结果见[RESULTS.md](RESULTS.md)。
可读JSON中的明细样例仅rank0；计数覆盖全部16rank，完整逐rank差异保存于对应`.full.json.gz`。原始证据位于
`../results/low_acceptance_20261001/five_way_sync/{128k,8k}/`。
固定输入和真实0～5接受边界已建立；验收失败不能改写为基准通过。
同次16个DP副本一致也不能替代同版本跨运行复现性测试。

此前两版CSA同时配PTO HCA的历史对照见验证日志§510；其“精度版+PTO HCA”不是本轮第2组。
旧[128K两版对照](comparison_128k_b24.json)、[8K两版对照](comparison_8k_b40.json)不属于当前同步加载矩阵。
位置差异是在各自生成轨迹上的比较：首个token分歧后输入上下文不同，
不能把后续大量位置差异等同于同输入浮点误差，更不能直接归因为功能错误或可接受量化误差。

早期筛选、单问题复制批和两类问题准备只作历史证据，不作为当前回归基准。

## 后续回归

[baseline.json](baseline.json)固定用例池、实际bank、批次顺序、主批/其他batch的接受范围。
五组回归使用`run_mixed.sh`与`five_way.py`，其他batch在CPU汇总时指定`--mean-range 1 5`。
[run_pair.sh](run_pair.sh)仅用于“两种CSA都配PTO HCA”的额外对照，其他batch可在四个必填参数后传`1 5`。
[reference.py](reference.py)将本轮逐请求输出及接受统计保存成压缩JSON；后续用`check`直接比较实际token和计数。
参考来自明确指定的实现，不能把它说成独立Native正确性证明。

```bash
python tests/pypto_test/low_acceptance_20261001/reference.py check \
  --root /new/native --reference /pinned/reference_8k_b40_native.json.gz --output /new/reference_check.json
```

固定bank需要重建时，先用`prepare_mixed.py`引用对应已审计的原始H8192/H131072 bank，
通过`--encoding`传入当前固定vLLM的`vllm/tokenizers/deepseek_v4_encoding.py`；再用
`fixed_bank.py prepare --source /suffix_bank --bank /new_fixed_bank`创建目标计划。
经16卡队列运行`extend_mixed.sh /new_output /frozen_source /suffix_bank /new_fixed_bank 24`
（8K用40），脚本会完成Native导出和一次结构审计。日常回归直接复用已有固定bank，不重复prefill。

同步加载矩阵的长档`h131072_q08`：精度CSA+Native HCA的输出与Native一致但接受事件不同，
性能CSA+Native HCA在第3个输出token分歧；`h131072_q18`则是精度CSA在第3个token分歧、
性能CSA输出与Native一致但接受事件不同。这两种CSA组合均无抢占、同次跨DP结果一致。
不能根据“精度版”名称或不同token总量直接给算法精度排序。
先核对同一步入参/状态及Native结果，再区分算术权衡与回退/状态功能错误；当前不先给原因定性。
