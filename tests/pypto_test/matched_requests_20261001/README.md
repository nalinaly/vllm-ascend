# 历史三方一致请求的独立复现

从同步加载修复后的完整五组结果中，筛选全Native、精度CSA＋Native HCA、性能CSA＋Native HCA
在全部16个DP上逐token、草稿轮数、提出/接受计数、直方图及逐轮接受事件都一致的请求。
这是正向复现，不替换[低接受率基准](../low_acceptance_20261001/README.md)。

## 固定输入

共检查128K/B24和8K/B40的64条请求；选出9条，不重复填充请求：

| 来源 | 请求 | 首轮实际接受个数 | 后续轮次 |
| --- | --- | --- | --- |
| 128K | q05、q11、q17 | 1、3、5 | 均为5 |
| 8K | q05、q11、q17、q23、q29、q35 | 1、3、5、4、4、5 | 均为5 |

真实历史长度为131104～131111和8224～8231；不是精确131072/8192。
9条请求按长短交错顺序放入一个B9批次，每条生成192 token。
原先为B24/B40，新批次的图档位为6/54；重新组批后是否仍一致必须实际验证。

所选请求都是easy_repeat，历史均值4.952055/轮；6条只在首轮有拒绝，3条全接受。
没有覆盖接受0和2，没有覆盖持续的复杂回退。不得据其通过证明其他输入不存在功能错误，
也不得因此把原基准中未定位的差异认定为可接受量化误差。

[prepare.py](prepare.py)读取原始逐请求记录，生成[筛选记录](selection.json)和
[固定参考](selected_reference.json.gz)。派生bank只链接既有Native bank中的相同物理文件，
保持prompt、target/draft cache/state不变，不重跑prefill、不做hash或全量cache扫描。
所有64条请求的入选/排除原因都在筛选记录中；不因复测结果重新挑选请求。

## 功能检查和复测

先用现有单卡入口对两版各检查一次新的B9形状，history=131111、layer_index=2、正式权重、合成历史：

- Native及PTO各自重复执行的一致性。
- 固定地址的A→B→A图输入更新，输出和cache/state与对应eager结果逐元素一致。
- 满批→8条有效请求→1条有效请求→满批，检查补位、写入保护区及只读metadata。
- Top-K合法性和输出/状态有限值。

这部分覆盖合成输入下的实现检查，不冒充所选真实请求的逐层bit验收；不做性能计时或profiling。

再分别执行全Native、精度CSA＋Native HCA、性能CSA＋Native HCA三个16卡任务，保持Native HCA。
检查三方彼此一致、与原批次筛出的固定参考一致、同次跨DP一致、抢占为0，
并核对实际21个CSA/20个HCA选择、图覆盖和逐请求计数与框架统计。

保持CANN9.2.0-beta.2、TP1/DP=EP16、TMR、NZ2、atomic0、确定性level1/HCCL确定性、EPLB关闭；
沿用当前整模型FULL_DECODE_ONLY与static kernel入口、生产inplace_pass=False。
Native进程event=0、PTO进程event=1，继续披露该运行时差异，不单凭本轮结果给算术原因定性。
用例显存利用率0.95、默认watermark=0，不引入额外容量策略。

复用上一轮冻结的`source_v9_sync_load`：从算子提交84dc9a3f到当前a37df1fa没有生产源码变化，
本轮PyPTO/Simpler提交和干净工作区均已确认。排队期间不编辑冻结源码。
版本与任务见[source.json](source.json)、[tasks.json](tasks.json)。

## 复现命令

从仓库根目录执行，所有NPU运行都经统一队列。原始大体积bank须已在本机；新目录不能覆盖旧结果。

```bash
# 已生成的bank：tests/pypto_test/results/matched_requests_20261001/bank
# 在新目录重新生成bank与筛选参考，仅CPU：
PYTHONPATH=tests/pypto_test python tests/pypto_test/matched_requests_20261001/prepare.py \
  --source tests/pypto_test/results/low_acceptance_20261001 \
  --bank /new/bank --evidence /new/evidence

# 先单卡两版检查；脚本一次依次执行两版：
/data/server-toolkits/pto-task/app/task-submit --device auto --device-num 1 --max-time 1200 \
  'bash /path/to/repo/tests/pypto_test/matched_requests_20261001/run.sh guard /new/guard /frozen/source 131111'

# 单卡通过后，在已有父目录下分别执行三个组合；configuration为
# native、csa_precision_native_hca、csa_performance_native_hca：
/data/server-toolkits/pto-task/app/task-submit --device auto --device-num 16 --max-time 1200 \
  'bash /path/to/repo/tests/pypto_test/matched_requests_20261001/run.sh model /new/model/native /frozen/source /new/bank native'

# 三组结束后的CPU比较，不再占卡：
PYTHONPATH=tests/pypto_test python tests/pypto_test/matched_requests_20261001/compare.py \
  --root /new/model --bank /new/bank --reference /new/evidence/selected_reference.json.gz \
  --guard-root /new/guard --output /new/comparison.json
```

通用入口的`acceptance.qualified`仍检查0～5全覆盖，所选正向子集会显示false。
[compare.py](compare.py)单独记录这一限制；三方一致性仍严格检查逐轮事件，不只比较平均接受数。

本轮实测结果见[RESULTS.md](RESULTS.md)。
