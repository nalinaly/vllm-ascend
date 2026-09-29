# HCA Q_B生产者/消费者与并发任务联动（2026-09-30）

基线9015d45a，CANN9.2 beta2、NZ2、atomic0/det0、BF16残差、正式第3层权重。
只改冻结的HCA私有包；共享CSA、Native、PyPTO、Simpler、pypto-lib没有修改。
这是HCA实验，不是CSA七档的新结果；没有新的Native或整模型验收。

## 设计与参考

本地ops-nn19614968的
`matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_basic.h:211–293`
用两份GM中间槽连接Cube的INT32结果与Vector后处理。参考的是生产者/消费者组织，
不是证明安装的Native走此模板。pypto-lib2164563的`qkv_proj_rope.py`仍采用独立投影/反量化。

MIX版每个任务绑定一个Cube和两个AIV，两个head槽轮转，M128/N256/K256 stage2。
保留INT32点积、先token scale后channel scale、完整512列RMS、BF16舍入和原尾块算术。
基线20个Cube任务加48个Vector任务；候选分别试20、24、16组MIX及同步/提前解析联动。
初版静态容量见[codegen_memory.json](codegen_memory.json)，Vector118848字节，未扩大工具链容量。

分组版保持原Q布局，将64个head分为四组，每组5个Cube发布自己的INT32中间结果给12个AIV。
不在Cube尚未完成时绑定AIV等待。`streamed4_edges`试图只对私有Q使用manual_dep，
四组写入互不重叠的head范围；四个生产者TaskId直接作为长短Attention的依赖。
不新增中转dummy。多tile时同组下一轮Cube依赖上一轮消费者，最终TaskId覆盖该组先前写入。
但该版外层reshape重置了manual_dep，仍存在假写依赖，见下面的纠正。
后续`streamed4_flat`直接分配[T,32768]私有Q，物理字节和顺序不变，取消写入前的外层reshape。

## MIX版同进程ABBA

每侧10次实际设备span，单位μs；每行只与自己的base比较，不能跨行相减。

| 档位/候选 | base min/max/mean | candidate min/max/mean |
| --- | --- | --- |
| 128K/B16 MIX20 | 560.500/609.500/575.750 | 562.750/597.750/584.050 |
| 128K/B16 MIX20 MTE2释放 | 566.750/614.500/578.400 | 574.750/593.750/584.425 |
| 128K/B16 MIX24 | 564.750/623.000/578.850 | 567.500/588.000/575.800 |
| 128K/B16 MIX24同步 | 555.000/597.750/572.750 | 589.500/622.750/604.300 |
| 8K/B24 MIX24同步 | 572.750/642.750/602.700 | 599.000/633.250/609.100 |
| 128K/B16 MIX16 | 540.500/587.500/559.500 | 543.500/576.000/555.000 |
| 128K/B16 MIX24关闭提前解析 | 569.500/609.500/591.750 | 581.250/612.250/598.750 |

MIX24的mean下降3.05，但P50为574→575；五组ABBA差值
`[+2.25,-0.75,+2.75,-24.375,+4.875]`，主要受一组较慢base影响，不能称稳定提速。
MIX16的mean下降4.50，P50为558.75→549.50；五组差值
`[-14.625,0,-1.25,-14,+7.375]`，是待验证线索，未据此接入。
样本max下降也保留为线索，但不是EP16尾延迟已解决的证据。
MIX同步、MTE2释放和关闭提前解析无整体收益。以上七次完整输出/SWA/compressed/state
跨版本逐bit一致，自身eager/graph及保护区通过；尚未扩大尾块和全档位验证。

## 泳道解释：不能只看当前task

DFX与ABBA属于独立采样，不用DFX的组跨度替代实际整段耗时。
base与MIX20同卡依次采集；MIX24是另一窗口，只作时序观察。

| 128K/B16 | base | MIX20 | MIX24 |
| --- | --- | --- | --- |
| Q_B Cube核内mean | 60.415 | 73.644（含融合协议） | 63.014（含融合协议） |
| Q_B Cube组跨度 | 69.900 | 162.480 | 141.780 |
| 基线反量化核内mean/组跨度 | 35.398/50.920 | 已融合，不能直接相加比较 | 已融合，不能直接相加比较 |
| 压缩gather组跨度 | 33.460 | 106.400 | 65.780 |
| Attention启动时刻 | 306.880 | 345.020 | 319.780 |

MIX20的前15个块在156.68–172.80μs启动，后五块在233.90–245.52μs于复用核上启动。
这说明实际调度分成两波；同时压缩gather核内mean没有增加，组跨度却显著增大。
融合占用AIV与并发组的等待变化必须联动观察，不能把融合前两个任务mean相加后宣称核内收益。

## 编译与功能边界

- `streamed4`和`streamed4_unroll`：外层head偏移无法证明非负，NZ检查失败；四段字面常量解决。
- `streamed4_const`：编译通过，但完整Q的四个InOut自动产生写依赖，未上卡。
- `streamed4_views`：子视图形状为[T,8192]，生成写回行距也为8192，实际需要32768。
  真机自身eager/graph不一致，在计时前退出1；属于功能错误，没有性能结论。
- 显式TensorView路线未进入真机：JIT拒绝该参数注解；改直接pl.function后又遇到
  下标赋值丢失View类型、有效形状改变不能重赋值，修正后发现JIT依赖收集不包含直接pl.function。
  没有修改工具链绕过检查。相关冻结失败包只用于诊断。
- `streamed4_edges`：恢复完整Q形式参数，生成BF16写回行距32768；四个TaskId均实际出现在
  长短Attention的提交依赖中，CPU解析/编译/load通过，跨版本完整状态、自身图重放和保护区通过。
  但只检查根Tensor的manual_dep和显式边还不够，reshape的新视图又恢复了自动依赖。

## 分组版：必须检查依赖是否真正消除

| 128K/B16 | base min/max/mean | candidate min/max/mean |
| --- | --- | --- |
| 四组，外层reshape | 576.750/623.500/594.925 | 620.750/646.750/630.075 |
| 两组，外层reshape | 579.750/638.000/591.625 | 575.250/610.750/590.025 |

两版均完整状态精确，但不作为独立分组发布的有效性能验证。
四组同卡独立DFX里，四个Cube组约219–232μs完成；四组反量化却依次在
243.40/289.74/334.68/379.54μs启动、280.58/327.34/371.50/402.90μs结束。
Attention AIV启动从309.62推迟到406.80μs；尽管其核内mean185.875→155.767，整段仍慢。

查证`simpler/src/common/tensormap_and_ringbuffer/tensor.h:376–382`：
reshape第三参manual_dep默认false，并直接赋给结果。生成orchestration只传两个参数，
因此根Q设置manual_dep不等于四个q_flat视图都保持该标记，四次InOut仍被自动串联。
撤回“仅因组数增多导致调度开销增加”的推断，也不将两组的−1.60μs当作独立流水的收益。
未修改Simpler；直接二维分配的候选用于在算子侧绕开这一边界。

## 消除假依赖后的结果与取舍

`streamed4_flat`根Q直接为[T,32768]，写入参数仅作Tensor拷贝，manual_dep保留。
最终Attention仍读相同的物理字节，并显式等待四组TaskId。CPU解析/编译/load通过；
长短四类完整状态跨版本逐bit、自身图重放和保护区通过。

| 档位 | base min/max/mean | streamed4_flat min/max/mean |
| --- | --- | --- |
| 128K/B16 | 544.000/583.000/553.150 | 538.000/564.750/550.325 |
| 8K/B24 | 596.750/634.500/611.200 | 600.500/626.500/607.975 |

长短均值−2.825/−3.225μs，相对−0.511%/−0.528%，8:2约−0.514%；样本max−18.25/−8.00。
每档五个ABBA小组均只有三组改善，长档P50仅−0.50μs；不声称稳定跨档或多卡收益。
候选和证据保留供下一轮联动，尚未替换生产9015d45a，也没有扩大尾块或模型验证。

独立DFX确认四组反量化已重叠：三组约237.70/246.90/248.16启动，第四组290.24启动。
第四组启动晚对应其Cube组跨度117.80（其余65.66–68.34）；Cube核内mean仍约60μs，
说明仍有worker晚起跑的调度问题。压缩gather跨度43.08，Attention AIV在332.72启动。
DFX来自另一窗口，不与前一张卡的绝对起点作严格A/B；组内重叠和启动分布本身是有效证据。
接下来优先减少分组生产者的晚起跑，联动检查gather，不把全部任务一并sync_start。

本轮定向Ruff、shell语法、diff检查通过；全仓`format.sh ci`因缺pre-commit未通过。
任务句柄见[tasks.json](tasks.json)。临时解析试错脚本不纳入维护入口，保留失败说明与本地冻结包。

## 复现与证据

- [冻结来源和失败边界](source.json)、[完整ABBA汇总](summary.json)、[DFX核时](incore.json)。
- `prepare.py`→`refine.py`/`prepare_sync.py`/`prepare_late.py`生成MIX候选。
- `prepare_streamed.py`生成常量分组版，`prepare_edges.py`在其上生成直接依赖版。
- 原始报告：`../results/hca_qb_mixed_20260930/<side>_h<history>_b<batch>/report.json`。
- 泳道：`../results/hca_qb_mixed_20260930/dfx/<side>/dfx/merged_swimlane.json`。
- 排队：`task-submit --device auto --max-time 900 --run 'bash tests/pypto_test/hca_qb_mixed_20260930/run.sh streamed4_edges 131072 16'`。
- 仅CPU汇总：`python tests/pypto_test/hca_mix_schedule_20260930/collect.py --experiment-root tests/pypto_test/hca_qb_mixed_20260930`。

输出目录禁止覆盖，重测使用新名字。是否保留要结合核内、并发组完成、Attention启动、
整段min/max/mean；长短档8:2。两个独立优化叠加后必须测组合增量，不直接相加独立收益。
