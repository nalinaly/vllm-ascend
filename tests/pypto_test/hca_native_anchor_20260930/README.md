# a7a9f314与Native部署口径的128K/B16新基线

同一队列任务、同卡1、ABBA顺序Native/PTO/PTO/Native，正式第3层权重。
CANN9.2、NZ2、atomic0、det0；两侧`torch.compile(backend="npugraph_ex", dynamic=False)`、
`inplace_pass=True`、`static_kernel_compile=True`。包含mHC pre/norm/HCA/O/post，不含FFN。
PTO使用冻结的a7a9f314整包，未混入本轮Q实验。

Native启用SuperKernel，两次均实际静态编译成功且安装1个包，编译器收到True标志。
PTO走kernel模式服务入口；此侧没有可静态编译的aclnn序列，装包0且观测到3次PTO调用。
PTO的SuperKernel关闭：首轮开启时在图优化报107017无效funcHandle，不能将其当作已支持。
初次失败轮的Native数据没有混进下表；修改PTO开关后完整ABBA重跑。

## 实测

每pass预热5次、20次图外Event，独立profile采9次设备重放；下表每侧共18次真实设备span。
**Event含主机派发，不用来当Native/PTO验收值**。单位μs：

| 实现 | min | max | mean | P50 |
| --- | ---: | ---: | ---: | ---: |
| Native static + SK | 533.25 | 562.25 | 544.35 | 541.50 |
| PTO a7a9f314 | 590.75 | 646.50 | 614.94 | 609.50 |

PTO mean慢12.97%，P50慢12.56%。20%目标要求本档P50≤433.20μs，尚差176.30μs。
当前多个PTO前后核内收益并没有达成对Native领先20%；不重新解释目标，也不引用旧版本较慢Native掩盖差距。
Native两pass P50为550.75/538.25，PTO为610.50/608.50，原始样本和pass明细见[summary.json](summary.json)。
不把本档外推为七档验收或整模型性能；旧绝对值也不能直接与本轮跨卡相比。

## 数值边界

四pass保护区均通过，全部诊断张量无非有限值；PTO自身eager/图重放四类完整张量逐bit一致。
Native编译图相对eager的零容差比较有差异，两pass相同：输出15814元素不等，max_abs=0.015625；
SWA cache有9元素不等，max_abs=0.0078125；compressed/state精确相同。
没有以此单独认定模型精度失败，也没有放宽测试后宣称PASS；编译/fusion带来的具体来源和token影响尚未验收。
本入口没有跨Native/PTO逐元素比较，不能把以上自身诊断当作跨实现精度结论。

## 复现

`run.sh`必须经`task-submit --device auto --run 'bash /绝对路径/run.sh'`，不重新映射卡号。
`collect.py`只读取完整成功轮，检查编译配置、实际静态包、同卡、9次profile样本及保护区。
结果目录必须新建；不要覆盖历史失败或成功报告。

- 首次失败：`task_20260930_032122_216888230045`，退出1，目录`../results/hca_native_anchor_20260930/h131072_b16/`。
- 正式ABBA：`task_20260930_032530_227722523167`，退出0，目录`../results/hca_native_anchor_20260930/h131072_b16_compatible/`。
