# mHC pre 访存与多任务分工实验（2026-09-30）

基线 ad0e6bbe，私有整包冻结在 `.cache/hca-pre-ub-ad0e6bbe-20260930/`。
三个8行版本均未接入：mix 核内没有收益。4行有效分工+D512已保留核内收益，整段提速尚未证实。
只修改算子私有包，不改公共 PyPTO/Simpler/PTOAS/PTO-ISA。

参考最新本地 ops-transformer 28f40354 的
`experimental/mhc/mhc_pre_sinkhorn_premix/op_kernel/mhc_pre_sinkhorn_premix_base.h:583`：
`ProcessY` 在 UB 中将原始输入转换为 FP32，完成乘法/规约，再转回输出类型。
本仓 Native `csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h:407` 同样直接从原始输入做混合。
没有声称安装包当前形状必然使用所查模板，也未引入训练接口。

pypto-lib 2164563 的 `hc_pre_norm` 与当前基线仍把 BF16 混合值存到 GM，统计完成后再读回归一化。
基线 HCA 已将门控固定顺序归约融合进消费者，widen 时顺便做前段 RMS；这里继续保留这些实现。

## 已完成的访存候选

- `mix_ub_v7`：8×4096 BF16 混合值在 UB 按16个连续8×256块保存；RMS统计也留UB。
- `mix_gm_bf16`：混合值仍走 GM，mix 直接读原始 BF16 再转 FP32，线性投影仍读 FP32。
- `mix_ub_bf16`：叠加上述两项。

三者保持12个mix worker（B16）、D256 stage2、split0→3、四路混合加法、BF16舍入与RMS顺序。
静态 Vec 分配边界分别131360、74016、139552字节，均在当前188416配置内。
最初横向组装8×4096 UB子视图不满足A3 TMOV形状约束，改成连续块布局后显式编译/load通过。
Tensor/Tile API、scratch及store接口的CPU诊断已修正；失败版本未占NPU。

128K/B16，同进程 ABBA 每侧10次，CANN9.2/NZ2/atomic0/det0/正式第3层权重。
npugraph_ex dynamic=False、inplace/static开、PTO SK关。真实设备span，单位μs：

| 候选 | 自己的base min/max/mean | 候选 min/max/mean |
| --- | --- | --- |
| UB | 551.50/588.25/571.48 | 574.25/592.75/582.73 |
| BF16输入 | 576.50/742.25/606.90 | 571.50/606.75/588.40 |
| UB+BF16输入 | 550.50/582.00/561.88 | 558.75/591.00/572.65 |

BF16输入轮 base 有一次742.25慢点，仍完整保留；P50 588.38→588.75，没有普遍时延下降证据。
不能只根据mean/max下降宣称稳定收益。其他候选mean/max均回退。

独立DFX（单窗口，仅作核内和联动观察）的mix min/max/mean：

| 版本 | mix核内 | mix启动—结束 | Sinkhorn启动—结束 |
| --- | --- | --- | --- |
| base | 37.58/39.24/38.47 | 65.42—105.50 | 66.34—103.08 |
| UB | 40.44/42.14/41.03 | 62.86—105.00 | 63.54—100.02 |
| BF16输入 | 39.66/40.66/40.23 | 68.12—109.42 | 68.82—104.52 |
| UB+BF16输入 | 41.38/42.32/41.93 | 72.10—114.64 | 73.14—109.30 |

UB 虽省 GM 搬运，但还需 UB 组装/提取；BF16读取节省流量但增加转换。
核内实测均增加，未以流量估算代替收益。Q投影等下游也改变起跑/跨度，详见 [incore.json](incore.json)。
三候选四类完整输出/cache/state跨版本逐bit、自身eager/graph与保护区通过。
无新短档、七档、Native跨实现精度或模型token验收；未扩大无收益候选的测试。

## 4行分工的后续候选

`prepare_m4.py` 将mix的有效行改4行，保持物理8行以满足FP32列向量32B对齐。
B16的mix由12→24worker；并发Sinkhorn仍12worker、权重预热仍8worker，实际重叠以DFX判断。
pre门控直接留UB，用1×8扁平Gather再reshape，避免窄转置或窄行Gather的16B/4B对齐限制。
所有读写明确限制有效行；不再使用原先只有单个尾块的共享暂存，避免多worker竞争。
`prepare_m4_wide.py` 额外用D512读写/计算，统计仍提取两个D256按原次序求和。
保持stage2，不把512列整体规约造成的舍入变化混入本轮。

CPU失败边界：直接物理M4的FP32列只有16B；物理M8但Gather输出8×1行主序每行只有4B。
现采用M8物理盒和1×8 Gather输出。失败版本只做CPU编译，不能标设备验证通过。

两版CPU编译/load和长档完整状态通过；M4/D256整段min/max/mean
551.00/592.00/572.23→556.75/598.50/575.88，mix核内mean37.885→31.82。
M4/D512核内更好：37.060/38.620/37.885→25.160/28.800/26.548，mean下降29.93%。
组跨度38.64→29.08，QA启动110.06→102.78；Sinkhorn结束99.40→103.26。
保留实际联动数据，没有将核内节省直接折算成完整HCA节省。

M4/D512同进程ABBA各10次，min/max/mean：

| 档位 | base | M4/D512 |
| --- | --- | --- |
| 128K/B16 | 548.50/597.25/569.23 | 556.00/589.25/571.18 |
| 8K/B24 | 592.00/650.00/613.40 | 602.50/631.75/615.88 |

两档mean分别+1.95/+2.475，8:2加权相对mean约+0.355%；样本max分别−8/−18.25。
没有稳定整段收益结论。依用户保留核内收益的要求，接入M4/D512，继续调度联动。
长短档四类完整状态、自身图重放和保护区精确；16507/B3验证最后worker仅2行及补位3→2→1→3。
B3仅cycles=1功能检查，不能用其性能样本声称提速。未新增七档/Native/模型token验收。

生产仅改`deepseek_v4_flash_hca/hc_pre_fused.py`及root传入原始BF16的调用。
与已测候选相比只删除未使用导入和两个已被DCE消掉的尾块暂存；其余AST相同，生产显式编译/load通过。
未改残差类型、Native路径、共享CSA实现或依赖仓。

## 原始文件与复现

[summary.json](summary.json) 保存每轮配置、完整状态和原始报告路径；[incore.json](incore.json) 保存独立DFX。
所有耗时都从已有报告提取，运行 `python tests/pypto_test/hca_pre_ub_20260930/collect.py` 可重建汇总。
生成器按父快照逐项复制，patch给出相应源码差异；CPU编译使用既有
`hca_residual_reuse_20260930/compile.py --operator-source ... --output ...`，显式解析两入口并compile/load。

设备测试必须通过 `task-submit --device auto`：`run.sh SIDE HISTORY BATCH` 为同进程ABBA；
`run_bf16.sh` 顺序测两BF16候选；`run_dfx.sh` 独立收集四版泳道。
结果根：`tests/pypto_test/results/hca_pre_ub_20260930/`；
筛选 `{SIDE}_h131072_b16/report.json`，泳道 `dfx/{SIDE}/dfx/merged_swimlane.json`。

已完成 exit=0：UB筛选 `task_20260930_041849_31135763898`；
两BF16候选 `task_20260930_041945_313304127483`；
四版DFX `task_20260930_042105_316448416916`。
M4两版长档 `task_20260930_043138_334282719249`；M4单独DFX
`task_20260930_043408_33806646598`（对应`dfx/base_m4`控制）；
短档与尾块补位 `task_20260930_043707_340971624107`，均exit=0。
脚本Ruff/shell及diff检查通过；生产文件没有新增Ruff问题，保留已有问题范围。
全仓format.sh ci受缺pre-commit限制，未声称全仓检查通过。
