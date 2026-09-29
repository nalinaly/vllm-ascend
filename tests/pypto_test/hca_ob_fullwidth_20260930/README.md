# O-B 整宽量化与完整 K 乘法实验（2026-09-30）

基线0742f07c：Q四组发布＋mHC post门控转置。CANN9.2 beta2、NZ2、atomic0/det0、
正式第3层权重、BF16残差，单卡128K/B16。所有候选冻结整包，未修改生产HCA、共享CSA或工具链。

## 已核实的实现差异

本仓 `vllm_ascend/attention/dsa_v1.py:1744–1758` 的 TP1/A3 路径先做 O-A，
再把8组拼接成 `[tokens,8192]` 交给 wo_b；
`quantization/methods/w8a8_dynamic.py:78–124` 对整个输入做 npu_dynamic_quant 后调用量化矩阵乘。
本地最新 ops-nn19614968 的 `quant/dynamic_quant/op_kernel/dynamic_quant_multi_row.h:182–272`
对完整行归约，独立计算127/amax与amax*(1/127)，采用RINT→ROUND→TRUNC转换链。
没有据源码存在就断言安装Native选择了某个具体模板。

当前PTO和pypto-lib2164563的 `models/deepseek_v4_flash_dspark/decode_o_proj.py:170–275`
按8个1024列组分别取amax、量化、计算INT32 partials，再各乘自己的尺度后在FP32里合并。
这种方式允许每组O-A→quant→O-B独立流水；改成统一尺度会失去这项并发，不能只计算少了多少读写。

本实验恢复O-A落BF16、全8192列取共同amax、O-B整数全K累加、先channel后token反量化。
仍保留既有 `INT8_AMAX_EPS=1e-4` 下界，因此只是对齐Native主要粒度/舍入策略，
未完成Native量化逐元素验收，不能直接声称算术逐bit对齐。

## 功能验证与性能诊断分开

`grouped`参考与`full`候选共享整宽量化，前者沿用8组整数partial再INT32合并，
后者完整K8192乘法，不做8路输出归约。两者长档完整输出/cache/state逐bit，自身eager/graph、保护区通过。
`full_m96`与`full`也逐bit。cycles=1的参考轮只有每侧2个span，只作功能证据。
这些结果验证完整K与分组整数和的等价性，不是Native独立量化参考，更不代表整模型token通过。

与生产性能版比较时，只有 `x_out` 允许记录浮点差异后继续性能筛选；cache/state、
自身eager/graph及保护区仍须精确。公共pair入口新增显式`--diagnostic-output-differences`，
默认仍全部逐bit，报告保留零容差FAIL及`PENDING_INDEPENDENT_REFERENCE_AND_MODEL_TOKENS`。
首轮1572864个输出元素中263150处不同，max_abs=0.03125、RMSE=0.00120844，非有限数0；
不把该误差自动判为精度可接受。数值中性候选仍走原精确门禁。

## 目前实测

同进程ABBA每侧10个实际设备span，μs，每行只与自身基线比较。

| 候选 | 生产base min/max/mean | candidate min/max/mean |
| --- | --- | --- |
| M32/N256/K512 | 546.750/577.000/560.175 | 613.000/620.500/616.550 |
| M96/N128/K512 | 538.500/563.250/547.875 | 590.000/606.750/597.375 |
| M96/N128＋连续行归约 | 566.000/594.750/572.550 | 583.500/612.750/592.100 |

三版mean分别回退10.064%、9.035%、3.415%，均不接入。不扩七档和整机测试。
第三版 `full_m96_rowmax` 保持共同amax，把1024×8窄列归约改成8×1024连续行归约，
然后合并8个最大值；CPU编译/load通过。单独比较相同算术的M96前后版本，每侧10次ABBA：
583.75/603.50/594.05→568.25/585.75/578.50 μs，mean少15.55（−2.618%），max少17.75。
完整输出/cache/state逐bit，保留这个核内改进的私有候选；仍不能把它推广为生产版收益。

## 多任务联动证据

独立同卡DFX，不能与ABBA窗口直接相减。下表是各组首worker实际启动至最后结束，单位μs。

| 任务 | base起止/跨度 | full M32起止/跨度 | 核内mean base→full |
| --- | --- | --- | --- |
| O-A | 515.96–586.32 / 70.36 | 502.42–557.96 / 55.54 | 20.01→16.50 |
| quant | 546.12–600.70 / 54.58 | 559.16–596.60 / 37.44 | 6.74→35.60 |
| O-B | 562.52–623.02 / 60.50 | 602.06–656.34 / 54.28 | 9.05→24.31 |
| 融合post | 629.96–664.46 / 34.50 | 657.32–686.50 / 29.18 | 31.29→26.26 |

base O-B在O-A结束前23.8μs已开始；full必须等所有O-A和统一量化，相关区间串行化。
虽然O-B跨度、post核内和跨度缩短，完整区间反而变慢。不同worker工作量不同，
不能把上述核内mean相加或简单相除，充当整段收益。改归约后的组合必须再单独测。

另一次同卡DFX专门比较M96的两种归约：统一量化48 worker的min/max/mean从
29.08/32.36/30.9938变为9.36/11.92/10.6325 μs，mean−65.695%，跨度32.74→12.02。
O-B跨度45.44→42.32，post跨度26.12→23.74，O-A跨度65.34→67.32；
完整K候选内部的核内改进有证据，但与生产版相比依然有同步等待及组间流水的差异。

## 编译边界与可复现证据

首版至第六版只有CPU编译尝试，无设备执行，具体限制见[compile_failures.json](compile_failures.json)。
最终量化用8个有效副本满足32B物理对齐，在Vector里归约/除法，只读取一个标度；
scale通过有效1元素的MTE3写出，避免用跨worker共享cache line的标量GM写入。
没有修改PTOAS或PTO-ISA。M32量化Vec最高65824字节，矩阵乘Mat294912、L0C32768、
Left16384、Right65536字节，见[capacity.json](capacity.json)。

[source.json](source.json)、[summary.json](summary.json)、[incore.json](incore.json)、[tasks.json](tasks.json)。
原始报告在 `../results/hca_ob_fullwidth_20260930/`，DFX泳道为
`dfx/{base,full,full_m96,full_m96_rowmax}/dfx/merged_swimlane.json`。
`prepare.py`须从0742f07c算子状态创建新命名私有副本；之后`prepare_m96.py`、`prepare_rowmax.py`派生候选。
脚本拒绝覆盖已有副本，源码在提交任务前只读冻结；已有任务的包不得改写。
CPU复用 `hca_residual_reuse_20260930/compile.py --operator-source <私有包> --output <本目录>`。
排队入口 `run_stage.sh`、`run_m96.sh`、`run_rowmax.sh` 接受history、batch；
它们先验证相同算术，再比较生产版，均须由 `task-submit --device auto` 调度。
`collect.py`只读汇总并明确区分功能验证和精度待验收的性能诊断。
定向Ruff、shell、diff检查通过；全仓format.sh ci缺pre-commit，未声称通过。
