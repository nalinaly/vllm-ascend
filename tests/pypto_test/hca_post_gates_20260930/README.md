# mHC post 门控整理与组合增量（2026-09-30）

CANN 9.2 beta2、NZ2、atomic0/det0、BF16 残差、正式第3层权重。
只改 HCA 局部后处理；Q 分组以第121节的已测版本为组合基线，不改共享 CSA、Native 或工具链。

## 实现与参考

每8行把 post/comb 的两块门控在 UB 内转置，以静态列切片取代20次 Gather 和索引构造。
生成代码是两次 TTRANS、无 TGATHER；任务划分、四路残差的 UB 复用、八组 INT32 反量化累加顺序、
attention 落 BF16 再转 FP32 的舍入点，以及 post*x 后依次加残差0/1/2/3 的顺序全部保持。

参考本地 ops-transformer28f40354 的 `mhc/mhc_post/op_kernel/arch22/mhc_post_arch22.h:203–238`：
该实现复用四路输入，以标量 GetValue/Muls/Axpy 处理系数；它本身没有本候选的转置，
也不能据此断言安装的 Native 一定选择这个模板。
pypto-lib2164563 的 `models/deepseek_v4_flash_dspark/hc_post.py:37–77` 是逐 token 读取标量、
每 task 四 token，并采用 FP32 残差/输出。这里保持 Native BF16 残差且融合 O-B 反量化，
以8行系数列匹配现有批量计算，不能直接照搬其整套精度契约。

## 独立与组合测试

同进程 ABBA，每侧10次真实设备 span，单位 μs。不同轮只能和该轮基线比较。

| 档位/增量 | base min/max/mean | candidate min/max/mean |
| --- | --- | --- |
| 128K/B16，仅门控转置 | 598.250/646.500/615.225 | 587.750/625.000/606.650 |
| 8K/B24，仅门控转置 | 609.750/656.250/630.125 | 609.250/633.500/621.000 |
| 128K/B16，Q分组上增加门控转置 | 580.250/622.500/591.550 | 568.250/587.000/576.550 |
| 8K/B24，Q分组上增加门控转置 | 618.250/672.000/630.200 | 606.000/620.250/613.775 |

组合增量均值分别少15.000/16.425 μs，8:2相对加权约−2.550%；max分别少35.50/51.75。
这证明在当前 Q 分组基础上仍有收益，因此接入生产。不能把这15 μs 与第121节独立窗口的
22.325 μs 相加，称为一次实际测得的总收益；单卡样本 max 也不代表 EP16 尾延迟已经解决。

独立同卡 DFX 中，post 的48 worker min/max/mean 从26.26/32.44/28.8325变为
23.94/29.84/26.4096 μs，核内均值−8.403%，组跨度32.74→30.16。
DFX 另一次采样，不把其中其他组的时序变化混作 ABBA 整段收益的唯一原因。

四类完整输出/cache/state 跨版本逐 bit、自身 eager/graph 与保护区全部通过。
组合版16507/B3含2行尾，同址补位3→2→1→3通过；cycles=1只作功能证据。
生产后处理完整模块 AST 与已测源码一致，最终组合生产代码依赖图解析、CPU 编译及 load 通过。

## 当前 Native 同卡对照

Native→PTO→PTO→Native，同卡3、每轮9次真实 span，每侧18次。Native 实际 static 编译、
SuperKernel 标志与安装包均已核实；PTO SK0，双方 dynamic=False、inplace 开启。
区间包括 mHC pre/norm/HCA/O/post，正式第3层权重。

| 128K/B16 | min | max | mean | P50 |
| --- | --- | --- | --- | --- |
| Native | 531.250 | 595.250 | 551.431 | 548.625 |
| PTO，Q分组＋门控转置 | 554.500 | 628.250 | 578.014 | 576.250 |

PTO mean 仍慢4.82%，P50慢5.04%。以本次 Native mean 计算，快20%的界线为441.144 μs，
还需减少136.869 μs；目标未达到。没有新七档或整模型 token 验收。
Native compiled 与 eager 的零容差差异如实保存在每轮报告中，未出现非有限数或保护区破坏。
本次是性能对照，没有进行跨实现逐元素验收，不把 Native 自身诊断误称为 token 验收。

## 证据与复现

[source.json](source.json)、[summary.json](summary.json)、[incore.json](incore.json)、
[native_anchor.json](native_anchor.json)、[tasks.json](tasks.json)、[production_ast.json](production_ast.json)。
summary 仅汇总变体之间的 ABBA，Native 单独见 native_anchor。

原始数据在 `../results/hca_post_gates_20260930/`；独立泳道为
`dfx/{base,transpose}/dfx/merged_swimlane.json`，Native 四轮在 `native_anchor_h131072_b16/`。
`prepare.py` 从 f14d8d90 生产状态创建只读私有包，`prepare_combo.py` 在 Q ready_v2 上叠加；
已存在目录禁止覆盖，新实验须另命名。

排队示例：`task-submit --device auto --max-time 900 --run 'bash tests/pypto_test/hca_post_gates_20260930/run.sh combo 131072 16'`。
汇总复用 `hca_mix_schedule_20260930/collect.py --experiment-root tests/pypto_test/hca_post_gates_20260930`；
Native 汇总用本目录 `collect_anchor.py`，检查实际编译选项、设备/CANN、采样数和保护区。
定向 Ruff、shell 与 diff 检查通过；全仓 `format.sh ci` 因缺 pre-commit 未通过，未声称全仓检查成功。
