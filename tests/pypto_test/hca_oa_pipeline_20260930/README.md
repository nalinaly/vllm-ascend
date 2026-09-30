# O-A 显式两层流水实验

按第135节的核占用差距排序，先优化O-A。基线为生产0742f07c，
128K/B16的O-A满24核占用当量为PTO62.64μs、Native28.38μs。
差值用于排序，不是可直接回收的整层延迟。

## 改动与来源

参考本地`ops-nn/matmul/transpose_batch_mat_mul/op_kernel/pp_matmul_ein_sum_kernel.h`
第277–343行的L1下一K段预取与L0 ping-pong。此源码仅提供策略参考，
不表示已证明安装版Native在该形状使用同一模板。

仅在冻结私有包的`deepseek_v4_flash_dspark_perf/decode_o_proj.py`中修改N128路径：

- L1由K256改为K512、stage=2；L0显式K128、stage=2。
- 单一FP32累加器，首个K128通过`init_cond`初始化，其余按原顺序累加。
- 保留8组、每组8个任务、原依赖和early-resolve；保持quant/O-B分组衔接。
- 直接读Native三维NZ权重；仅在L1去掉长度为1的组维，无GM重排。
- N256路径继续使用原实现；公共CSA文件与生产HCA没有改动。

生产生成码已有两个32KiB的L0地址，但其首段与主体由不同matmul展开而来。
不能继续沿用旧LOG55中“生产根本没有双缓冲”的笼统结论；本候选改变的是
明确的L1加载粒度、两层循环及累加器初始化组织，是否提速由实测决定。

## 编译与容量

`nested`首版在CPU编译的AccCompactValid检查失败：动态有效M小于物理128行，
显式Acc种子不能按全128行间距读回。这是布局问题，不是允许放宽的精度误差。

`compact`新副本为种子声明compact并设置有效行数，依赖图、完整设备编译及load通过。
当前`compact=True`在PyPTO文档中属于内部编译器接口；若实测值得采用，
需优先改为由`tile.matmul`推导种子或受支持的窄行种子写法，不能将实验接口风险藏入生产。

| N128资源 | 生成码实际字节数 | 分配 |
| --- | ---: | --- |
| L1 / Mat | 524288 | A/B各两个131072字节缓冲 |
| L0A / Left | 65536 | 两个32768字节缓冲 |
| L0B / Right | 65536 | 两个32768字节缓冲 |
| L0C / Acc | 65536 | 一个FP32累加器 |

来源见[memory.json](memory.json)和[CPU编译结果](compile_compact.json)。
CPU通过不等于NPU正确或性能通过。

## 验证

已通过自动单卡队列提交`task_20260930_094555_174788815257`，
测试真实第3层128K/B16，同进程ABBA五组、每侧10个设备跨度。
复用现有脚本检查完整输出、三类cache/state跨版逐bit、重放及保护区，
同时报告min/max/mean/P50。首先验证纯流水改动的等价性。
任务句柄见[tasks.json](tasks.json)，状态须以`task-submit --status`为准。

后续按结果处理：核内DFX检查O-A及quant/O-B并发；有价值时补短档和尾块，
再考虑正式接入。不得将排队或CPU通过记成性能收益，也不先扩到七档/整机。

重建冻结候选：依次运行`prepare.py`、`prepare_compact.py`，
需要新的空目标目录；禁止覆盖已经排队的私有包。
补丁顺序为[nested.patch](nested.patch)再[compact.patch](compact.patch)，均为零上下文。
