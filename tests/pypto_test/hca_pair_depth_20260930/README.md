# 双query三槽：用Q重载换回两步预取（2026-09-30）

控制是第128节`head_extract`双query完整UB候选，尚未接入生产。
当前生产0742f07c不变。候选只在长档改变流水深度，在线max/BF16概率、FP32累积、mask/cache保持。

最新本地ops-transformer28f40354的experimental sparse_attn_sharedkv使用PRELOAD_NUM=2，
两query候选因L1容量只用两个KV槽、一步预取。本轮用三个KV槽、两步预取，仍逐pair排空。
为容纳三槽，将128×512 BF16 Q移到每块QK内部重载，使PV阶段可以复用它的L1区域。
这是用额外Q读取换流水重叠，需要实测，不能仅凭与Native预取数相同判为收益。

CPU依赖图、编译与load通过；[memory.json](memory.json)记录分配后地址上界：

| 内存 | 字节 |
| --- | --- |
| Mat/L1 | 524288 |
| Acc/L0C | 131072 |
| Left/L0A | 65536 |
| Right/L0B | 65536 |
| Vec/UB | 188160 |

采用第128节修正后的动态head extract，保留八片N64完整FP32 UB累积，不退回旧GM累积方案。
源码为[source.json](source.json)中的冻结副本；[depth3.patch](depth3.patch)精确记录增量。
patch为零上下文，应用到head_extract需`git apply --unidiff-zero`。
真机通过task-submit自动排队，句柄在[tasks.json](tasks.json)。未测结果不能写成精度或性能通过。
CPU复用`hca_residual_reuse_20260930/compile.py`，真机复用`hca_pair_screen_20260930/run.sh`。
