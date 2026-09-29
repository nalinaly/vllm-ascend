# HCA 完整有效压缩块的 mask 快路径

基底2cb8714b（已保留residual复用）。只改长档attention，短档函数和调度参数不变。
参考最新本地ops-transformer：
`experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_scfa_block_vector.h`
的`DealBmm1ResBaseBlock`将缩放后的score与实际有效列数交给`SoftmaxFlashV2Compute`。
这里仅借鉴省掉无效工作的方法，不改变PTO的分块max、BF16概率及running归约规则，
也不声称与Native的softmax量化顺序相同。

基线每个压缩K128块都加载mask，计算bias、广播加、设置valid_shape、fillpad，
再将exp乘以mask。候选满足以下两项时直接对scale×score做row_max/exp：

1. 当前query确实可见完整的128行（因果长度判断）。
2. gather实际成功搬入128行（每个页号、页表范围、有效长度均已检查）。

任何一项不满足均走原来的通用分支；不根据128K标签或页表容量省略保护。
gather新增一份行数表，逐项相隔16个INT32，即每项独占64字节；不同worker只写不相交的项。
这是为避免scalar write以cache line回写覆盖邻项。编译器对动态索引仍给保守提示，
生成代码确认使用item×stride写入，stride为16；不把提示消失作为正确性的前提。
代价为每个块额外一次标量发布/读取及运行时分支，是否抵消向量收益必须实测。

首份CPU候选发现嵌套分支缺少外层yield，同时发现行数表未按cache line隔离，均未上卡。
v2补齐分支返回，并将表改为64字节行距，完整CPU trace/编译/load通过。
脚本冻结v2完整算子包，只读源路径见source.json；不修改共享PyPTO/Simpler。

任务task_20260930_013917_58586026298，单卡128K/B16同卡ABBA：
`../results/hca_fullmask_20260930/h131072_b16/`。
任务已完成exit=0，候选未接入生产。每侧18次单次设备重放，单位μs：

| 实现 | min | max | mean |
| --- | ---: | ---: | ---: |
| base | 557.25 | 624.25 | 582.63 |
| fullmask | 578.75 | 645.25 | 605.39 |

独立DFX：attention AIC核内mean148.04→166.96，AIV152.03→171.05；
对应AIV最大值155.62→174.18，整组跨度155.80→174.26。
gather核内mean26.33→21.04，但组跨度34.80→36.48；这也说明核内均值与整组收尾不能互相替代。
没有观察到目标attention任务的核内收益，不追加短档或边界扩测，不保留该实现。
新增scalar读取/分支或其生成码可能抵消向量收益，单份DFX不足以唯一归因，
只记录该组合失败，不推出所有mask快路径均无价值。

完整输出/cache/state在ABBA四份中与首份base逐bit一致，自身eager/graph与保护区通过；
这只覆盖本档正常页表，不冒充无效页、压缩边界或Native模型验收。
原始路径和样本见`result_h131072_b16.json`。CPU失败日志本地忽略，仅保留上述修正说明。

结果提取不再上卡：

```bash
source ../env-dsv4-0251rc1.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 python \
  tests/pypto_test/hca_residual_reuse_20260930/collect.py \
  --experiment-root tests/pypto_test/hca_fullmask_20260930 --candidate-label fullmask
```
