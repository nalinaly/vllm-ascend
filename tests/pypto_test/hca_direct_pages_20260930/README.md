# Cube分页直读：省掉压缩KV的GM暂存（2026-09-30）

状态：`direct_store` 已通过CPU依赖图、完整层编译/load和独立Attention编译/load。
单卡独立边界和整层逐bit检查通过，但完整层明显变慢，当前不接入。
生产仍0742f07c，控制为未接入的qr_late候选。独立泳道仍排队。

## 为何重审旧结论

长期日志第57.3节仅推算了把AIV gather移入每个query的成本，没有实现或真机测量。
当前Cube本来就为同请求六个query分别从`cmp_work_kv`读KV；
直接按页装入同一个L1环形缓冲区，不需要先把AIV的清零、复制、mask构造全部重做六遍。

128K/B16的有效压缩历史为16×1024×512 BF16，16MiB。
原路径先读16MiB、写16MiB到暂存，Cube再为96个query读96MiB；
新路径保留Cube的96MiB，移除前面的至少32MiB逻辑读写（这里未计原padding搬运）。
新代价是每128行从一次连续加载变为四次32行加载、更多页表查询和分支。
不是实测HBM流量，也不据此保证更快。本地最新Native sparse_attn_sharedkv的
`op_kernel/arch22/sparse_attn_sharedkv_swa_block_cube.h`同样在Cube中处理pageAddress的ND2NZ加载；
这里只参考策略，不声称复制了其全部多query/预取实现或已解析安装二进制的实际tiling。

## 私有候选具体行为

控制来自`hca_qqueue_groups_20260930/qr_late`：Q24四组、QR禁止提前解析，
Attention仍使用生产的128列块内max/BF16策略，未混入online或512列候选。

- 只改长档compressed路径；raw、小上下文、权重和外部Native cache布局不改。
- Cube按Native页表将32行页写入原三槽L1，满页直接读；部分页先清零再写有效行，
  无效/负/越界页只读共享零页，防止零概率乘上未初始化NaN。
- 保持原QK/PV分块、跨query两步预取、每query归约顺序、sink及逆RoPE。
- AIV只生成有效mask和一个32×512 BF16零页，没有大KV暂存。
  每页32个FP32 mask直接写GM的对应列，地址对齐，不使用不支持的32→128 UB移动。
- metadata只读取调用输入的页表和长度，可在cache写回前准备；Attention显式依赖
  metadata以及压缩cache写入完成。生成编排确认两个TaskId均在依赖列表中，
  metadata参数不含cache数据，Attention直接接收原cache的reshape视图。

多task重点：原gather与四组反量化交叠，并影响最后反量化完成；
新小metadata可更早完成，但直接分页加载会增加Attention核内工作。
必须比较完整HCA及Q/DQ/metadata/Attention/O-A时序，不能只报告“删除了gather”。

## 编译修正与验证要求

冻结版本依次为`direct`、`direct_mask`、`direct_cast`、`direct_store`。
前三版分别暴露scalar-left减法、INDEX直接转FP32、32列assemble到128列的TMOV形状限制。
最终在算子侧使用Tile左侧算术、INT32中转及每页直接store解决；未改共享工具链。
生成码地址上界见[memory.json](memory.json)：Mat475136、Acc65536、Left32768、Right65536、Vec181856。

`run_probe_and_pair.sh`通过task-submit占用一张卡：

1. 独立Attention比较131072历史与16507尾块/负页/空请求/NaN补位，检查自身图重放和输出保护区。
   helper里的`online`标签在这里指direct_store，真实路径见[probe_sources.json](probe_sources.json)。
2. **必须与控制逐bit相同**，不允许512列实验的算术差异豁免；不一致则停止，不继续性能测试。
3. 通过后再测真实第3层128K/B16同进程ABBA，检查完整层输出、三类cache/state、重放和保护区，
   报告min/max/mean与全部小组。独立FP64结果和单层精确一致不替代最终模型token验收。

测试句柄见[tasks.json](tasks.json)，仍存活的任务不重复提交。
有实际收益后才做必要的短档/七档与Native对照；不把较慢实验控制的增量收益当成生产收益。

## 完成结果与剩余问题

task_20260930_090117_20745210527已exit0。独立131072/16507两档的输出与控制逐bit，
各自图重放、保护区以及空请求零输出通过；实际层的输出、三类cache/state也全部逐bit，
图重放及保护区通过。没有算术精度策略变化，没有新增模型token验收。

CANN9.2/NZ2/atomic0/det0、真实第3层、128K/B16，ABBA每侧10次，min/max/mean（μs）：
qr_late控制 **561.000/590.500/574.150**，direct_store **627.250/654.750/640.050**。
mean增加65.900μs，当前方案不接入。不能仅凭移除32MiB暂存读写就声称高效。

性能fixture继续使用`dsv4_csa_single_layer.make_fixture`的Native BlockTable：
每请求物理页按倒序映射，没有为候选更改成连续正序。因此不能省略页表或假设四页可以
直接按逻辑顺序用一次128行加载；负页、尾块路径另由独立probe覆盖。
页表查询、分段加载以及metadata提前运行对调度的影响仍需分开诊断。
已排队采集同卡控制/候选独立泳道；没有证据前不指定唯一瓶颈，不扩大无收益版本测试。
完整结果见[summary.json](summary.json)、[probe_result.json](probe_result.json)。

## 复现来源

源码身份见[source.json](source.json)，所有包为私有只读副本。
`prepare.py`生成第一版；然后依次运行`prepare_mask.py`、
`prepare_mask.py --index-cast`、`prepare_mask.py --page-stores`。
精确冻结增量依次为`direct.patch`、`mask_fix.patch`、`cast_fix.patch`、`store_fix.patch`，
零上下文patch用`git apply --unidiff-zero`；生成器排版可能不同，以冻结patch为本轮编译身份。
CPU复用`hca_residual_reuse_20260930/compile.py`，独立probe复用
`hca_online_softmax_20260930/probe.py`，未增加重复的通用测试框架。
