# UB环动态槽偏移的生成代码问题

本轮PyPTO版本为`88f605986`，PTOAS及CANN沿用公共环境，未修改依赖仓库。
v1设备任务`task_20260930_020841_107403616628`在跨候选逐bit比较时失败，未进入性能计时。
两侧各自eager/graph重放通过；这不能代替跨实现的数值检查。

v1读法为：

```python
out_m = pl.reshape(pl.tile.slice(
    updated_max_ring, [1, H // 2], [out_work % QK_TRANSFER_SLOTS, 0],
), [H // 2, 1])
```

最终IR仍有`[out_work % 3, 0]`，但生成的AIV C++把切片和后续reshape都绑定至环的首地址。
具体文件：`compiled/ring/kernels/aiv/hca_unified_attention_aiv.cpp`，约1628行：

```cpp
// t__tmp_v339，预期是动态槽的一行
uint64_t v307 = (uint64_t) v35;
TASSIGN(v306, v307);
// out_m，仍是同一固定地址
uint64_t v309 = (uint64_t) v35;
TASSIGN(v308, v309);
```

写入端则正常使用`vec_tick % 3 * 128 + ring_base`，故读写槽位不一致。
这是本组合的实测问题；不外推为所有动态slice均有问题，也不在本会话修改公共编译器。

v2用显式复制表达读取：

```python
out_m = pl.reshape(pl.tile.extract(
    updated_max_ring, out_work % QK_TRANSFER_SLOTS, 0, [1, H // 2],
    target_memory=pl.MemorySpace.Vec,
), [H // 2, 1])
```

生成码变为带动态行偏移的`TEXTRACT`，不再依赖切片视图经reshape传播偏移。
`compiled/ring_v2/kernels/aiv/hca_unified_attention_aiv.cpp`约1641行可核对。
v2任务`task_20260930_021255_114737517494`完整输出/cache/state逐bit一致。
此结论依赖生成代码和整层对照，尚未整理成编译器的独立最小复现或上游issue。
