# PTOAS planner 为什么在 HCA 上卡在 attention

> 2026-09-29。对应优化日志 `DSV4_FLASH_HCA_OPTIMIZATION_LOG.md` §66、§81、§82。
> 变体代码在 `tests/pypto_test/variants_planner_20260929/`。

一句话结论：**把 `memory_planner` 换成 `PTOAS` 能从根上消掉 `PH-MR-001`
（每个 cube 矩阵乘的 L0 乒乓被削掉），非 attention 的部分我已经全部改通；
但 `hca_unified_attention` 重度依赖 Mat（L1）驻留的缓冲，而 PTOAS 对 Mat 的限制
远比默认 planner 严格，四种写法全部被拒。planner 是整个 kernel 编译级的设置，
一个函数编不过就整体编不过，所以连"其余任务在 PTOAS 下快多少"都量不到。**

---

## 一、为什么要换 planner

JIT 的 build output 里有一份此前没人看过的 `report/perf_hints.log`。
128K/B16 那份有 **50 条 `PH-MR-001`**，每个 cube 矩阵乘都在报同一句话：

```
software pipelining requested depth 2 for pipeline group 0 in Right,
but only 1 of 2 buffers fit (32768 B per stage, 65536 B free)
— stages 1 apart share storage and serialize.
The operand would fit depth 2 on its own, but co-resident buffers /
other pipeline groups over-subscribe the space
```

翻译过来：`pl.pipeline(stage=2)` 想要乒乓双缓冲，每片 32 KiB、L0B 有 64 KiB，
**单看这一个操作数是装得下 2 片的**；但同一个 task 里有**多个 pipeline group**
在抢同一块 L0，于是 `MemoryReuse` 把每个都削到 depth 1 ——
"stages 1 apart share storage and serialize"，搬运（MTE1）与计算（MAD）无法重叠。

实测印证（§62.3，pipe 级 profiler 数据）：

| | AIC 忙时/核 | 其中 mte2 在跑 | 占比 |
|---|---:|---:|---:|
| PTO | 449.2 μs | 252.6 μs | **56%** |
| Native | 374.2 μs | 288.2 μs | **77%** |

那 21 个百分点就是搬运与计算没重叠掉的部分。

### group 为什么会多

默认的 **PYPTO planner** 兑现 `pl.pipeline(stage=F)` 的办法是
**把循环体复制 F 份**，每份各自一套缓冲（`LowerPipelineLoops`）。
再加上 `AutoTileMatmulL0` 会给**每个** `matmul` / `matmul_acc` 调用各生成一个
L0 K-loop，于是像 `proj_a` 那样"先剥一个 `pl.matmul` 再用循环做 `matmul_acc`"
的写法就有 2 个 group；`hca_kv_score_proj` 有两个累加器各剥一次，有 4 个。

而 pypto 文档 `dev/passes/30-lower_pipeline_to_slots.md` 写明，
**PTOAS planner** 走的是另一条路：

> `pl.pipeline(N, stage=F)` 表达的是乒乓缓冲的诉求。`LowerPipelineLoops` 用**复制**
> 来兑现……本 pass 用 `pl.MemRef(name, slots=F)` 表达同一个意图，循环只保留**一份**
> 循环体……**而且该流水线本就跳过 MemoryReuse。**
> ……**自门控于 `memory_planner=PTOAS`。**

没有复制就没有"多个 group 争空间"，`PH-MR-001` 从根上消失。
文档还说 `DSA_RP` 与 `PTOAS` "已按实际生命周期放置缓冲"、**dbC=2 是自动的**
（L0C 双缓冲，正是 §67.1 里 Native 有、PTO 没有的那一项）。

### 怎么切：一个调用点参数，不改框架

`pypto.torch.register(kernel, name, *, constexpr=None, config: CompileOptions | None)`，
而 `CompileOptions` 有 `memory_planner: MemoryPlanner | None`。
所以在 `deepseek_v4_flash_hca/native_adapter.py` 的 register 处加一个 config 即可：

```python
return cls(pypto.torch.register(
    decode_hca_tp1_layer_test, "dsv4_hca::attention",
    config=CompileOptions(platform="a2a3", memory_planner=MemoryPlanner.PTOAS),
))
```

⚠ **`CompileOptions.platform` 默认是 `"a2a3sim"`（模拟器），必须显式写 `"a2a3"`**，
否则会静悄悄编到仿真后端。

`DSA_RP` 直接不可用：`DSA-RP could not find a placement for 'proj_a_mm'
within the on-chip memory capacities: canonical greedy found no capacity-fitting placement`。

---

## 二、非 attention 的部分：阻塞项 8 → 1

首次切 PTOAS 有 **8 个函数编不过**，四次迭代降到 1 个：

| 迭代 | 改动 | 修掉的函数 | 剩余 |
|---|---|---|---:|
| 0 | 只换 planner | — | 8 |
| 1 | `A_K_TILE` 256→128、`QR_K_TILE` 256→128 | `proj_a_mm`、`proj_a_mm_0`、`qr_proj_matmul`、`qr_proj_matmul_0` | 4 |
| 2 | `PROJ_A_LARGE_N_TILE` 256→128 | `proj_a_mm_1` | 3 |
| 3 | `qr_rms_norm_quant` 改用 `pl.store(tile, off, out, [valid_rows, 1])` | `qr_rms_norm_quant` | 2 |
| 4 | 强制只走长路径（仅取证，把 8K 短路径排除在编译外） | `hca_short_attention_pack` | **1** |

变体链：`ptoas` → `ptoas_k128` → `ptoas_k128b` → `ptoas_k128c` → `ptoas_long`。

### 报错本身说明方向是对的

```
proj_a_mm      : left  overflow, requires 1048576 bits (128 KiB) while 524288 (64 KiB)
qr_proj_matmul : right overflow, requires  786432 bits ( 96 KiB) while 524288 (64 KiB)
proj_a_mm_1    : right overflow, requires 1048576 bits (128 KiB) while 524288 (64 KiB)
```

PTOAS **强制**每个 L0 片只占容量的一半，这样 2 个 slot 才装得下 ——
**这正是 Native 的配方**。`ops-transformer` 里
`experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/..._scfa_block_cube.h`：

```c
static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);   // 2 片 = 满 64 KiB 的 L0A
static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);   // 2 片 = 满 128 KiB 的 L0C
static constexpr uint32_t K_L1_SPLIT_SIZE = 256;       // GM→L1 的 K 分块
static constexpr uint32_t K_L0_SPLIT_SIZE = 128;       // L1→L0 的 K 分块（两级不同）
```

换句话说：**PTOAS 在逼着 kernel 用 Native 的分块，而默认 planner 只是默默把
ping-pong 削掉、写一条 perf hint 就算了。**

还有一处很说明问题：`A_K_TILE=128` 在默认 planner 下是**硬错误**
（`Right buffer usage 131072 > 65536`，根因是旧 allocator "只把复用类顺序堆叠、
从不细分已释放区域"），**在 PTOAS 下却是正解**。同一个分块，两个 planner 判断相反。

### 顺带修出来的一条写法

`qr_rms_norm_quant` 原来是

```python
qr_scale_tail = pl.set_validshape(qr_tile_scale_dq, valid_rows, 1)
pl.store(qr_scale_tail, [out_tg, 0], qr_scale_view)
```

PTOAS 报 `pl.set_validshape cannot narrow a tile view (a slice or reshape result)`
——`qr_tile_scale_dq` 是 `pl.reshape` 出来的视图，视图的 valid 范围写在类型里、
没有可改的运行期操作数。按提示改成在切片处收窄又变成 `failed to legalize`。
最后发现 **`pl.store` 的第三个位置参数 `shapes` 可以直接限定写入范围**：

```python
pl.store(qr_tile_scale_dq, [out_tg, 0], qr_scale_view, [valid_rows, 1])
```

一次通过，而且根本不需要给 tile 打 valid 标记。

---

## 三、最后一个：`hca_unified_attention`

### 这段代码在干什么

核心是一段**手写的软流水**（`decode_sparse_attn_hca.py`，长路径）：

```python
kv_l1 = pl.create_tile([QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM],
                       dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)   # 3 个槽的环形缓冲，驻留 L1
for tick in pl.range(work_count + QK_PRE_LAUNCH):        # QK_PRE_LAUNCH = 2
    if tick < work_count:
        l1_row = (tick % QK_TRANSFER_SLOTS) * ATTN_K_TILE
        kv_l1 = pl.gather_row(kv_l1, cmp_work_kv, [l1_row, 0], [cmp_row, 0],
                              [ATTN_K_TILE, HEAD_DIM])          # 第 tick 块 KV → 第 tick%3 个槽
        key_t = pl.tile.slice(pl.tile.transpose_view(kv_l1),
                              [HEAD_DIM, ATTN_K_TILE], [0, l1_row])
        qk = pl.matmul(query, key_t, out_dtype=pl.FP32)
        pl.store(qk, [row, 0], scores)
        pl.system.sync_set(0, ...)                              # 交给 AIV 做 softmax
    if tick >= QK_PRE_LAUNCH:
        pv_work = tick - QK_PRE_LAUNCH                          # ← 错开 2 拍
        pl.system.sync_wait(1, ...)
        probability = pl.load(probs, [pv_row, 0], ..., target_memory=pl.MemorySpace.Mat)
        left = pl.tile.move(probability, target_memory=pl.MemorySpace.Left)
        pv_l1_row = (pv_work % QK_TRANSFER_SLOTS) * ATTN_K_TILE
        right = pl.tile.extract(kv_l1, pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE],
                                target_memory=pl.MemorySpace.Right)   # 读 2 拍前写进去的那个槽
        ... PV 矩阵乘 ...
```

关键是**错拍**：第 t 拍 AIC 做第 t 块的 QK，同时做第 t−2 块的 PV，
中间两拍留给 AIV 算 softmax。所以那块 Mat 缓冲**必须跨 3 拍存活** ——
第 t 拍写进去的数据要到第 t+2 拍才被读走。
§66.1 里记的"PTO attention 166.9 μs 快于 Native `SparseAttnSharedkv` 187.7 μs"，
靠的就是这个机制。

### 四次尝试，四种拒绝

**① 原样切 PTOAS**

```
'pto.tmov' op expects a supported tmov address-space pair for this target
   at decode_sparse_attn_hca.py:1194   ← pl.gather_row 那一行
```

**② 探针：会不会只是"写子区域"不行？**

把 `QK_PRE_LAUNCH` 置 0 → 槽数变 1 → `l1_row` 恒为 0 →
`gather_row` 写的是**整片**而非子区域。**仍然同样报错。**
所以被拒的不是偏移，是 `gather_row` 这个**操作形式**本身。

这里的 DPS = destination-passing style：`gather_row(dst, src, ...)`
往**已经存在的** tile 里写并返回它。对照同一段第 1188 行：

```python
query = pl.load(q_flat, [token * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
```

这**也是 GM→Mat，在 PTOAS 下是通的**。区别只有一个：
`pl.load` **新建**一个 tile，`gather_row` 往**已有** tile 里写。

**③ 换成 `pl.load` 到临时 Mat 片 + `pl.tile.assemble` 塞进环形缓冲**

（即 Mat→Mat 的子区域写）→ 同样的 tmov 报错。

**④ 那就不要持久缓冲：改成移位寄存器**

`QK_PRE_LAUNCH=2` 意味着 PV 只需要 2 拍前的数据，所以用
`pl.range(..., init_values=(kv_p1, kv_p2))` 携带两个片，
每拍 `kv_p1, kv_p2 = pl.yield_(kv_new, kv_p1)` 往下移。

```
Verification failed after 'ConvertToSSA' for properties {SSAForm}
6 errors:  Variable 'kv_p2' used outside its defining scope
           Variable 'kv_seed_inline2022' used outside its defining scope
```

点名的是 iter_arg `kv_p2` 和那个 `pl.create_tile(..., target_memory=Mat)` 的种子。

**为什么这证明"Mat 片不能被携带"**——因为**同一个函数里就有一个能工作的例子**
（`decode_sparse_attn_hca.py:1233`）：

```python
running_m    = pl.load(sink_col, [head0, 0], [H // 2, 1])              # Vec
running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)   # Vec
for vec_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
        work_count + QK_PRE_LAUNCH,
        init_values=(running_m, running_l, running_left, running_right)):
    ...
    maximum, exponent = pl.yield_(raw_m, raw_exp)     # 在 if 分支里 yield，同样的用法
```

同样的 `pl.range(init_values=...)`、同样在 `if` 里使用、同样用 `pl.yield_` 汇合，
**唯一差别是它携带的全是 Vec 片**。所以不是这个机制不支持 tile，
是**不支持 Mat 片**——Mat 是片上暂存，生命期绑在定义作用域上，
不能作为 SSA 值跨迭代传递。

而 `pl.matmul` 的 **B 操作数必须是 Mat**（`AutoTileMatmulL0` 文档原话：
"对于自动 tiling，右（B）操作数必须是 Mat"），KV 正是 B 操作数。
**既不能让它持久存在 Mat 里，又不能不放在 Mat 里。**

**⑤ 最后一次：连错拍也去掉**

`QK_PRE_LAUNCH = 0` → PV 在同一拍消费同一片 → 彻底不需要跨拍存活。
循环体展平，KV 每拍用 `pl.load(..., target_memory=Mat)` 新建一片，
分支用 `pl.yield_` 汇合。→ 第四种错：

```
'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes
   at pypto/python/pypto/language/parser/ast_parser.py:3837
```

位置是编译器内部生成的 move，最可能是 `pl.yield_` 合并两个分支的 Mat 片时
发出的 phi move。

**⑥ 再去掉分支：把 raw 块剥出循环**

`work_count = 1 + cmp_blocks`，raw 块恒为最后一拍，所以可以把它剥出循环——
循环体内就没有分支，也不会产生合并两个 Mat 片的 phi。
→ **仍然是 `non-mat tmov to use matching src/dst shapes`。**

说明 ⑤ 的失败也不是 phi 造成的：这段代码在 PTOAS 下还有别的构造会生成非法 move
（报错位置是编译器内部的 `ast_parser.py:3837`，看不出对应源码里的哪一条；
可疑的是 `pl.tile.move(probability, target_memory=Left)` 与
`pl.tile.extract(kv_new, ..., target_memory=Right)` 这两类跨空间搬运）。

### 为什么不能把环形缓冲放 Vec

Vec 片是可以被携带的（见上面 1233 行的例子）。但容量不够：
`QK_TRANSFER_SLOTS = 3`，每片 `[ATTN_K_TILE=128, HEAD_DIM=512]` BF16 = 128 KB，
三片 384 KB，而 UB 只有 192 KB。即便退到 2 片也要 256 KB。装不下。

---

## 四、结论与可选出路

六次重写、五种不同的编译器拒绝（②与①同错），说的是同一件事：
**PTOAS 对 Mat 的限制远比默认 planner 严格**——
不能对 Mat 做 DPS 写、Mat 片不能跨迭代存活、Mat 片的分支合并也受限。
而 `hca_unified_attention` 重度依赖 Mat：KV 环形缓冲、`query`、`probability`
都驻留 Mat，且 `pl.matmul` 的 B 操作数还必须是 Mat。

**planner 是整个 kernel 编译级的设置**，这一个函数编不过就整体编不过，
所以连"其余任务在 PTOAS 下快多少"这个数字都拿不到，
也就无法判断值不值得为它重写 attention。

两条可选出路，都超出"改算子"的范畴：

1. **重新设计 KV 进 L1 的机制**，不用持久的 Mat 缓冲。
   但那个缓冲正是 attention 的手写软流水（QK 写第 t 槽、PV 读第 t−2 槽），
   它是 PTO attention 比 Native 快约 21 μs 的原因。
   改掉它等于拿这个优势去换 L0 ping-pong，**净收益未知**。
2. **让 PTOAS 支持 `gather_row` 的 GM→Mat 下降**（PyPTO 改动）。
   按 2026-09-29 的约定，PyPTO / Simpler 原则上不做大修改，要做先商量。

非 attention 部分的修改（`variants_planner_20260929/ptoas_k128c`）是有效且可复用的，
将来若 attention 的问题解决，可以直接接上。
