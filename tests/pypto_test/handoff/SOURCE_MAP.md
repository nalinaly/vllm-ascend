# 源码导航与未完成的适配设计

此文件区分已确认的源码事实和未实施的候选方案。以[source_lock.json](source_lock.json)的提交为准。

## 1. 原生调用链

从仓库根目录检查以下位置；行号会随后续修改变化，函数名是定位依据。

| 文件 | 关键入口/用途 |
| --- | --- |
| `vllm_ascend/models/deepseek_v4/model.py` | `DeepseekV4Attention.forward`、外层HC/norm/decoder边界 |
| `vllm_ascend/ops/dsa.py` | DSA custom op、ForwardContext metadata查找 |
| `vllm_ascend/attention/dsa_v1.py` | `AscendDSAMetadataBuilder`、`_forward_attention`、QKV多流、SAS/indexer调用 |
| `vllm_ascend/models/deepseek_v4/compressor.py` | state/cache两组metadata、CompressorMetadata等待、Compressor实参 |
| `vllm_ascend/models/deepseek_v4/indexer.py` | indexer cache spec、QLIv2、quant/scatter |
| `vllm_ascend/device/device_op.py` | A3 indexer量化、INT8 K/FP16 scale写回与scatter |
| `vllm_ascend/ops/rope_dsv4.py` | default/c4 RoPE缓存与位置选择 |
| `vllm_ascend/models/layer/attention/layer.py` | `DSV4_BLOCK_SIZES` |
| `vllm_ascend/core/kv_cache_interface.py` | `get_storage_block_size`：逻辑token数与物理行数 |
| `vllm_ascend/worker/block_table.py` | CPU allocator mirror、commit、slot mapping、swap/move/clear/add |
| `vllm_ascend/worker/model_runner_v1.py` | `_adjust_kv_layout`、`_sync_metadata_across_dp`、graph初始化和输入准备 |
| `vllm_ascend/worker/device_metadata.py` | `DeviceMetadataExecutor`、stage/group、external events和reuse |
| `vllm_ascend/spec_decode/utils.py` | `update_num_computed_tokens_for_batch_change`、`correct_optimistic_seq_lens_cpu` |
| `vllm_ascend/utils.py` | `should_skip_allreduce_across_dp_group`、Ascend customop注册 |
| `vllm_ascend/ascend_config.py` | ND配置、通信/graph选项 |
| `csrc/torch_binding.cpp` | RMS、rotary、scatter、Compressor的C++调用参数 |
| `csrc/attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h` | 当前QLIv2 custom ABI，含stride和candidate参数 |
| `csrc/attention/compressor/op_host/arch32/compressor_tiling.cpp` | state首维stride允许大于连续值 |

固定vLLM checkout中还需要：

- `vllm/v1/cudagraph_dispatcher.py`：真实capture key和dispatch；验6的capacity必须是原生选择结果。
- `vllm/config/compilation.py`：`resolve_cudagraph_mode_and_sizes`、spec decode大小调整。
- `vllm/distributed/parallel_state.py`：TP1/DP2初始化时真实world rank和world size的展开。
- `vllm/config/parallel.py`：DP rendezvous port列表；两个子进程必须拿到同一个端口，不能各自随机生成。

## 2. PyPTO参考及其不一致点

pypto-lib参考目录：`models/deepseek_v4_flash_dspark/`。

| 文件 | 作用 |
| --- | --- |
| `decode_csa.py` | `_decode_csa_tp1`及其inline/test包装；整个调用编排 |
| `config.py` | 模型尺寸、decode S/B、压缩state和cache容量 |
| `decode_compressor_ratio4.py` | 主compressor投影、pool、state/KV写回 |
| `decode_indexer_compressor.py` | indexer compressor、Hadamard、INT8 K/FP32 scale写回 |
| `decode_indexer.py` | Q/weights、score/top-k、indexer cache分页访问 |

QKV、稀疏attention、O-proj的具体文件名从 `decode_csa.py` 的imports定位，避免根据旧版本猜名称。
参考入口有HC前后处理、51个参数，不能直接替换原生三个实参的forward。

已确认的差异：

- 参考模块在子模块import之前解析 `sys.argv --tp`，默认TP2，并修改 `config.TP`。
  要做TP1 factory或受控局部模块加载；不要在服务进程全局乱改argv。
- 当前参考fixture每请求S8，target验6。`S//4`的紧凑压缩循环和固定容量不是简单改常量就能证明正确。
- 参考state按小ring访问，原生state通过绝对位置页表和带padding物理页定位。
- 参考indexer scale是FP32分离Tensor；原生是每页INT8 K后面的FP16 scale，两个view共享存储。
- 原生post-load `wo_a`为 `[8,4096,1024]`，参考PTO为 `[8,1024,4096]`；
  原生 `wo_b`为 `[8192,4096]`，参考为 `[4096,8192]`。准备期处理，不能每步重新转置权重。
- 本机Torch interop不允许非连续入口或部分重叠可写参数。普通 `@pl.jit` 入口与
  `@pl.function` 的TensorView能力也不能混为一谈。

TP1入口不使用TP多rank的gather window/通信owner参数。
`num_tokens_per_owner`等TP路径参数的问题，应与本次TP1实际签名分开检查。

## 3. 已可复用的实现

- `dsv4_csa_metadata_kernel.py::metadata_probe`：直接读取NPU边界、长度、页表、slot和layout描述。
  每个SPMD worker拥有完整输出cache line；cache sentinel写入用单task提交，避免4字节写回竞争。
- 同文件 `expand_compressed_slots`：从真实device边界/长度/位置展开紧凑C4 slot，保留无效-1。
- `dsv4_csa_shared_storage_kernel.py`：一个INT8连续载体包含K/FP16 scale，核内reinterpret子区域。
- `dsv4_csa_native_layout.py::allocate_native_cache`：实际调用原生runner布局函数，不模拟连续cache。
- `dsv4_csa_contract.py`：host已知shape/dtype/容量检查；测试preflight的CPU allocator mirror检查。
  后者不是生产热路径的完整页表扫描，也不能保证检测任意device内存损坏。
- `dsv4_csa_all_groups.py`：五组builder、native metadata executor、external events、graph和独立公式。
- `dsv4_csa_dp_metadata.py`：两个真实rank，原生DP同步及dispatcher，探针在共同容量上运行。

## 4. 候选适配方案（尚未实施，不能当成已有代码）

首选目标是核内直接消费原生布局；为分阶段定位，也考虑过以下过渡方案。
这些方案都必须通过完整数值与状态验证，不能仅靠静态推导宣称等价。

### 4.1 Query6到参考S8

可以先把每请求6行在内部scratch扩为8行，保留真实device query边界作为外部接口。
额外两行hidden填零，额外slot为-1，输出只取真实6行；扩展位置只用于内部计算。
需要逐项证明额外行不参与有效compression、top-k、attention或state写回。
代价约增加三分之一token计算量，所以即便通过也只是正确性中间方案，不是性能结论。

更直接的方案是修改参考子核，使请求划分和压缩输出数量完全由device边界决定。
两种方案都应保留异长起始位置和query6的closing token测试。

### 4.2 State小窗口

若复用参考16行ring，可仅在device gather每请求当前需要的小窗口进入scratch，
调用参考计算后只commit实际新token更新的state行到原生页。
不能把整个128K KV复制成连续cache，也不能漏掉拒绝token影响和state跨步覆盖规则。
主state/indexer state分别使用各自原生表和byte stride。

### 4.3 Indexer共享页

可把 `[pages,4160]` INT8底层页作为唯一InOut，K部分为前4096字节，scale为后64字节FP16。
score/top-k核读取K时按真实页stride取值；压缩写回同时维护K和scale。

若过渡期沿用FP32 scale scratch，需要device从原生FP16读入，新scale必须先按原生规则舍入成FP16，
再反馈给打分与原生cache，避免PTO实际用更高精度scale而原生用半精度。
它引入额外读写，只是候选设计，未实现、未计时、未验收。

## 5. Qwen参考与同事HTML

Qwen分支 `qwen3-14b-pto` 的 `tests/pypto_test/README.md`、profile脚本和对应PTO attention接法可参考。
其历史结果确实包含完整Qwen模型Native/PTO相同token输出 `[12095,13,3555,374]` 和3次graph replay。
原始对比摘要已归档为 [Qwen comparison](snapshots/qwen_20260920_comparison.md.txt)，
其机器可读汇总也在同一snapshots目录；未把大型Qwen profiler trace作为DSV4验证数据搬运。
它只证明Qwen那套环境及支持条件，不能为DSV4的S6、共享cache、W8A8或DP16背书。

同事HTML原件保存在上一级目录，审阅见 `CSA_LIB_ASK_REVIEW.md`。
缺少其配套adapter源码与原始精度数据，不能把它报告的转换耗时当作接入成本下界。
用户只让参考一次，后续应回到本文件第1～4节对应的实现与验证。
