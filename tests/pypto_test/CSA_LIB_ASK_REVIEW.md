# csa-lib-ask.html 源码核对与审阅

- 日期：2026-09-21。
- 对象：[csa-lib-ask.html](csa-lib-ask.html)。保留原文件，不修改同事的结论或数字。
- 核对基线：vLLM-Ascend `34bb51f93724c565362f5108f5226303e1b56cad`、
  vLLM `84030bbe3d74d99bad477a3d2e37a973ccd8865c`、pypto-lib `205255b4770ee84dfa176bcbc7bbef651953c7e1`，
  PyPTO 使用本机 `pto_eager/pypto` 的实际源码。
- 审阅方法：阅读全文，对照当前源码和已经保存的本机探针结果；没有重跑文中同事的实验。
- 原 HTML 没有给出完整版本、adapter 源码或精度报告。文中引用的 `csa-attn-perf.py/json/html`、
  `csa-lib-ask-render.py` 在所给目录中不存在；在本机已搜索的工作区也未找到同名配套文件。
  找到的其他旧 CSA trace 不能擅自认定为本报告的数据源。因此本文不确认或否认它报告的实测耗时。

## 1. 判断

有参考价值：它指出的非连续 cache、共享 allocation、FP16 scale 与 FP32 入参不一致，
以及过多转换算子的风险，确实对应当前接入需要解决的问题。

但它把特定转换实现的代价，推成了不可消除的接口成本；又把启动/重放成功，推成了功能正确。
这些结论缺少必要证据。可以作为待验证问题清单，不能直接作为性能下界、正确性验收或最终接口设计依据。

## 2. 关键问题

### 2.1 “转换层无解，至少还剩 1 ms”没有成立的依据

HTML 第78～82、184～210行声称：必须翻译地址，因此至少剩下1～1.5 ms，转换层无法解决。
但第190～192行同时承认未测量融合能减少多少启动，再人为给出1000 us下限。这不是下界证明。

必须计算地址，不等于必须为每个 `gather/clamp/div/where` 启动独立 Torch 算子。
至少存在两种需要实测的实现：

1. 在 adapter 内用一个或少量 device kernel 融合 metadata 转换，输出预分配的 metadata buffer。
2. 让 CSA 内部读取原生页表和长度，在消费数据时完成地址计算。

第一种仍属于接入层，未必需要修改所有 CSA 数值子核；第二种则确实需要改库。
两者都不会令地址计算本身消失，但都可能减少启动和中间 tensor。没有实测，不能承诺耗时为零，
也不能断言至少1 ms。

现有 [metadata probe](dsv4_csa_metadata_kernel.py) 已在 device 内完成边界扫描、页表读取和slot换算，
见 [device 0结果](results/local_20260921/transport_device0.json)。这只证明实现方向可行，
不等于已测量同事转换层的替代性能或完成全部真实metadata对接。

### 2.2 “图模式要求翻译全部在图内”是错误的限制

HTML 第130～133、208～210行把“捕获段Python在replay时不执行”扩大为“不能在图外准备metadata”。

实际约束是：图消费输入时，所绑定的地址、容量、内容和流依赖必须正确。
允许在replay前更新同一地址的device buffer；也允许图外device kernel产生输入，再通过event保证顺序。
已由host掌握的分配器页号、shape和stride，也可以在host更新后按既有流程拷入buffer。

当前原生实现本身就有这些路径：

- [BlockTable.commit_block_table](../../vllm_ascend/worker/block_table.py) 将host维护的页表拷到device。
- [model_runner_v1](../../vllm_ascend/worker/model_runner_v1.py) 在输入准备阶段更新请求边界、位置和实际计算长度。
- [DeviceMetadataExecutor](../../vllm_ascend/worker/device_metadata.py) 在独立stream提交任务，用event约束消费者及buffer复用。

但这不意味着应把真实device长度每步拉回host。异步spec decode中CPU长度可能是乐观值，
必须保留原生device纠正链；见 [update_num_computed_tokens_for_batch_change](../../vllm_ascend/spec_decode/utils.py)。
“host能否取得正确值”与“能否在图外更新输入”是两个不同问题。

### 2.3 “vLLM页128行，PTO页32行”不是当前目标的通用事实

HTML 第118～123、216～229行使用128行、16640 bytes和130行重解释。
它们可能是同事某次 `block_size=128` 实验的真实布局，但不是vLLM固定约定。

当前 [DSV4_BLOCK_SIZES](../../vllm_ascend/models/layer/attention/layer.py) 的A3配置为：

| cache_config.block_size | SWA物理行/页 | C4压缩KV物理行/页 | C4调度逻辑token/页 | indexer K+scale未额外padding的bytes/页 | C4 state物理行/页 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32（本计划） | 32 | 32 | 128 | 4160 | 2 |
| 128 | 128 | 128 | 512 | 16640 | 8 |

C4的128个原始token与32条压缩KV，可能描述同一页。
必须区分调度逻辑token数、物理压缩行数和page byte stride，不能只看 `spec.block_size`。
源码：[IndexerCache.get_kv_cache_spec](../../vllm_ascend/models/deepseek_v4/indexer.py)、
[get_storage_block_size](../../vllm_ascend/core/kv_cache_interface.py)。

如果确实要把128条物理行的页拆为4个32行子页，连续且无padding时，可用
`subpage_id = 4 * native_page_id + subpage_offset`；也可在消费点算地址。
若要实际生成新页表，仍有计算/写入成本；动态页号变化后也须更新。
“整数倍”只说明某些布局可以不复制cache，并不自动证明所有表和slot转换为零开销。

### 2.4 “改成130行就能直接传原buffer”没有覆盖共享写入约束

HTML 第227～231行的算术成立：128行INT8 key、每行128维占16384 bytes，
128个FP16 scale占256 bytes，总计16640 bytes，恰好130×128 bytes。

但一个连续INT8载体只解决入口的连续性，仍必须正确表达：

- 每页的有效key行数128和物理byte stride16640是不同量。
- scale位于每页的偏移16384，dtype为FP16，需要按页跨距访问。
- key与scale属于同一allocation，都是可变状态，必须保持原位写回。
- 若将全页INT8载体和覆盖各页scale的载体作为两个可写参数，字节区间会重叠。

当前 [PyPTO interop](https://github.com/hw-native-sys/pypto/blob/6b49cfd58de65f8a325339a30d6fa45e1973c237/python/pypto/torch/interop.py) 既要求contiguous，
也拒绝部分重叠的可写参数。仅改shape与scale dtype不足以覆盖这些约束。
不能把补齐的scale区域当作有效key行，也不能把整个padding区清零而破坏相邻状态。

本计划的方向是：每个共享allocation只传一个连续存储参数，另带不可变布局描述，
核内访问key/scale子区域。现有 [shared-storage探针](dsv4_csa_shared_storage_kernel.py)
已覆盖非零offset、FP16 scale和页padding，图A→B→A与保护区逐字节对照通过。
这仍是小核结果，完整CSA尚未验收。

### 2.5 “无故障190步，功能已通”超出了所给证据

HTML 第136～140行列出的证据能够支持：入口被调用、程序可运行、capture/replay时有PTO调度活动。
`simpler_aicpu_kernel_exec` 出现在trace，不能单独证明CSA输出、TopK或cache写回正确。

尚缺少以下结果：

- 同一输入/权重/状态下，Native与PTO的attention输出误差。
- 六份可变状态的更新及未触碰区域对照。
- 非连续页表、负slot、padding、请求重排和页释放/复用。
- 同地址改变length/block table后，graph消费的是新值。
- DSpark部分接受后，下一轮历史长度、位置、compressor历史和新投影叠加是否正确。

文中第266～267行的实验为prompt1024、BS4、3层、graph size `[1,2,4]`，
没有提供每请求query=6、128K历史或接受步长3.8的证据。
仅凭这些信息不能确定它如何处理speculative token与padding，更不能替代本计划P1～P4。
文中也没有给出足以复核全流程ND与量化一致性的配置清单。

### 2.6 性能拆分值得参考，优化后预测不能作为验收数字

按AICPU派发区间与AICore执行区间的并集计时，避免重叠重复相加，这个方向合理。
但缺少原CSV、归因脚本和完整stream时间线，目前无法复核28.4、20.8、2.07、0.976 ms这些数字。

删除某些trace行不能保证端到端等量减少：新实现仍有访存和调度，stream可能重叠，
移除大搬运还可能改变其他算子的带宽争用与等待。HTML第279～282行自己也承认这一点。

因此7.2、5.6 ms及1.17倍只能保留为假设下的估算；
“剩下差距只是算子本身”也未经新的同条件profile证明。
capture/replay内的183个设备任务不能直接称为183次Python/host launch开销。

## 3. 需要保留的正确提醒

### 3.1 动态轴缺少跨参数一致性检查，是当前需要显式防范的GAP

这点不能因为文中其他推断有问题就否认。

- 当前PyPTO [interop](https://github.com/hw-native-sys/pypto/blob/6b49cfd58de65f8a325339a30d6fa45e1973c237/python/pypto/torch/interop.py) 和
  [registration](https://github.com/hw-native-sys/pypto/blob/6b49cfd58de65f8a325339a30d6fa45e1973c237/python/pypto/torch/registration.py) 检查dtype、rank、静态维度等，
  动态维度在参数描述中为 `-1`，没有据此自动检查跨参数同名动态轴相等。
- [GenerateDynamicDimDefs](https://github.com/hw-native-sys/pypto/blob/6b49cfd58de65f8a325339a30d6fa45e1973c237/src/codegen/orchestration/orchestration_codegen.cpp)
  按symbol去重，从第一个声明参数取extent，没有为后续同symbol参数自动生成相等断言。

准确说法应是“当前这条接口缺少动态轴之间的关系校验”，而非所有参数都没有校验。
接入层可以检查host可见的shape、stride、dtype、offset、容量，正常情况下不需要device同步。
另外还要校验请求数、query边界和padding语义；shape相等不能证明device内容正确。

HTML第242～243行“签名不匹配会报错，不会静默算错”过于宽泛：
形状和dtype相同的两个cache group传反、逻辑/物理页单位混淆，都可能通过签名检查。

### 3.2 上游提供可维护的原生分页入口，有长期价值

内部寻址确实需要了解compressor的历史/当前token叠加、量化和状态生命周期。
但“不熟悉语义，需要库作者协作”与“接入层在技术上无解”不能互相替代。

现有 [_decode_csa_tp1](https://github.com/hw-native-sys/pypto-lib/blob/205255b4770ee84dfa176bcbc7bbef651953c7e1/models/deepseek_v4_flash_dspark/decode_csa.py) 有51个参数并包含HC前后处理，
本计划需要attention-only入口。文中46个参数属于其自行拆出的入口，不能当作当前上游原签名。

上游本来已有两份state block table和slot参数，问题是其环形列号及物理stride契约不等同于原生。
16行也来自当前 `8个历史行 + DECODE_SEQ=8` 的配置推导，
不是所有CSA或所有DSpark设置的固定语义；见 [config.py](https://github.com/hw-native-sys/pypto-lib/blob/205255b4770ee84dfa176bcbc7bbef651953c7e1/models/deepseek_v4_flash_dspark/config.py)。
当前 [compressor](https://github.com/hw-native-sys/pypto-lib/blob/205255b4770ee84dfa176bcbc7bbef651953c7e1/models/deepseek_v4_flash_dspark/decode_compressor_ratio4.py)
还使用 `s_dim=T//B` 分配token。出5验6、非均匀query、dummy request和padding必须单独处理。

## 4. 对本次验证的实际参考价值

保留三项工程方向：避免每步全cache materialize；融合必要的device metadata变换；
最终按原生分页/共享存储直接消费可变状态。

向库侧提出的接口需求应描述能力，避免固化同事单次实验的130行或16行常量：

1. attention-only边界；外层HC/RMS不进入替换。
2. 接收device query边界、真实length、position及按group明确绑定的页表/slot。
3. 分离逻辑页大小、物理行数、byte stride、子区域offset、scale dtype。
4. 保持共享allocation的原位写回、负slot/padding语义及graph稳定地址。
5. 约束或支持query=6、非均匀请求、接受后续步与动态shape关系，并提供拒绝路径。
6. 先通过输出和状态对照，再用相同ND/量化/128K/BS条件重新profile；不预设1 ms下界或5.6 ms终值。

本次审阅不改变验证计划的完成标准，不把同事的启动成功记入本机P1～P4的PASS。
