# KV投影直接复用Native NZ权重：设备收益成立，兼容收尾中

基线369ad2c1，不叠加O-A短档候选；17:44已正常auto提交task_20260929_174400_225474124825，生产未改。
本地ops-nn19614968的mat_mul_v3.cpp通过FORMAT_X2选择CubeFormat::NZ，
并将transB=1传入MatmulType。当前Native七档profile的KV MatMulV2也实际使用NZ物理形状。
这里只参考Native的权重布局与转置读法，不认定旧命名MatMulV2二进制与该源码完全相同。

当前PTO/pypto-lib2164563的KV入口是ND [4096,512]；shared prepare_weights在加载阶段将Native
[512,4096] NZ解包、转置成ND私有副本。该成本只发生于加载期，不计入CSA事件，不能当成逐步性能收益。
候选用[512,4096]与BF16_WEIGHT_LAYOUT，Cube b_trans=True；生产接入可借用Native原地址。
两档设备结果已确认原地址复用与核内收益，具体数字及兼容范围见文末；生产尚未接入。

## 改动边界

性能版根/qkv及共享nz_mode/native_adapter共四文件。仅根签名明确为Native几何的wkv进入直接绑定；
精度版仍[4096,512] ND，继续旧准备流程。现有M/N/K分块、完整K遍历、工作编号、任务数量/依赖保持。
先验证新鲜真实权重路径；若值得采用，必须补旧快照ND转置方向迁移、ND/atomic和尾行兼容，不能当前就接生产。

NZ切片需要可证的非负偏移。第一版动态M组除数无法证明；加入max会被早期简化移除，仍失败。
将原有1/2/3个M组显式constexpr化后，常量若先赋给编排局部变量，outline仍会将其捕获为动态入参。
最终常量直接用于核体，保持原分组公式/编号，完整PTOAS/CCE/link/load及两入口解析通过；无工具链修改。
生成2种M尺寸各3种组数，共6个KV核体；L0仍K128双缓冲，无额外TMOV，NZ GM描述符确实进入生成码。
这包含编排分支与常量除数的变化，不能将未来全部收益只归因成NZ读取。

长128K/B16、短8K/B24按同卡反序、5预热20正式事件及独立四窗运行，max-time7200；私有源码和入口已冻结。
已检查目标核均值/工作量与跨度、完整CSA/P95、八类完整状态，以及Native wkv format29/data_ptr。
Native和PTO的KV工作分工分别不同，不能把各自单核均值直接相减声称同工作量差距。
短长若收益分化，再按实际输入在同一个算子里选择，不能拼不同包结果。

[补丁](candidate.patch)、[冻结来源](source.json)、[CPU编译](compile_candidate.json)、
[生成码类型与指令证据](static_evidence.json)、[准备入口](prepare.py)。

## 若设备收益成立，再做的兼容工作

旧快照的weight_layouts仅记录四张根权重，但tensors.wkv已记录ND/layout与source_format。
迁移应核对该显式元数据及已知[4096,512] BF16几何、连续存储，不能只凭shape猜来源；
其他缺失/未知布局仍拒绝。还需验证新性能版NZ快照回放到旧方向精度版，不能只覆盖旧→新。
接入时两版根布局元数据都应包含wkv，矩阵方向由真实根签名决定；精度版数学接口与ND准备行为保持。
两种方向转换继续要求独占只读存储，保留同布局同形状的零转换分支。
复用现有test_csa_replay.py与test_csa_nz_config.py补受影响项即可，无需重跑无关测试。

## 两档结果：保留核内收益，补兼容后接生产

任务在auto设备1完成退出0，八类完整跨版本状态零容差、图重放保护区和16个官方DFX窗口通过。
候选两档均确认wkv为Native format29、逻辑[512,4096]、物理[256,32,16,16]且data_ptr相同。
KV长B16核时23.320→12.498μs（−46.408%），短B24 37.961→19.221（−49.367%），8:2 −47.000%。
四窗范围长20.777–25.063对12.260–13.002，短34.830–40.890对18.102–20.053，均完全分离。
12/8份KV工作量和编号保持，可对比同范围均值/总量，未把Native不同分工直接相减。

完整CSA长969.455→970.119（+0.068%），短919.268→913.069（−0.674%），8:2 −0.080%。
P95长981.240→983.280、短934.060→929.320；max长984.480→993.680、短936.520→939.260μs。
四组均0/20超过P50的105%，候选P95/P50长1.0148、短1.0162；保留长档P95+2.040和两档max增加的事实。
未改Score/Sparse在DFX中仍有波动，不能算成新的KV算术收益或与正式CSA直接相减。
按用户要求保留明确核内收益，旧快照/ND/atomic/尾行兼容尚待补，生产尚未接入；不推算七档或EP16。
[完整结果](RESULTS.md)、[四窗和跨度](kv.json)、[原地址证据](bindings.json)、[阶段决定](decision.json)。

## 全部静态 matmul B 的 ND 缺口

下表是生产369ad2c1的根签名与实际消费路径；八张均为BF16/INT8静态矩阵。
同一个Hadamard供两条路径使用，只计一张。配置mode=2不会替代PTO根声明和适配器的布局处理。

| PTO入参 | 当前PTO逻辑形状/类型 | Native来源与NZ接入方向 |
| --- | --- | --- |
| wkv | [4096,512] BF16 ND | Native NZ [512,4096]；候选已验证原地址，b_trans=True |
| idx_wq_b | [1024,8192] INT8 ND | Native profile为NZ，优先直接借用；矩阵方向不变 |
| weights_proj | [4096,64] BF16 ND | Native profile为NZ [64,4096]，按Native方向接入并配套转置读法 |
| cmp_wkv | [1024,4096] BF16 ND | Native融合Compressor要求ND；PTO初始化准备NZ副本 |
| cmp_wgate | [1024,4096] BF16 ND | 同上，保持门控投影与归约算术 |
| inner_wkv | [256,4096] BF16 ND | Indexer Compressor，同样可给PTO准备NZ |
| inner_wgate | [256,4096] BF16 ND | 同上 |
| hadamard_idx | [128,128] BF16 ND | 固定变换矩阵，初始化准备NZ并供Q/K两条matmul复用 |

上述格式准备均在初始化完成；Native已有NZ则直接借用，禁止每个decode重复转换。
仅改加载不够，根签名、inline签名、切片方向/b_trans、回放来源布局必须一致。
Compressor保留Native ND不意味着PTO只能ND；四张NZ私有副本的原始数据量合计20MiB/CSA层，
实际分配/共享应实测记录，不能为改PTO布局破坏Native prefill/回退路径。
Hadamard虽很小也纳入候选，但不能预先断言每项NZ必然有完整CSA收益。

hc_attn_fn是FP32 matmul B，当前Native转换策略不将FP32转NZ，现有只读格式29桥接范围也不同；
不能为NZ擅自转BF16。Score/Attention中运行时生成的B不是静态权重，不能初始化预打包。
上述审查基于本地生产代码及已有Native profile，不新增占卡测试。
