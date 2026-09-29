# KV投影直接复用Native NZ权重：CPU候选

基线369ad2c1，不叠加O-A短档候选；17:44已正常auto提交task_20260929_174400_225474124825，生产未改。
本地ops-nn19614968的mat_mul_v3.cpp通过FORMAT_X2选择CubeFormat::NZ，
并将transB=1传入MatmulType。当前Native七档profile的KV MatMulV2也实际使用NZ物理形状。
这里只参考Native的权重布局与转置读法，不认定旧命名MatMulV2二进制与该源码完全相同。

当前PTO/pypto-lib2164563的KV入口是ND [4096,512]；shared prepare_weights在加载阶段将Native
[512,4096] NZ解包、转置成ND私有副本。该成本只发生于加载期，不计入CSA事件，不能当成逐步性能收益。
候选用[512,4096]与BF16_WEIGHT_LAYOUT，Cube b_trans=True；生产接入可借用Native原地址。
是否确实复用以及核时收益尚待真实权重设备验证，不能从声明直接断言。

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
应检查目标核均值/工作量与跨度、完整CSA/P95、八类完整状态，以及Native wkv format29/data_ptr。
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
