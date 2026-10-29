# CSA 94个policy合并C++：设备选档原型

2026-10-29。公共工具链`csa-shared-20261024-group-dims`，CANN9.2。
最新可执行产物：`probe/cpp94_v2`。精简结果：`cpp_progress.json`。
本文件记录原型阶段；后续稳定态性能与正式ACLNN接入已经完成单卡验证，详见
[ACLNN_AND_PERFORMANCE.md](ACLNN_AND_PERFORMANCE.md)。MPMD和共享SDK仍未修改。

## 已确认实现

94种tile/数值策略全部保留，以独立命名空间分开编译，最后链接为一个MIX入口
使用的AIC/AIV镜像。每次调用只执行所选分支，不提交94个任务。
原计算代码取自既有wrapper内的`ptoas-generated code`区段，不重新走PyPTO/PTOAS。

48个AIV清零各自11520个INT32 phase信号；仅AIV0根据设备上实际Q/L、T/B容量
及已分配页表宽度计算整批policy。失活请求不贡献Q/L最大值。AIV0通过Scalar写入
与DCache刷回发布ID，所有核复用清零结束的唯一一次MIX syncall，过屏障后选分支。
原分支的启动清零和首个屏障移到公共入口；后续流水依赖的预置event保留。
独立512B policy_state不会被phase清零覆盖；replay每次重新计算。

workspace沿用78号策略的容量上界，按本次T/B分配，不累加94份空间，不改变KV布局。
正常外部ABI仍为55个Tensor；内部133项payload末尾追加policy_state。
非融合分支的group_offset_words扣除共享空间中额外T个尾部元素，保持原逻辑长度。

## 本轮超时根因及修复

v1错误地直接编译`ptoas/dynamic_csa_full.cpp`，漏掉正式wrapper预处理。
原始文件含`set_ffts_base_addr()`，但普通GM Tensor不是硬件FFTS控制地址；
正式后端会删除这些调用，使用Simpler已初始化的地址。v1因此在硬同步处超时。
这是合并脚本错误，不是PyPTO缺陷、镜像大小限制或策略数限制。

隔离验证：保留完整94份镜像但只写回时72核正常返回；单纯policy计算通过；
单个错误FFTS初始化后的syncall即可超时；只去掉错误地址覆盖后，原selector
探针通过，72核均得到policy50。随后v2采用处理后的C++并融合公共启动段。

## 已有验证

- 188/188对象编译和两镜像链接通过，C++ selector的94组代表描述符与原policy一致。
- 128K/B1/S6、128K/B16/S6：eager及图重放输出、更新cache/state与原对应策略逐bit一致。
- B16在输入地址、容量及已捕获图不变时，从真实128K切到8K；Q/L、位置、slot、
  RoPE一起更新。policy76切到26，重放结果与原26号策略逐bit一致。
- 混长+padding：容量B16/T96，8个有效请求Q=[1,2,3,4,5,6,1,2]、24个有效token，
  历史为8K至128K；其余72行集中到L=0的失活请求。eager/replay及保护区通过。
- 上述所有场景只读metadata不变，cache/state保护区完整。

这些是3个实际策略（50/76/26）的代表场景设备验证，不能写成94档全部上卡通过。
未新增Native/整模型/token/DSpark结论，未测稳定态性能。result中的首次墙钟包含
加载和注册，不能当作CSA耗时。

## 重跑与后续

同目录`run_probe.sh`统一加载公共SDK。依次运行`build_cpp.py --output <目录> --jobs 188`、
`cpp_dispatch.py --output <目录>`、`link_cpp.py --output <目录>`；最后一步可与对象编译
重叠，会等对象就绪。`validate_cpp.py --artifact <目录> --case C04 --cross-policy-replay
--output <新结果目录>`直接加载已有二进制，不触发JIT。
CPU构建经task-submit --no-device，NPU验证经task-submit --device auto。

下一阶段接通正式PyTorch/ACLNN入口及图合同，再测稳定态开销。保持94份实现，
遇到具体限制再讨论，不擅自删策略或改tile。早期Python前端合并尝试已放弃；
初始探针过程见验证日志735.135。详细文件与分工见CPP_MERGE_HANDOFF.md。
