# 94档设备选档：稳定态性能与正式ACLNN入口

日期：2026-10-29。CANN9.2.0-beta.2、公共csa-shared-20261024-group-dims。
原型阶段提交89f541752已推送；本文件记录后续性能和ACLNN接入阶段。
正式接入指`backend=aclnn`。默认`backend=python`的JIT入口尚未替换；以下PyPTO
计时通过运行时直接加载统一预编译产物，不应误解为默认Python入口已经切换。

## 稳定态性能

以下是PyPTO/Simpler路径的完整CSA L1首个incore开始至最后一个结束，单位μs。
同场景、同卡、同SDK，双方各预热10次，交替顺序采10个单次graph replay窗口。
输入/cache恢复在窗口外，不剔除慢样本。全部原始L1记录、min/mean/max/median见
[stable_performance.json](stable_performance.json)。以下数值不代表ACLNN核耗时。

| 场景 | 原独立策略中位数 | 94档合并中位数 | 增量μs | 耗时变化 |
|---|---:|---:|---:|---:|
| 128K/B1/S6 | 382.12 | 362.07 | -20.05 | -5.25% |
| 128K/B16/S6 | 779.19 | 782.83 | +3.64 | +0.47% |
| 8K/B16/S6 | 647.95 | 651.50 | +3.55 | +0.55% |

采样后输出与更新状态仍逐bit一致。B1改善的具体来源尚未拆解，不把它全归功于选档；
长短B16新增约3.6μs。本阶段没有做进一步性能调参，也没有扩大到八档全表。
任务：task_20261029_210342_339863030512、task_20261029_210943_371026117598、
task_20261029_210943_371026027025。

## ACLNN实现

- 新op/vendor为CsaSpmdTndDevicePto/csa_spmd_tnd_device_pto，Torch命名空间
  csa_pto_device。保留现有旧62档包，独立wheel名为csa-device-pto-ops。
- 55个业务Tensor，Host仅按描述符计算一份容量上界；tiling key固定0。
  variant参数为沿用调用接口的固定0，不代表Host选择94档中的某一档。
- 设备复用已验收的94份数学实现、单AIV policy计算与公共清零屏障；分文件188路
  CCEC编译再链接到CANN的MIX入口，没有巨型单编译单元，也没有重新生成PyPTO数学。
- CANN9.2构建进程通过私有sitecustomize补入188份对象到最终设备链接；只在构建
  子进程设置PYTHONPATH，不修改CANN安装目录。不要把aclnn_link_hook加入运行环境。
- 正式PreparedDynamicCSA在backend=aclnn且新包可见时选择设备包；未装新包仍支持
  原62档包。设备包不调用Host select_policy，不按Host Q/L生成不同的图缓存键。
  描述符/地址合同仍保留，MPMD不改。
- 绑定对象不分配外部工作区/辅助TopK；workspace由ACLNN每次调用管理。
  当前新包只提供55 Tensor业务ABI，显式拒绝57 Tensor诊断TopK，避免返回未写输出。

## 已有验证和边界

[aclnn_validation.json](aclnn_validation.json)记录直接ACLNN及正式后端的对照结果。

- 直接ACLNN：128K/B1、128K/B16、混长+集中padding，eager/replay输出与更新cache/state
  均与原独立PTO策略逐bit一致，只读metadata及保护区通过。
- B16同址同图保持T/B容量，真实128K切8K并同步修改位置、slot、RoPE及Q/L，
  设备policy76切26，重放仍与原对应策略逐bit一致。
- 正式PreparedDynamicCSA→backend=aclnn→新包链路：B16的eager/replay/跨档全部通过，
  Host policy key为{}，prepared.allocations为空。
- 35项CPU合同测试通过，包含旧62档合同、ABI漂移拒绝、新包禁止Host选档及诊断拒绝。
- wheel已安装到独立target，正式loader/55参数ABI检查通过且未初始化NPU。

未验证SuperKernel、npugraph_ex、整模型、ACLNN稳定态性能及全部94策略的设备覆盖。
本阶段不改变既有Native/PTO复杂问题精度差异结论。功能对照进程中Simpler仅用于
原策略参考；候选通过真正的ACLNN/CANN执行，不是改名后仍派发Simpler。

## 使用已构建wheel

本机产物：
`tests/pypto_test/csa_device_policy_20261029/probe/aclnn_package_v1/dist/csa_device_pto_ops-0.2.0-cp310-cp310-linux_aarch64.whl`。
Python3.10/aarch64，要求与包清单一致的Torch/TorchNPU；CANN使用9.2。
在目标环境安装wheel后，使用既有开关：

```bash
export PTO_CSA_VARIANT=spmd_tnd
export VLLM_ASCEND_PTO_CSA_BACKEND=aclnn
export VLLM_ASCEND_ENABLE_NZ=2
```

本轮仅安装到独立target，没有把新包全局装入三个session共用环境。
本机可把`probe/aclnn_installed_v1`加到PYTHONPATH进行选择；或在自己的环境
执行`python -m pip install --no-deps <上述wheel>`。后端加载时激活独立OPP vendor。

正式路径单卡复跑示例（在仓库根目录，output须为新目录）：

```bash
task-submit --device auto 'bash tests/pypto_test/csa_device_policy_20261029/run_probe.sh validate_cpp.py --artifact tests/pypto_test/csa_device_policy_20261029/probe/cpp94_v2 --formal-aclnn-package tests/pypto_test/csa_device_policy_20261029/probe/aclnn_installed_v1 --case C04 --cross-policy-replay --output tests/pypto_test/csa_device_policy_20261029/probe/formal_aclnn_new'
```

## 构建入口

所有构建/测试先source工作区公共env和共享toolchain/env.sh，使用CANN9.2。
先按DESIGN.md得到cpp94_v2；prepare_aclnn.py接收该产物和旧ACLNN工程的Host/接口
生成模板。参考模板为mihudaner/vllm-ascend的caed2c32896ddefdbcfab180b253d777bfab11e0，
本机位于工作区.worktrees/csa-aclnn-reference-caed2c32/examples/csa_spmd_tnd_aclnn。
模板只用于Host/接口，不使用旧62档计算代码。

1. prepare_aclnn.py生成probe/aclnn_v1。
2. build_aclnn_objects.py --output <绝对aclnn_v1路径> --jobs 188；run_aclnn_build.sh
   构建Host。二者可并行，均通过task-submit --no-device。
3. run_aclnn_build.sh kernel等待对象齐全，构建MIX镜像、JSON和独立.run包。
4. build_aclnn_torch.py构建Torch库到probe/aclnn_torch_v1；package_aclnn.py
   --output <绝对probe/aclnn_package_v1路径>准备wheel内容。
5. build_aclnn_wheel.sh打包；install_aclnn_wheel.sh安装到新的独立target并验证loader。

两次构建脚本适配错误（namespace与函数同名、policy_state形状变量命名）已修正；
不计作SDK缺陷。ACLNN的188对象成功编译228.31秒，未改变共享SDK。
