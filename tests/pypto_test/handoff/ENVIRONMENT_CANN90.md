# 2026-09-21：CANN 9.0 环境恢复记录

本文对应当前16逻辑设备的Ascend910_9392机器。旧机器的CANN9.2、Python3.11和ABI实验
仍保留在[历史环境记录](ENVIRONMENT_AND_BUILD.md)，不作为当前启动配置。
最终权重格式已由用户确认W8A8；48分片目录是cann_recipe量化，只作参考。
用户将通知vllm_ascend W8A8目标权重的下载结果，最终checkpoint尚未指定。

2026-09-22修订：pypto-lib已恢复干净，只作只读参考；CSA实现归属本vLLM-Ascend仓库。
PyPTO的editable缓存识别及subview codegen补丁已撤回并重建安装，保留Simpler版本对齐、
TorchNPU2.10.0.post2兼容及对应测试/文档。下文持久缓存PASS为撤回前历史证据，
当前双目录editable安装可能bypass缓存，不能按历史结果声称当前缓存已复验通过。
详细原因及回退产物见验证日志第34节。

## 启动与源码

从vLLM-Ascend仓库根目录执行`source ../env.sh`。该文件选择工作区`.venv`、CANN9.0、
GCC15.2、官方PTOAS0.61和隔离安装的custom vendor，并禁用Python用户site。
当前环境文件快照见[env_cann90.sh.txt](../results/cann90_20260921/env_cann90.sh.txt)。
共享conda环境、系统PTOAS及系统vendor未被替换。

| 组件 | 实际选择 |
| --- | --- |
| Python | 3.10.19，aarch64，工作区`.venv` |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0.post2 |
| CANN | 9.0.0 |
| PyPTO | `02c0026993b08353e6cee4fdb93e86eddabb8701`，`feat/kernel-mode-integration-test` |
| Simpler | `e914837d540a899dfce0cf48f2adac91e3884930`，`feat/kernel-mode-integration-test` |
| PyPTO runtime子模块 | 与上述Simpler提交一致；本地ABI pin同时更新 |
| pypto-lib | `205255b4770ee84dfa176bcbc7bbef651953c7e1` |
| vLLM | `.cache/csa/vllm-84030bbe`，`84030bbe3d74d99bad477a3d2e37a973ccd8865c` |
| vLLM-Ascend | `e771542f7790d651589c90c75b157790462b0920`，`dsv4-flash-pto`及当前本地修复 |
| PTOAS | 官方0.61 cp310 wheel，独立`.cache/csa/ptoas-0.61`环境 |
| PTO ISA | `03e45c4bda48a6909feb239f0acc45f04176c2a1` |

完整包版本、import来源、源码状态和参考权重哈希见
[environment_restore.json](../results/cann90_20260921/environment_restore.json)。
editable包的`0.1.dev...`版本来自浅克隆元数据，判断源码身份时使用上述SHA。

## 已安装的构建产物

PyPTO/Simpler由本工作区调试分支编译，GCC15.2，构建并行度2。
PyPTO启用TorchNPU和相关测试扩展。框架门禁精确增加`torch_npu2.10.0.post2`，
不泛化为允许所有2.10/2.12版本；对应CPU回归和8个真机退出清理场景已通过。

PTOAS wheel SHA256：
`f808968019a11598b9418bfb25e4fc81c6869e12683b67921f15743095ac8df5`。
此值与PyPTO的`toolchain/versions.env`一致。系统PTOAS包装脚本不满足持久化缓存的launcher
依赖审计语法，因此改用官方wheel；未放宽缓存断言。后续补齐editable多包根依赖盘点，
原缓存失败用例真机复验已通过，详见验证日志第28节。

原生扩展使用CMake3.31.10、系统GCC10.3.1、**Unix Makefiles**，并行度2：

- build：`.cache/csa/native-build`；install：`.cache/csa/native-install`。
- SOC：`ascend910_9392`；实际导入的扩展为cp310/aarch64。
- Ninja在CANN9.0的host-stub对象路径查找失败；源码setup.py也指定Make生成器。
- 未使用旧机器的CANN9.2 ABI实验补丁。

配套custom包使用CMake3.31.10、系统GCC10.3.1、Ninja、SOC`ascend910_93`。
初始构建入口为：

```bash
bash csrc/build.sh --pkg --soc=ascend910_93 \
  --ops='compressor;compressor_metadata;quant_lightning_indexer_v2;quant_lightning_indexer_v2_metadata;sparse_attn_sharedkv;sparse_attn_sharedkv_metadata;rms_norm_dynamic_quant;inplace_partial_rotary_mul;scatter_nd_update_sk' -j2
```

构建时需将本地CMake3.31实际bin目录和`.cache/csa/build-tools/bin`加入PATH，
设置`CC=/usr/bin/gcc`、`CXX=/usr/bin/g++`、`MAX_JOBS=2`、`CMAKE_BUILD_PARALLEL_LEVEL=2`。
安装位置为`.cache/csa/csa-native-ops-install/vendors/custom_transformer`，由`env.sh`加载。
9个执行符号及9个workspace符号均已检查。产物哈希见
[custom_ops_manifest.json](../results/cann90_20260921/custom_ops_manifest.json)，
编译配置见[native_build_config.json](../results/cann90_20260921/native_build_config.json)。
3个metadata算子设备smoke已通过；修复QLI长度刷新后，参考Native B4/B40完整执行输出有限。
这不是目标权重加载或完整Native/PTO数值对照通过的证据。

2026-09-22补充：本仓库scatter快速分支已修复忽略负slot的问题；重建时必须核对生成目录
中的kernel头文件，首次增量构建未自动刷新该文件。安装后8组真机保护区回归通过，
完整B4 Native allocation保护区也恢复通过。最新产物hash见
[scatter修复记录](../results/cann90_20260921/scatter_negative_fix_manifest.json)，
此前`custom_ops_manifest.json`的包hash保留为历史构建记录。

## 兼容修复与约束

- CANN9.0缺失的`OP_LOGE_FOR_INVALID_*`宏有带`#ifndef`的本地兼容实现，
  同时保留Dlog和内部错误报告；避免opdev同名`OP_LOGE`的参数含义冲突。
- Python3.10不支持starred subscript，两个dtype补丁改用兼容的`typing.Literal[tuple]`写法。
- 测试环境选择器同时识别新旧scikit-build-core editable hook，仍要求来源唯一。
- 参考checkpoint的`.scale`及一维channel scale由单层测试loader规范化。
  该修复不代表最终checkpoint整模型加载已验证。
- NumPy2.2.6满足PyPTO要求，但与Triton-Ascend声明的1.26.4冲突。
- torch_npu post2适配本机CANN9.0，但仓库依赖声明post4。
- FastAPI0.136.3满足锁定vLLM要求，但与Ascend声明的`<0.124`冲突。
  HTTP路由构建及TestClient请求已通过，模型引擎及PD尚未验证。
- 上游Triton与Triton-Ascend共用模块路径；此环境最终单独重装了Triton-Ascend。
  当前可导入的backend仅为Ascend，不能仅凭上游distribution metadata判断。

这些依赖冲突仍由`uv pip check`明确报告，没有通过修改metadata隐藏。
安装时的约束和override文件位于`.cache/csa/setup-cann90/`，对应日志已归档。

## 验证与后续入口

所有实际NPU运行都通过`task-submit`，使用其分配的`TASK_DEVICE`。
单卡入口`run_csa_validation.sh`覆盖layout、metadata、native算子smoke与native forward；
`dp_metadata`阶段要求两卡，并将分配设备映射到逻辑0、1。
命令示例见[README](../README.md)。排队较长时使用异步提交，避免客户端超时自动取消尚未运行的任务。

已完成：PyPTO kernel-mode CPU回归611项、Simpler CPU回归295项（另1项跳过）、
Ascend兼容回归7项，以及18个runtime真机场景。剩余1个缓存场景经PTOAS及editable盘点修复后也已通过。
新增cache identity CPU回归163项、DSA metadata CPU回归55项均通过。
六档各100步metadata及真实DP2已按新增每builder QLI长度/余数断言重新验证通过。
参考Native B4/131072、B40/131073完整forward输出全部有限。
完整PTO attention adapter、P2/P3/P4完整CSA及P5均未验收。

持续结果见[验证日志](../DSV4_FLASH_CSA_VALIDATION_LOG.md)和
[日志归档清单](../results/cann90_20260921/log_archive.json)。

## 2026-09-23 完整worker所需ATB环境

TP4×DP4启动暴露此前单层测试未经过的ATB注册：`libatb.so`不在库路径。
本机安装包 `/home/CANN-version/cann-0514/Ascend-cann-nnal_9.0.0_linux-aarch64.run`
存在。普通安装因当前用户与系统CANN所有者不同退出；未改系统目录或文件属主。
使用安装包自带的`--noexec --extract`提取NNAL与其中ATB子包到本仓库`.cache/csa/`，
按其原始 `atb/set_env.sh --cxx_abi=1` 配置用户态动态库。
`../env.sh`已加入此本地环境入口；torch.compiled_with_cxx11_abi()为True。
`torch_npu.op_plugin.atb._atb_ops._register_atb_extensions()` CPU注册检查通过。
ATB自带version.info：9.0.0.B160，commit 8bf8f4054d644f14d3924a7f0b4d0467ef5cfa52。
第三次P任务已通过ATB注册和16rank HCCL分组，进入正式75分片模型加载；
不将此状态记作完整推理通过。
