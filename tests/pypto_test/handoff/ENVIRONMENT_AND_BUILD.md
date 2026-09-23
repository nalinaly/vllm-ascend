# 本机环境、构建和迁移注意事项

这是历史复现记录，不是要求16卡机器照抄本机不完全兼容的软件组合。
优先使用新机器配套的模型环境，再记录差异并重验必要接口。

## 1. 最后实际使用的环境

| 项目 | 本机值 |
| --- | --- |
| 模型Python | `/mnt/workspace/inductor/vllm-ascend/.venv/bin/python` |
| Python | 3.11.4，aarch64 |
| PyTorch / torch_npu | 2.12.0+cpu / 2.12.0+git5462a1b |
| triton-ascend | distribution 3.2.2；包内module版本号可与distribution不同 |
| transformers | 5.14.1 |
| CANN | `/home/developer/Ascend/cann-9.2.0` |
| NPU | 2×Ascend910_9362，逻辑0/1，对应本机物理14/15，各64GiB |
| PTO env | `/mnt/workspace/inductor/pto_eager/env.sh` |
| PyPTO | `6b49cfd58de65f8a325339a30d6fa45e1973c237`，含下述两处既有本地修改 |
| Simpler | `6354e3f075aee220eaa25686fce18a2a0f3b73ec` |
| PTOAS | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81`；本机v0.57/LLVM21构建 |
| PTO ISA | `f51c92f610827daad0ddfb383072e03d514b4ae9` |
| PTO platform/runtime | `a2a3` / `tensormap_and_ringbuffer` |
| 固定vLLM源码 | 仓库内 `.cache/csa/vllm-84030bbe`，84030bbe提交 |
| 原始C++扩展 | `.cache/csa/native-install/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so` |
| 五算子custom bundle | `.cache/csa/csa-native-ops-install/vendors/csa_validation_transformer` |

最终实际包版本和import路径见 [environment_final.json](environment_final.json)，
各仓库remote/SHA见 [source_lock.json](source_lock.json)。
较早结果目录中的manifest和packages文件保留历史原样；最后状态不再由那些文件单独决定。

现有模型环境的distribution metadata仍可能报告旧vLLM/Ascend版本或CANN的PyPTO0.2.1，
而测试已通过源码选择器从上述固定checkout实际import。两类值均保留在快照中，
不要仅凭 `pip show` 认定执行的源码版本；先检查 `module.__file__` 和该目录SHA。

### 1.1 PyPTO原有本地修改

[pypto-existing-local.patch](patches/pypto-existing-local.patch) 记录任务开始前已存在、此次未改动的两处差异：

- `_kernel_abi.py`中的Simpler revision改为本机实际的6354e3f。
- `torch/shutdown.py`的framework版本门禁额外允许torch_npu2.12.0。

它们是本机通过条件的一部分；不能只checkout PyPTO SHA便假定完全复现。
其中放宽版本门禁并不证明整个组合得到上游正式支持。新机器若已有匹配的新版本，
应重新核对而不是盲目应用旧patch。

### 1.2 已经走过的环境弯路

- 最初找不到vLLM，是看错解释器。Qwen原有模型env可用，后来复用，未重装其torch。
- 用户site另有PyPTO/Simpler editable hook；`dsv4_csa_env.activate()`在当前测试进程修正选择。
  vLLM会开子进程检查模型类，因此仅改sys.path不够，还传递了相同源码PYTHONPATH。
- aarch64 sklearn曾报static TLS不足，启动时预加载它自带的libgomp；未改业务逻辑绕过。
- pto_eager解释器最初装过triton3.2.0，触发SIMT warp stack常量不兼容；原模型env原本已有3.2.2。
  不要再把这个已处理的导入问题当作metadata或CSA计算错误。
- HTTP依赖组合未做完整服务验收；单层import通过不能代表API server已通过。

## 2. 被归档的文件与未搬运的产物

仓库包含：验证Python、全部JSON结果、关键原始日志、计划/记录、构建patch、源码/产物哈希和脚本快照。

大体积且依赖机器的 `.venv`、CANN、编译器、模型权重、`.so`、`.run` 安装包和CMake缓存没有作为Git文件搬运。
原始二进制哈希记录在 [local_binary_manifest.json](local_binary_manifest.json)。
它们应在16卡环境使用配套产物或从记录的源码重建，不能靠Git clone取得。

- [log_archive.json](log_archive.json)：原 `.cache/csa` 日志到 `results/local_20260921/logs/*.txt` 的映射和SHA。
- [snapshots/pto_eager_env.sh.txt](snapshots/pto_eager_env.sh.txt)：本机完整工具链设置；绝对路径需要迁移。
- [snapshots/source_scripts.json](snapshots/source_scripts.json)：原test_script来源和哈希。
  快照以 `.txt` 保存，仅供检查；源机器网络地址已替换为 `SOURCE_HOST_ADDRESS`。
- [snapshots/reference_checkpoint_config.json](snapshots/reference_checkpoint_config.json)：参考权重配置，无权重payload。

原脚本的容器镜像不是当前基线版本锁，不能依据镜像名字认定其custom算子匹配34bb51f9。
新机器应记录实际加载产物，而不是仅复用一个历史镜像标签。

## 3. 环境辅助脚本的路径合同

`dsv4_csa_env.py`是独立测试的工作区选择器，当前要求：

1. 已设置PTO工具链的 `PTO_EAGER_ROOT`，其下有 `pypto/`、`simpler/`。
2. 该root下 `.venv/lib/pythonX.Y/site-packages/` 存在 `_pypto_editable.py` 与 `_simpler_editable.py`。
3. 仓库内 `.cache/csa/vllm-84030bbe` 如存在则优先使用；不存在时使用当前解释器已安装vLLM。
4. `load_native_extension()`默认要求 `.cache/csa/native-install` 中唯一的本机构建扩展。

这些目录不在Git包里。新环境可以按同一合同准备，也可以修改测试选择器以使用实际安装路径，
但必须保留import来源与版本断言，并把变化记入新manifest。
不要把工作区选择器复制到最终生产forward；生产接入使用环境中已安装的PyPTO和原生插件。

固定vLLM源码可以这样取得，目录是测试脚本当前约定：

```bash
git clone https://github.com/vllm-project/vllm.git .cache/csa/vllm-84030bbe
git -C .cache/csa/vllm-84030bbe checkout 84030bbe3d74d99bad477a3d2e37a973ccd8865c
```

其余库按source_lock中的remote和commit获取。编译安装按各库该提交的说明执行；
不要因为这个文档列了版本就对新环境已有软件执行一整套无差别升级。

## 4. 本机原生C++扩展的构建记录

仓库CMake要求Torch2.10，本机已有Torch2.12。仅在 `.cache/csa/native-source/CMakeLists.txt` 副本中
把版本fatal改为warning，symlink回原 `csrc`、`cmake`，没有修改原始生产源码。
差异为 [native-cmake-version-experiment.patch](patches/native-cmake-version-experiment.patch)。

完整历史配置argv见 [native-configure-command.json](snapshots/native-configure-command.json)。
关键条件：Python3.11 headers、torch_npu路径、pybind11 CMake路径、CANN9.2、SOC `ascend910_9362`。
随后执行：

```bash
cmake --build .cache/csa/native-build -j8
cmake --install .cache/csa/native-build
```

这套构建可以加载和执行已验证的probe所需算子，不代表整个Torch2.12组合已获正式支持。
新环境有配套扩展时优先用配套扩展，不需要复制版本门禁实验。

## 5. 本机custom算子构建与过期JSON问题

最初从当前仓库成功构建CompressorMetadata，安装在任务目录，没有覆盖全局vendor。
之后扩展为以下五算子包并全部构建、打包、安装成功：

```text
compressor_metadata
rms_norm_dynamic_quant
inplace_partial_rotary_mul
scatter_nd_update_sk
quant_lightning_indexer_v2
```

首次初始化custom构建目录的历史命令是在 `csrc/` 下执行：

```bash
bash build.sh --pkg --ops=compressor_metadata --soc=ascend910_93 --vendor_name=csa_validation -j8
```

CANN9.2出现新旧graph error header冲突后，仅通过CMake强制include参数兼容。
后续增量扩展（以下为已执行的五算子集合）：

```bash
cmake -S csrc -B csrc/build \
  '-DASCEND_OP_NAME=compressor_metadata;rms_norm_dynamic_quant;inplace_partial_rotary_mul;scatter_nd_update_sk;quant_lightning_indexer_v2' \
  "-DCMAKE_CXX_FLAGS=-include ${CANN_ROOT}/include/graph/error_codes.h"
```

**增量构建必须核对生成的ops-info。** 本次INI已包含新增算子，但两处JSON仍只有CompressorMetadata。
构建规则对INI缺少依赖，造成OPC读取内置RMS的4输入定义，拒绝custom的5输入。
已执行的修复是使用仓库原生成脚本刷新JSON，再复制到临时custom目录：

```bash
python csrc/cmake/scripts/util/parse_ini_to_json.py \
  csrc/build/autogen/aic-ascend910_93-ops-info.ini \
  csrc/build/autogen/inner/aic-ascend910_93-ops-info.ini \
  csrc/build/autogen/exc/aic-ascend910_93-ops-info.ini \
  csrc/build/autogen/aic-ascend910_93-ops-info.json

cp csrc/build/autogen/aic-ascend910_93-ops-info.json \
  csrc/build/custom/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json

cmake --build csrc/build -j8
CMAKE_BUILD_PARALLEL_LEVEL=8 cmake --build csrc/build --target package -j8

bash csrc/build/cann-ops-transformer-csa_validation_linux-aarch64.run \
  --quiet --install-path="$(pwd)/.cache/csa/csa-native-ops-install"
```

`inner` INI在本次构建中不存在，官方生成脚本允许这种情况；`scatter_nd_update_sk` 位于 `exc` INI。
确认最终JSON含全部预期算子之后才重跑，不能把一次错误opstore冲突认定为源码无法编译。

本机加载顺序为原全局vendor再加载任务目录vendor，使任务产物优先：

```bash
source "$CANN_ROOT/opp/vendors/custom_transformer/bin/set_env.bash"
source .cache/csa/csa-native-ops-install/vendors/csa_validation_transformer/bin/set_env.bash
```

Compressor、SAS仍来自旧全局vendor，最新forward因此在Compressor stride检查失败。
新机器若需要重建，至少把 `compressor` 和 `sparse_attn_sharedkv` 也纳入匹配源码的集合，
重新生成完整描述并验证依赖。**这个扩展集合尚未在本机执行**，不作为已通过构建命令记录。

## 6. ABI实验只作历史参考

[build_native_cann92.py](../build_native_cann92.py) 在隔离目录把RMS和rotary调用改为CANN9.2内置ABI。
patch为 [native-cann92-abi-experiment.patch](patches/native-cann92-abi-experiment.patch)。
它不修改生产源，必须显式 `--cann92-abi` 才会加载实验扩展。

- RMS单独修改后实际跑到rotary并失败，结果已保留。
- 加入rotary修改后的双改版本只编译成功，未再做完整运行。
- 随后配套五算子custom包构建成功，最新forward改回**原始C++扩展**，越过RMS/rotary后在Compressor失败。

因此新环境不应默认加 `--cann92-abi`。QLIv2还有candidate/stride等真实语义，不能照此盲目删参。
优先解决整个原生算子包与当前仓库匹配的问题。
