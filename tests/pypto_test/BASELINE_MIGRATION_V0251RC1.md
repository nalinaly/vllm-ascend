# DSV4 CSA 基线迁移：v0.25.1rc1

2026-09-23，用户要求以官方 `v0.25.1rc1` 为基线创建新分支，并保留此前所有过程文件。
本文件是新基线的接续入口。旧机器及旧 main 基线的记录保留其原始语义；旧 PASS 不计为本基线 PASS。

2026-09-23 后续：用户明确要求删除旧工作目录和旧分支，现已完成。新分支已推送；
共享 Git 元数据迁至工作区 `.git-repositories/vllm-ascend.git`，另两个修复 worktree 保持可用。
PTOAS / ATB / 构建工具迁至工作区 `.cache/dsv4-toolchain`，`../env.sh` 也指向新 release 环境。
删除前的最终 WIP 和入口快照保存在
`handoff/legacy_main_18ec20a/`。305 份大快照仍保留在新目录，未纳入 Git。

## 固定版本与目录

| 项目 | 版本或路径 |
| --- | --- |
| vLLM-Ascend 起点 | 官方 `v0.25.1rc1`，`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` |
| 新分支 | `dsv4-flash-pto-v0.25.1rc1` |
| 新工作目录 | `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1` |
| 配套 vLLM | 官方 `v0.25.1`，`752a3a504485790a2e8491cacbb35c137339ad34` |
| 配套 vLLM 源码 | `../.cache/migration-v0.25.1rc1/vllm` |
| Python 环境 | `../.venv-dsv4-0251rc1`，Python 3.10；独立于旧 `.venv` |
| 环境入口 | `source ../env-dsv4-0251rc1.sh` |
| 当前 CANN / PTA | CANN 9.0.0，torch / torch_npu 2.10.0；具体包版本见环境快照 |
| 官方 A3 Dockerfile | CANN 9.0.1、Python 3.12、torch / torch_npu 2.10.0 系列；本地尚不是完全相同的镜像 |
| PyPTO / Simpler | debug分支 `feat/kernel-mode-integration-test`：5495749 / 166852bf；保留已有torch_npu 2.10本地适配并同步SDK版本绑定，详见日志第92节 |
| 正式权重 | `/data/model/DeepSeek-V4-Flash-0731-w8a8`，ModelSlim W8A8，75 分片 |
| 旧分支 / HEAD | 原 `dsv4-flash-pto` / `18ec20ae2f93d2c3965ecffdb6c2e9d80fd304c6`；分支及目录已删除，最终 WIP 已归档 |

## 迁移范围与接口处理

生产算子来自原 `f009e4f`、`18ec20a` 和后续动态 batch / 服务入口 WIP。
保留完整 CSA 一次 PTO 提交、动态 B=1..64 / S=6、精度修正、Indexer 整页读取、
Native device positions / slots / 页表，以及直接读写两组 Native state 的实现。
B=64 是实现容量上限，不是已完成数值验收的声明。

| 差异 | 新分支处理 |
| --- | --- |
| 模型文件 | 继承 release `vllm_ascend.models.deepseek_v4.AscendDeepseekV4ForCausalLM` |
| 请求 metadata | 读取 release 的 `.decode`；使用已有 host `max_seqlen_q` 判定均匀 S6，不新增 Native metadata 字段 |
| 紧凑 metadata | 调用 release `AscendDSAImpl._compute_compressor_metadata`，传入返回的原始 device Tensor；不展开或搬运 |
| metadata 同步 | release 在消费者 stream 上创建紧凑行；不引入 main 的 DeviceMetadataExecutor / ExternalEvent API |
| Native fallback | 使用 release context 的 `flash_comm_v1_enabled` |
| 权重准备 | release 无模型级 post-load hook；仅 PTO alias 在 runner 的 `get_model` 返回后准备资源，早于 profiling / graph capture |
| 图派发 | 保留 host gate，拒绝将非均匀或有 padding 的请求重放到 PTO S6 bucket；Native alias 不改变 |
| Indexer dtype | release A3 Indexer 构造器直接指定 INT8；不添加 main 的 `indexer_kv_dtype=int8` Literal 补丁，adapter 校验实际 INT8/FP16 共享 storage |

原 `db6f3e1` 两个 Native 修复不直接移植：

- release 使用 `vllm_quant_lightning_indexer`，不存在 main QLIv2 的 `qli_seqused_k` / `qli_cmp_residual_k` 缓冲问题。
- release 使用 `scatter_nd_update_v2`；其普通和 large-index 路径在写入前检查有符号 linear index 所属范围，
  不存在原 `scatter_nd_update_sk` 两个快速路径的无符号负 slot 转换点。
- 原补丁保存在 `handoff/legacy_main_18ec20a/native-fixes.patch`，供后续比对。

没有迁入 main 的 Native 模型、QLIv2、metadata executor 或其他大范围实现。

## 本次验证与实际边界

已通过：

- 隔离环境中的真实模型、release metadata、CSA custom op 与 runner 导入。
- `tests/ut/ops/test_dsv4_csa_service.py`：18 项 CPU 检查；包括实际 runner graph gate、
  Native fallback 条件，以及 release dataclass / 紧凑行 / cache-state 零拷贝绑定和单次调用。
- 完整 CSA CPU lowering，目标 `a2a3`。
- release Native C++ 扩展及 14 算子包在本机 CANN 9.0.0 构建、安装通过；`csrc` 与官方基线一致。
- A3 使用正式 checkpoint layer2 HC 权重执行 HcPre/HcPost，通过输出形状与有限值检查。
- SAS、QLI、Compressor 三种 Native metadata 真机执行通过。

证据：`results/migration_v0.25.1rc1_20260923/`。未执行提交检查，也未重复历史精度矩阵。
Native 构建与执行证据：`results/release_offline_pd_20260923/`，包含源码清单、包 SHA256、
API 符号和构建日志。执行任务 `task_20260923_150737_24250853714` 完成，exit 0。
HcPre 缺失问题已消除；这次执行检查不代表完整 CSA 数值 / 图重放 / 整模型已验收。
新环境只加载新 release 的 Native `.so` 与 custom vendor 包。

按用户要求恢复 P TP4×DP4 离线 cache → D TP1×DP/EP16。
离线脚本已按 release 接口适配；短场景 H255×4 的 P 任务
`task_20260923_150829_249108112866` 已完成(exit 0)，16份副本各191个tensor。
有效前缀直接逐bit比较及全部target/draft层覆盖通过；按用户要求，离线流程不做hash校验。
首轮D任务`task_20260923_154135_38813947929`的Native轮完成16rank×4request的缓存加载和128token输出。
PTO轮因compressor norm的BF16/FP32准备约束失败。按用户要求撤销初始化拓宽，
旧方案任务`task_20260923_161904_380071921124`在排队时取消，未运行。
CSA主入口及两路compressor均改为直接接收Native BF16 norm，保留原始地址；
仅在已有RMS设备任务内将加载的gamma tile转FP32，无额外适配调用或FP32权重副本。
BF16零拷贝准备的CPU回归检查与完整CSA lowering/PTOAS代码生成通过。
新PTO D16任务`task_20260923_163126_45104725192`完成，exit1：16rank的BF16权重准备
及64个请求缓存恢复通过，首次PTO调用前被idx_kv_cache可写别名检查拒绝，未进入设备计算。
已确认所有重叠均为同字节范围的合法共享视图。PyPTO/Simpler调试分支已更新并重新安装
至5495749/166852bf，PR #2867解决上述参数检查；CPU描述符回归、ABI一致性、完整CSA
编译和一个eager真机用例通过。重测任务`task_20260923_175252_286523232409`已完成
exit0：D TP1×DP16/EP16、history255、每rank B4的64请求均完成128 token生成；
16rank×21层均记录PTO执行，8192个输出token与此前Native D16完全一致。
本轮含首次编译及IO，不作为稳态性能结论；详见验证日志第91～93节。
release 的 Native compressor / QLI 实现与旧 main 不完全相同，数值结果需重新比较。

```bash
source ../env-dsv4-0251rc1.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests/ut/ops/test_dsv4_csa_service.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/pypto_test/dsv4_csa_reference_lower.py \
  --output-dir tests/pypto_test/results/migration_v0.25.1rc1_20260923/lowering
```

## 历史过程文件

整个旧 `tests/pypto_test` 目录已复制，包含计划、日志、复现说明、脚本、失败与成功结果、
profiler 数据、泳道图、生成代码和张量快照。旧目录外的相关 UT / WIP 补丁以及环境构建日志
保存在 `handoff/legacy_main_18ec20a/`。该目录记录原 main 接口，不参与本次 release 验收。

`handoff/MIGRATION_PROCESS_FILES.json` 记录旧过程文件的相对路径、字节数和 SHA256。
需要为 release 调整的入口及新增迁移说明所涉及的原文另存到 `handoff/legacy_main_18ec20a/`。
历史脚本保留不等于已适配：除本节列出的 release 检查及 offline P/D 入口外，其余 Native fixture
和服务图重放脚本尚需按 release 接口调整，不能直接拿旧结果作为通过依据。

大 `.pt` 快照完整保留在新工作目录的原路径。Git 保存其清单，普通源码提交不包含多 GiB 张量数据。
原过程文件与新基线说明单独提交，生产 CSA 提交中不包含测试、文档和历史产物。

## 旧 P 任务最终状态补记

`task_20260923_140032_228587613477` 已失败，并非仍在加载：TP4×DP4 / EP16 初始化及正式
75 分片加载完成，加载 draft 后在启动 memory profiling 调用 `aclnnHcPre` 时失败。
旧环境安装的是此前单层 CSA 的 9 算子包，没有 HcPre / HcPost；未生成可用离线 KV cache。
旧目录随后发起的 15 算子 CPU 构建，在用户要求切换基线后已停止，未安装其未完成产物。
