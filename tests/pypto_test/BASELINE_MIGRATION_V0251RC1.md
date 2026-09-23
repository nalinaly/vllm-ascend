# DSV4 CSA 基线迁移：v0.25.1rc1

2026-09-23，用户要求以官方 `v0.25.1rc1` 为基线创建新分支，并保留此前所有过程文件。
本文件是新基线的接续入口。旧机器及旧 main 基线的记录保留其原始语义；旧 PASS 不计为本基线 PASS。

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
| PyPTO / Simpler | 原 debug 分支 `feat/kernel-mode-integration-test`，未改依赖源码 |
| 正式权重 | `/data/model/DeepSeek-V4-Flash-0731-w8a8`，ModelSlim W8A8，75 分片 |
| 旧分支 / HEAD | `dsv4-flash-pto` / `18ec20ae2f93d2c3965ecffdb6c2e9d80fd304c6`，原工作目录及 WIP 保留 |

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

证据：`results/migration_v0.25.1rc1_20260923/`。未执行提交检查，也未重复历史精度矩阵。
本次未重建 release 的 Native C++ 扩展或自定义算子包，未运行 release 真机数值 / 图重放 / 完整模型。
新环境未引用旧 main 的 Native `.so` 或 custom vendor 包。因此不能声称新基线的运行环境或 NPU 验收已完成。

下一步先构建与 release 源码匹配的 Native 扩展和算子包，覆盖完整模型所需 HcPre / HcPost 等；
之后用正式权重做最小 B1/B4 eager 与图重放，再恢复 P TP4×DP4 离线 cache → D TP1×DP/EP16。
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
历史脚本保留不等于已适配：除本节列出的 release 检查外，其余 Native fixture、服务图重放和
offline P/D 脚本尚需按 release 接口调整，不能直接拿旧结果作为通过依据。

大 `.pt` 快照完整保留在新工作目录的原路径。Git 保存其清单，普通源码提交不包含多 GiB 张量数据。
原过程文件与新基线说明单独提交，生产 CSA 提交中不包含测试、文档和历史产物。

## 旧 P 任务最终状态补记

`task_20260923_140032_228587613477` 已失败，并非仍在加载：TP4×DP4 / EP16 初始化及正式
75 分片加载完成，加载 draft 后在启动 memory profiling 调用 `aclnnHcPre` 时失败。
旧环境安装的是此前单层 CSA 的 9 算子包，没有 HcPre / HcPost；未生成可用离线 KV cache。
旧目录随后发起的 15 算子 CPU 构建，在用户要求切换基线后已停止，未安装其未完成产物。
