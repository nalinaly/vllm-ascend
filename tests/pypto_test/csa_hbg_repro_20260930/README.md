# CSA HBG 编译阻塞与单卡排队尝试（2026-09-30）

**已复现 HBG kernel 编译入口错误；真实单卡任务没有启动。**
本次只定位当前阻塞点，不修改生产算子，不产生新的性能或精度结论。

## 环境与入口

| 项目 | 本次取值 |
| --- | --- |
| CSA 源码 | `c6128b72`，`dsv4-flash-pto-v0.25.1rc1` |
| PyPTO | `88f605986`，`feat/kernel-mode-integration-test` |
| Simpler | `a54c05095`，`feat/kernel-mode-integration-test` |
| 公共环境 | `env-dsv4-0251rc1.sh`，CANN 9.2.0-beta.2 |
| 算子配置 | performance、NZ mode=2、atomic_add=0 |
| HBG 配置 | kernel ABI / `pypto.torch.init` 的 runtime=`host_build_graph` |
| 拟执行单卡 case | 第 2 层正式权重，B4/8K，合成历史，不计时 |
| 权重目录 | `/data/model/DeepSeek-V4-Flash-0731-w8a8` |

原始工作目录为
`tests/pypto_test/results/csa_hbg_repro_20260930_c6128b72/`。
其中 `frozen_ops_pypto/` 保存当时整包源码，防止并发编辑干扰 JIT 重读。
源码已有 Git 版本，本提交只保存复现脚本和小型证据，不重复提交整包副本。

## 实际执行过程与首个错误

1. 确认 HBG 要绑定到 runtime；只切换 PassContext 或普通 program 编译不足以验证 eager kernel 的契约。
2. 复制算子包后，使用 `decode_csa_tp1_layer_test` 检查依赖图：成功。
3. 根据函数签名构造原始 IR，并推导 `platform=a2a3, runtime=host_build_graph` 的 kernel ABI：成功。
4. 调用 eager JIT 同样使用的 `_compile_impl(..., _kernel_abi=abi)`：立即报错。

```text
decode_indexer.py:1382:9: HBG kernel Host orchestration 'indexer_score_topk_forest' cannot use tensor.read on Tensor storage. Pass the required Host value as an explicit scalar argument.
```

对应 [decode_indexer.py](../../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)
在上述源码版本的第 1382 行：

```python
max_topk_cache_len = 0
for topk_batch in pl.range(b_dim):
    max_topk_cache_len = pl.max(max_topk_cache_len, pl.read(kv_seq_lens, [topk_batch]) // COMPRESS_RATIO)
```

Host orchestration 读取 `kv_seq_lens` 的张量内容，计算最大压缩缓存长度，
供后面的 Cube/Vector 与长短档 Score 分支选择使用。
HBG kernel 禁止 Host 直接读取 Tensor 存储；形状查询 `pl.tensor.dim`
和设备任务内部读取不属于这次报错。

调用链为 PyPTO `python/pypto/ir/compile.py:433`
→ `_kernel_compile.py::validate_hbg_kernel_orchestration`
→ `HostAccessVisitor.check_call` 抛出 `ValueError`。
错误发生在优化管线、PTOAS 和设备执行之前。

证据：[完整原始输出](compile_traceback.txt)、[结构化错误](compile_report.json)。
原输出中的绝对路径保留原样，指向当时的冻结源码与脚本。

这次编译检查在 CPU 上执行，没有初始化 NPU；签名推导保留动态形状，
不能称为已经完成 B4/8K 真机运行。它只给出首个阻塞点，
不证明其余代码支持 HBG，也不证明 HBG 性能或图重放兼容性如何。

## 真实单卡任务为什么没有启动

使用原有 `dsv4_csa_single_layer.py`，仅在进程内将 `pypto.torch.init`
的 runtime 覆盖为 `host_build_graph`，准备执行上述 B4/8K case。

- 提交任务：`task_20260930_154238_189866031600`，自动分配单卡，最大执行时间 600 秒。
- 当时 `pto-task.service` 为 `LoadState=not-found, ActiveState=inactive, SubState=dead`，未发现运行中的 task-daemon。
- 提交后状态为 pending；等待 30 秒仍未分配到设备。
- task-submit 明确输出已取消 pending 任务；随后查询返回 `not_found`。
- 没有设备执行、没有设备报错日志，也没有遗留占卡任务。

查询与等待输出转录见 [queue_attempt.json](queue_attempt.json)。
该文件按终端已返回的信息整理，未伪装成任务守护进程原始日志。

## 复现方式

本目录三个脚本是原尝试脚本的原样归档，按下面步骤复制到原有层级再运行；
不要直接在归档目录运行。复现使用上述 PyPTO/Simpler 和公共环境版本。
在仓库根目录执行，目标结果目录应为新的目录；已有原始证据时另取目录名，
保持 `tests/pypto_test/results/<实验目录>/` 的层级。

```bash
case_output=tests/pypto_test/results/csa_hbg_repro_20260930_c6128b72_repeat
mkdir -p "$case_output/frozen_ops_pypto"
git archive c6128b72 vllm_ascend/ops/pypto |
    tar -x -C "$case_output/frozen_ops_pypto" --strip-components=3
cp tests/pypto_test/csa_hbg_repro_20260930/compile_hbg.py "$case_output/"
cp tests/pypto_test/csa_hbg_repro_20260930/run_single_card.py "$case_output/"
cp tests/pypto_test/csa_hbg_repro_20260930/run_single_card.sh "$case_output/"
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
python "$case_output/compile_hbg.py" > "$case_output/compile.log" 2>&1
```

预期最后一条命令退出码为 1，错误为上述 Host `tensor.read` 限制。
单卡入口依赖仓库中的 `dsv4_csa_single_layer.py` 和 `dsv4_csa_env.py`；
要严格复现本次准备的版本，应在 `c6128b72` 的独立 checkout 中执行。
队列服务恢复后再单独提交真机任务：

```bash
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 600 \
    "bash $(realpath "$case_output/run_single_card.sh")"
```

## 后续处理边界

一种修改方向是从适配层显式传入 Host 分支需要的长度标量，
同时保证图重放时的更新语义；另一种是重新组织为受支持的设备端调度。
本次未实施任何方案，不能把 Host 元数据固化为首次调用值，
也不能通过去掉编译器校验来宣称 HBG 已兼容。

本次提交只整理已有证据，不新增占卡测试。
归档脚本的 Python/Bash 语法、JSON 解析及 Git 差异空白检查通过。
已执行仓库要求的 `bash format.sh ci`，但环境未安装 pre-commit，
该检查退出失败；不宣称全量格式检查通过。
