# DSV4 CSA Native/PTO profiling 与 PTO 泳道图

本次为已接入 Native state 的完整单层 CSA，B40、S=6、history=131073、TP1、ND，
`model.layers.2.attn`，seed1024，同一 logical device8。
使用 `/data/model/dsv4-flash-0731-dspark-w8a8` 参考权重；不代表正式 ModelSlim 整模型验收。

可将以下 JSON 直接导入 Perfetto 查看：

| 文件 | 内容 |
| --- | --- |
| [native/native_profiling.json](native/native_profiling.json) | Native PyTorch/torch_npu CPU+NPU trace，3次完整CSA ACL Graph重放 |
| [pypto/pypto_profiling.json](pypto/pypto_profiling.json) | PTO同条件trace，每次重放1次完整CSA提交 |
| [swimlane/merged_swimlane.json](swimlane/merged_swimlane.json) | 一次PTO CSA调用，AIC/AIV/调度泳道、真实核函数名和依赖边 |

两份profile在独立进程中采集，使用Level1，关闭stack、shape、memory及DFX；
各有2次eager预热、1次capture、1次graph预热，随后只采3次图重放。
搜索 `csa.profile.replay.0/1/2` 可定位各次采样，搜索 `AscendCL@aclmdlRIExecuteAsync` 可定位图提交。
每个marker包含Native metadata生产、完整CSA重放及完成同步；设备跨度包含两个stream，
按其中最早设备任务start到最晚end计，不包含加载、编译和预热。

| 项目 | Native | PTO |
| --- | --- | --- |
| 三次设备跨度（微秒） | 3010.64 / 2473.26 / 2432.02 | 56087.22 / 31145.62 / 31018.80 |
| 设备跨度中位数（微秒） | 2473.26 | 31145.62 |
| host提交到同步中位数（微秒） | 2845.91 | 31562.95 |

该组三次采样的PTO设备跨度为Native的12.593倍。首样本完整保留；本次固定输入单层采集
用于查看当前执行情况，不是整模型吞吐测试。输入、5份初始allocation和5组页表hash相同，
各自graph输出与eager逐bit相同，Native/PTO输出在冻结门槛下PASS，max_abs=0.0078125。
逐次核函数数量、条件核对和数值记录见 [comparison.json](comparison.json)。

泳道图另开eager进程，用level4+dep_gen只记录一次生产CSA，记录1265条原始AICore任务，
转换为2549个设备任务切片，并保留4220对依赖flow。原始数据在
`swimlane/chip_swimlane_records.json` 与 `swimlane/deps.json`。
`swimlane/name_map.json` 的45个名称来自本次实际编译的kernel_config，所有有效ID均覆盖。
DFX带采样及边界同步开销，不能与上表ACL Graph耗时混用。

## 在仓库中复现

```bash
source ../env.sh
task-submit --device auto --max-time 1800 \
  "bash $PWD/tests/pypto_test/run_csa_profiles.sh $PWD/tests/pypto_test/results/cann90_20260921/profile_new_run /data/model/dsv4-flash-0731-dspark-w8a8 --batch 40 --history 131073 --replays 3"
```

三个采集进程顺序使用同一张队列分配的卡；使用新输出目录，脚本拒绝覆盖已有run_metadata。
完成后从任务日志取得DFX进程对应的实际`build_output/_jit__decode_csa_tp1_attention_*/kernel_config.py`，
在CPU侧运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/pypto_test/compare_dsv4_csa_profiles.py \
  --root tests/pypto_test/results/cann90_20260921/profile_new_run \
  --swimlane-kernel-config <本次DFX实际编译目录>/kernel_config.py
```

对本包已有原始数据，仅重导出带名称泳道图无需NPU：

```bash
python -m simpler_setup.tools.swimlane_converter swimlane/chip_swimlane_records.json \
  --func-names swimlane/name_map.json -o swimlane/merged_swimlane.json
```

任务ID为`task_20260922_185453_15838428498`，exit0。
`manifest.json`记录采集源码和CPU后处理源码hash；`sources/`保存对应源码。
方法参考Qwen仓库的`qwen3_aclgraph_profile.py`、`qwen3_single_layer_swimlane.py`及对比脚本，
本次复用DSV4现有完整CSA和Native fixture，未修改计算算法或依赖仓库。
