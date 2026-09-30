# 七档 Native SuperKernel 开关对比（2026-09-30）

源 `.cache/csa-native-superkernel-4ffccb7b`，入口
`torch.compile(module, backend="npugraph_ex", fullgraph=True, dynamic=False,
options={force_eager: False, inplace_pass: False, static_kernel_compile: True,
super_kernel_optimize: <0|1>})`。这是当前唯一能让 **AclGraph 级 SuperKernel
真正生效**的入口：vllm 的 `@support_torch_compile` 带 `force_eager: True`，
那条路只有 op_compiler 的 `--enable_super_kernel` 半边生效，图内融合永不执行。

生效判据用 `compiler.static_super_flags`（sk=0 → `[false]`、sk=1 → `[true]`）
与顶层 `super_kernel`，**不能用 `installed_static_packages`**（它只说明
static_kernel 装了包）。

每档在**同一张卡**上按 `sk0_a → sk1_a → sk1_b → sk0_b` 的 ABBA 跑，每侧 20
正式设备事件，表内每侧取两次的平均；超过 1.25×p50 的样本剔除。
七档并行占七张卡。CANN 9.2.0-beta.2、mode2、atomic0、det0。单位 μs。

| 档位 | sk=0 min | sk=0 mean | sk=0 max | sk=1 min | sk=1 mean | sk=1 max | SK 收益(mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 927.22 | 1002.68 | 1134.20 | 795.89 | 850.18 | 914.52 | **15.21%** |
| 128K/B8 | 923.95 | 930.21 | 971.02 | 856.61 | 860.10 | 864.71 | **7.54%** |
| 128K/B16 | 1217.71 | 1223.75 | 1230.43 | 1144.26 | 1150.18 | 1157.05 | **6.01%** |
| 128K/B24 | 1400.50 | 1408.32 | 1419.78 | 1276.96 | 1289.71 | 1296.77 | **8.42%** |
| 8K/B16 | 1031.78 | 1070.23 | 1167.88 | 914.63 | 953.20 | 1005.21 | **10.93%** |
| 8K/B24 | 1179.19 | 1198.65 | 1253.34 | 1094.69 | 1119.64 | 1177.53 | **6.59%** |
| 8K/B32 | 1302.34 | 1323.32 | 1365.85 | 1178.36 | 1208.52 | 1362.42 | **8.67%** |

**128K 平均 9.29%、8K 平均 8.73%、8:2 加权 9.18%。**

## 数据卫生

- 8K/B32 第一轮的 `sk0_b` 整槽被外部负载污染（p50 1651.85，同档 `sk0_a`
  1313.17），是整槽偏移而非单点尖峰，样本级的 1.25×p50 过滤挡不住。
  该档已重跑一整轮完整 ABBA，表内是重跑值；污染轮留在
  `h8192_b32.round1_contaminated/`。
  **筛查办法：除样本级过滤外，还要看同侧两槽的均值差，超过档位自身极差就重跑那档。**
- 重跑后各档同侧两槽的均值差：128K/B4 45.2/75.3、128K/B8 19.1/3.9、
  128K/B16 15.0/20.4、128K/B24 12.6/3.1、8K/B16 40.7/50.7、8K/B24 7.7/75.8、
  8K/B32 1.7/13.7（sk0/sk1）。每档的 SK 收益都大于该档自身的漂移。

## 与验收 harness 的 Native 不可互换

`../csa_accept_20260930` 的 Native 走 vllm 装饰器入口。两条入口除
SuperKernel 外还差 **vLLM 的 FX 融合**——直接入口的 report 自己写明
`compilation_scope: {"vllm_ascend_fx_passes": false}`，即 `fuse_norm_quant` /
`fuse_qknorm_rope` / `fuse_muls_add` 虽在 `requested` 里为 true 却未应用。
因此两个目录的 Native 读数在 128K/B4 差 25.5%、8K 三档差 11.5%~22.9%，
**不能拿一个目录的 PTO 去除另一个目录的 Native**。详见日志第 501 节。
