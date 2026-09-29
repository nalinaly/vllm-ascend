# Q与Sparse整块Gather采用后的PTO七档与现有Native

Native复用任务task_20260929_095651_194851727802的最新标准基线；PTO两档复用task_20260929_193859_406772521712，五档补测task_20260929_200739_17098231176。
PTO为同一冻结源码；各侧设备/来源在summary/evidence中逐项保留。
这是当前已有结果对照，不是同次Native/PTO A/B，不用于归因单项优化收益。
Sparse整块Gather的独立同卡A/B见../csa_sparse_rope_flat_gather_20260929/RESULTS.md；Q整块Gather及此前各项收益分别记录，不能把累计变化归因于Sparse单项。
七档PTO各自图重放的八类状态通过；旧七档未保存完整state，不能声称七档跨版本状态一致。
跨版本状态依据仅限已完成的两档A/B和尾段/padding；不是Native/PTO或模型token/DSpark验收。

CANN9.2/mode2/det0；PTO atomic0；每档5预热20次设备事件，单位μs。
Native显式torch.compile backend=npugraph_ex、dynamic=False/fullgraph=True/inplace_pass=True，自管图/static/superkernel；PTO生产自定义算子图。

| 档位 | Native | PTO | 耗时变化 | Native/PTO P95 | Native/PTO max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 748.298 | 642.614 | -14.123% | 751.100/652.380 | 751.840/656.820 |
| 128K/B8 | 859.563 | 733.228 | -14.698% | 861.620/745.140 | 864.720/745.940 |
| 128K/B16 | 1130.853 | 964.806 | -14.683% | 1135.580/978.360 | 1139.520/986.840 |
| 128K/B24 | 1281.888 | 1230.986 | -3.971% | 1289.880/1247.340 | 1290.440/1251.140 |
| 8K/B16 | 757.482 | 744.598 | -1.701% | 760.120/753.860 | 761.040/760.160 |
| 8K/B24 | 915.371 | 909.341 | -0.659% | 920.340/924.280 | 920.900/933.760 |
| 8K/B32 | 1062.079 | 1038.443 | -2.225% | 1066.220/1055.960 | 1070.680/1057.700 |

各上下文内batch等权，长短8:2：-9.801%。

核内数据来自独立profile/DFX；以下完整融合范围不能拆成单个Native QLI或Sparse核时，
也不能与各PTO均值直接相减计算纯算术收益。完整原始kernel名称和范围见evidence。

| 档位 | Native可见端点及完整范围AIC/AIV | PTO系数 | PTO Score AIC/AIV | PTO Indexer merge | PTO融合Sparse AIC/AIV |
| --- | --- | ---: | ---: | ---: | ---: |
| 128K/B4 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 273.608/276.361 | 1.590 | 64.517/71.367 | 8.273 | 46.034/49.832 |
| 128K/B8 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 317.510/320.053 | 1.780 | 115.175/119.481 | 10.076 | 78.349/81.856 |
| 128K/B16 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 517.282/519.891 | 2.901 | 223.657/226.679 | 9.890 | 148.476/152.117 |
| 128K/B24 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 582.740/585.160 | 2.303 | 330.191/337.861 | 13.820 | 220.398/223.808 |
| 8K/B16 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 136.321/138.705 | 6.235 | 40.567/45.043 | 8.747 | 123.951/127.339 |
| 8K/B24 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 192.451/194.807 | 3.312 | 30.576/45.381 | 8.513 | 175.114/178.563 |
| 8K/B32 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 292.632/295.272 | 3.683 | 77.159/81.395 | 11.351 | 233.531/237.101 |

PTO已融合最终归一化、逆RoPE和发布，没有独立merge_norm；不是把缺失样本填0。

核内细节另采Native static compile开启、SuperKernel关闭的profile；不把其区间填入主性能表。

| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO系数/merge | Native Sparse AIC/AIV | PTO融合Sparse AIC/AIV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 66.363/64.369 | 64.517/71.367 | 1.590/8.273 | 53.214/57.090 | 46.034/49.832 |
| 128K/B8 | 126.316/126.021 | 115.175/119.481 | 1.780/10.076 | 95.667/99.998 | 78.349/81.856 |
| 128K/B16 | 242.293/241.730 | 223.657/226.679 | 2.901/9.890 | 167.255/169.662 | 148.476/152.117 |
| 128K/B24 | 376.467/376.063 | 330.191/337.861 | 2.303/13.820 | 221.309/223.867 | 220.398/223.808 |
| 8K/B16 | 39.630/39.050 | 40.567/45.043 | 6.235/8.747 | 95.677/98.084 | 123.951/127.339 |
| 8K/B24 | 56.051/55.666 | 30.576/45.381 | 3.312/8.513 | 166.763/169.371 | 175.114/178.563 |
| 8K/B32 | 68.901/68.365 | 77.159/81.395 | 3.683/11.351 | 210.209/212.812 | 233.531/237.101 |

Native Sparse不含独立逆RoPE，而PTO融合Sparse包含它；以上仍是不同边界的参考，不能机械相减。

| 档位 | PTO P95/P50 | PTO max/P50 | 超过P50的105% |
| --- | ---: | ---: | ---: |
| 128K/B4 | 1.0167 | 1.0236 | 0/20 |
| 128K/B8 | 1.0161 | 1.0172 | 0/20 |
| 128K/B16 | 1.0122 | 1.0209 | 0/20 |
| 128K/B24 | 1.0118 | 1.0148 | 0/20 |
| 8K/B16 | 1.0130 | 1.0214 | 0/20 |
| 8K/B24 | 1.0141 | 1.0245 | 0/20 |
| 8K/B32 | 1.0142 | 1.0158 | 0/20 |

不删拖尾样本；20次正常采样不能关闭历史间歇拖尾或EP16稳定性问题。
Native重放对同图compiled callable、PTO对自身eager；不冒称Native/PTO跨实现精度通过。
全部配置、样本、28个官方join核对窗口、两侧正式路径的14个PyTorch JSON及7个Native核内诊断JSON见[evidence.json](evidence.json)。
