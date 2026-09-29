# 七处early及O-B/HC收尾采用后的PTO七档与现有Native

Native复用任务task_20260929_095651_194851727802的最新标准基线；PTO两档复用task_20260929_151902_390858217467，五档补测task_20260929_154347_45284811883。
PTO为同一冻结源码；各侧设备/来源在summary/evidence中逐项保留。
这是当前已有结果对照，不是同次Native/PTO A/B，不用于归因单项优化收益。
七处early的独立同卡A/B见../csa_indexer_early_chain_20260929/RESULTS.md；此前O-B及收尾融合的独立收益分别记录，不能把累计变化归因于early单项。
七档PTO各自图重放的八类状态通过；旧七档未保存完整state，不能声称七档跨版本状态一致。
跨版本状态依据仅限已完成的两档A/B和尾段/padding；不是Native/PTO或模型token/DSpark验收。

CANN9.2/mode2/det0；PTO atomic0；每档5预热20次设备事件，单位μs。
Native显式torch.compile backend=npugraph_ex、dynamic=False/fullgraph=True/inplace_pass=True，自管图/static/superkernel；PTO生产自定义算子图。

| 档位 | Native | PTO | 耗时变化 | Native/PTO P95 | Native/PTO max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 748.298 | 642.806 | -14.098% | 751.100/656.580 | 751.840/657.660 |
| 128K/B8 | 859.563 | 738.032 | -14.139% | 861.620/748.320 | 864.720/755.440 |
| 128K/B16 | 1130.853 | 965.879 | -14.588% | 1135.580/986.840 | 1139.520/997.060 |
| 128K/B24 | 1281.888 | 1240.775 | -3.207% | 1289.880/1262.700 | 1290.440/1264.280 |
| 8K/B16 | 757.482 | 753.921 | -0.470% | 760.120/769.420 | 761.040/784.460 |
| 8K/B24 | 915.371 | 917.581 | +0.241% | 920.340/933.760 | 920.900/939.380 |
| 8K/B32 | 1062.079 | 1064.999 | +0.275% | 1066.220/1088.760 | 1070.680/1099.080 |

各上下文内batch等权，长短8:2：-9.203%。

核内数据来自独立profile/DFX；以下完整融合范围不能拆成单个Native QLI或Sparse核时，
也不能与各PTO均值直接相减计算纯算术收益。完整原始kernel名称和范围见evidence。

| 档位 | Native可见端点及完整范围AIC/AIV | PTO系数 | PTO Score AIC/AIV | PTO Indexer merge | PTO融合Sparse AIC/AIV |
| --- | --- | ---: | ---: | ---: | ---: |
| 128K/B4 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 273.608/276.361 | 1.695 | 67.824/74.492 | 4.780 | 44.666/50.676 |
| 128K/B8 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 317.510/320.053 | 2.296 | 118.887/123.043 | 6.358 | 79.235/83.032 |
| 128K/B16 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 517.282/519.891 | 2.589 | 231.594/234.632 | 8.326 | 152.027/155.857 |
| 128K/B24 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 582.740/585.160 | 4.198 | 328.287/335.803 | 10.960 | 229.268/233.282 |
| 8K/B16 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 136.321/138.705 | 4.953 | 41.932/46.313 | 7.936 | 127.463/131.246 |
| 8K/B24 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 192.451/194.807 | 2.610 | 29.710/43.371 | 11.271 | 177.692/181.528 |
| 8K/B32 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 292.632/295.272 | 9.555 | 80.880/85.184 | 10.752 | 237.615/241.586 |

PTO已融合最终归一化、逆RoPE和发布，没有独立merge_norm；不是把缺失样本填0。

核内细节另采Native static compile开启、SuperKernel关闭的profile；不把其区间填入主性能表。

| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO系数/merge | Native Sparse AIC/AIV | PTO融合Sparse AIC/AIV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 66.363/64.369 | 67.824/74.492 | 1.695/4.780 | 53.214/57.090 | 44.666/50.676 |
| 128K/B8 | 126.316/126.021 | 118.887/123.043 | 2.296/6.358 | 95.667/99.998 | 79.235/83.032 |
| 128K/B16 | 242.293/241.730 | 231.594/234.632 | 2.589/8.326 | 167.255/169.662 | 152.027/155.857 |
| 128K/B24 | 376.467/376.063 | 328.287/335.803 | 4.198/10.960 | 221.309/223.867 | 229.268/233.282 |
| 8K/B16 | 39.630/39.050 | 41.932/46.313 | 4.953/7.936 | 95.677/98.084 | 127.463/131.246 |
| 8K/B24 | 56.051/55.666 | 29.710/43.371 | 2.610/11.271 | 166.763/169.371 | 177.692/181.528 |
| 8K/B32 | 68.901/68.365 | 80.880/85.184 | 9.555/10.752 | 210.209/212.812 | 237.615/241.586 |

Native Sparse不含独立逆RoPE，而PTO融合Sparse包含它；以上仍是不同边界的参考，不能机械相减。

| 档位 | PTO P95/P50 | PTO max/P50 | 超过P50的105% |
| --- | ---: | ---: | ---: |
| 128K/B4 | 1.0222 | 1.0239 | 0/20 |
| 128K/B8 | 1.0154 | 1.0250 | 0/20 |
| 128K/B16 | 1.0241 | 1.0347 | 0/20 |
| 128K/B24 | 1.0182 | 1.0195 | 0/20 |
| 8K/B16 | 1.0205 | 1.0405 | 0/20 |
| 8K/B24 | 1.0185 | 1.0246 | 0/20 |
| 8K/B32 | 1.0259 | 1.0357 | 0/20 |

不删拖尾样本；20次正常采样不能关闭历史间歇拖尾或EP16稳定性问题。
Native重放对同图compiled callable、PTO对自身eager；不冒称Native/PTO跨实现精度通过。
全部配置、样本、28个官方join核对窗口、两侧正式路径的14个PyTorch JSON及7个Native核内诊断JSON见[evidence.json](evidence.json)。
