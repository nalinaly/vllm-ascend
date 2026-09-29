# Indexer query整块Gather

完整八类状态零容差：PASS。

同卡CANN9.2/mode2/atomic0/det0，inplace_pass=True；5预热20次正式事件，单位μs。

| 档位 | CSA基线→候选 | 变化 | P95 | max | idx_qr_dequant_rope |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 966.592→966.728 | +0.014% | 974.760→977.880 | 985.660→981.920 | 14.081→12.062 |
| 8K/B24 | 911.350→921.795 | +1.146% | 921.520→934.700 | 925.540→936.960 | 19.124→15.566 |

长短8:2 CSA变化：+0.240%；核内变化：idx_qr_dequant_rope -15.193%。

核时包含核内流水等待，独立DFX不能与正式CSA样本直接相减。
满行从连续8x128反量化输入按绝对元素索引Gather成1x512，再恢复8x64；避免直接展平带128列行距的64列切片。保留逐head、乘法/舍入及尾行。
H4095/B3真实筛选、尾行/padding通过后，按核内收益保留规则已接入；CSA/P95回退单列。不是Native或模型token/DSpark验收。
[采用取舍](decision.json)、[边界](boundary_selective/summary.json)、[既有Native差异](native_reference_residual.json)。
[四窗核时及原样本](summary.json)、[原始解析与完整状态](evidence.json)。
