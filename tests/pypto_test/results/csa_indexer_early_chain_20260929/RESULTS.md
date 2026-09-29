# Indexer cache生产链补七处early标志

完整八类状态零容差：PASS。

同卡CANN9.2/mode2/atomic0/det0，inplace_pass=True；5预热20次正式事件，单位μs。

| 档位 | CSA基线→候选 | 变化 | P95 | max | indexer_score_topk_native_pair_aic | indexer_score_topk_native_pair_aiv | qk_pv_aic | qk_pv_aiv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 972.964→965.879 | -0.728% | 984.860→986.840 | 985.840→997.060 | 221.699→231.594 | 224.757→234.632 | 156.309→152.027 | 160.519→155.857 |
| 8K/B24 | 935.083→917.581 | -1.872% | 965.200→933.760 | 975.400→939.380 | 30.753→29.710 | 46.436→43.371 | 184.975→177.692 | 188.831→181.528 |

长短8:2 CSA变化：-0.957%；核内变化：indexer_score_topk_native_pair_aic +2.892%、indexer_score_topk_native_pair_aiv +2.195%、qk_pv_aic -2.979%、qk_pv_aiv -3.097%。

核时包含核内流水等待，独立DFX不能与正式CSA样本直接相减。
纯调度改动，不能把未改任务的核时波动记成独立incore优化；按完整CSA的8:2与P95取舍。
调度候选以完整CSA与尾部决定，核时用于排查资源争用；状态失败则禁止采用。不是Native或模型验收。
[四窗核时及原样本](summary.json)、[原始解析与完整状态](evidence.json)。

[逐窗生产链的真实early及交接等待](chain.json)。
