# KV投影直接复用Native NZ原地址

完整八类状态零容差：PASS。

同卡CANN9.2/mode2/atomic0/det0，inplace_pass=True；5预热20次正式事件，单位μs。

| 档位 | CSA基线→候选 | 变化 | P95 | max | kv_proj_matmul | indexer_score_topk_native_pair_aic | indexer_score_topk_native_pair_aiv | qk_pv_aic | qk_pv_aiv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 969.455→970.119 | +0.068% | 981.240→983.280 | 984.480→993.680 | 23.320→12.498 | 225.191→232.036 | 228.152→235.086 | 148.423→157.319 | 152.387→161.257 |
| 8K/B24 | 919.268→913.069 | -0.674% | 934.060→929.320 | 936.520→939.260 | 37.961→19.221 | 37.240→30.007 | 53.070→45.054 | 181.305→191.665 | 185.176→195.450 |

长短8:2 CSA变化：-0.080%；核内变化：kv_proj_matmul -47.000%、indexer_score_topk_native_pair_aic -1.453%、indexer_score_topk_native_pair_aiv -0.590%、qk_pv_aic +5.937%、qk_pv_aiv +5.766%。

核时包含核内流水等待，独立DFX不能与正式CSA样本直接相减。
两侧保持相同M/N/K分块及工作编号，长B16为12份、短B24为8份KV工作。候选按原1/2/3 M组显式constexpr化以满足NZ偏移证明，完整CSA含新增编排选择成本。Native与PTO的分工不同，不能把其单核均值直接相减；此处比较PTO同工作量A/B。
采用还需依据四窗口分布判断真实核内收益，状态失败则禁止采用；不是Native或模型token/DSpark验收。
[四窗核时及原样本](summary.json)、[原始解析与完整状态](evidence.json)。
