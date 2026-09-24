# SPDX-License-Identifier: Apache-2.0
"""CSA 算子的性能版：改用 pypto-lib 上游的数值写法，不与 Native 对齐到 bit。

与同级的 `deepseek_v4_flash_dspark`（精度版）并存，两者的集成层完全相同——
动态形状绑定、compact 元数据、越界守卫、补位与档位处理、块表适配都直接沿用，
差别只在那些为对齐 Native 数值行为而刻意写成特定形式的站点。

由来：精度版与上游 pypto-lib 在同配置（b=16、S=6、TP1、8k）下的泳道对照显示，
kernel 时间 45,496us 对 26,822us，总差距 18,675us 里有 17,123us（92%）落在这些
对齐站点上。精度版保留它们以换取与 Native 逐 token 乃至逐步接受数完全一致；
本版本放弃该性质，用于验证上游约 730us 的窗口跨度能否在本集成下复现。

因此本版本**不做与 Native 的逐 token 比对**，它本就会不一致；验收看的是泳道的
窗口跨度与逐任务 kernel 时间是否追平 pypto-lib。

参照的上游提交：07b5d2f（含 e68e091 "Perf: support S=6 and tune DSpark decode
attention"）。精度版的对齐依据见各站点原注释与 `NATIVE_*` 常量。
"""
