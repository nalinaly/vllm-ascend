# Q_B 24→20 workers：独立双档候选

基线为f4861832对应已验证early包；独立整包副本v3，源码只读。
仅性能入口显式传20，ND/NZ底层helper按constexpr调整分工；原五参数入口显式传24，保护精度版调用契约。
INT8/INT32算术、列覆盖、NZ物理布局、Q_B依赖/early和其他任务保持。
不与KV链early、Q_B early或dequant分工一起修改；生产代码尚未改变。

依据HCA c5f4252a的独立20-worker思路，但CSA并非固定有四核被占。
已有长档24份Q_B只落16–22个物理AIC核；少worker可能降低多波交接，也可能拉长每worker负载。
128个NZ列块在20份下每份6–7块，24份下5–6块；ND64列块也完整覆盖一次。
上游pypto-lib 2164563仍为24份持久分工。
最新ops-nn 19614968 qmatmulv3考虑波次与使用核数，但其K>=4096/N>=7168/M<=256的特定规则
不适用于本Q_B的K1024，不能宣称直接复制Native tiling，也不能保证20份就是单波。

CPU检查发现嵌套dep调用不读取constexpr默认值，V1在ND/NZ旧调用均失败；
V2 wrapper直接返回子调用tuple不受inline pass支持，已改显式解包后返回。
失败记录保留为default_failure_v1/v2.json，两版都未占卡。
v3性能入口完整编译/load通过，AICPU生成码Q_B block_num=20；
ND/NZ原五参数入口分别编译/load通过，均仍block_num=24；基线两入口解析通过。
这只验证共享调用兼容，不冒称精度版整模型或设备验收。

16:04正常auto提交task_20260929_160457_135068517850，确认设备0上running。
单卡128K/B16、8K/B24，同卡反序两侧5预热20次正式计时及各四窗DFX。
比较完整CSA的8:2、P95/max、Q_B全部worker核时总和/首末跨度及后续query→Score链。
worker工作份数不同，不能用单worker均值变大判断优化失败，也不能仅凭分工减少认定收益。
八类跨版本状态零容差、图重放/保护区及官方worker覆盖先过；有收益再补受影响边界。
本候选不重新采Native、七档或整模型。

[两文件差异](candidate.patch)、[来源与列分工](source.json)、[编译依据](static_evidence.json)、
[双档入口](run.sh)、[分析入口](collect.py)。
