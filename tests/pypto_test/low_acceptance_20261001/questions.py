# SPDX-License-Identifier: Apache-2.0
"""固定问题模板；参数变化由prepare_mixed.py定义。"""

QUESTIONS = [
    ('python_trace', '''前面的背景阅读已结束。请精确推演以下Python程序，不调用工具。
from collections import deque
q = deque([19, 7, 41, 13, 29, 3, 37])
s = 23
out = []
for i in range(1, 25):
    q.rotate((i % 5) - 2)
    a = q.popleft()
    b = (a * 17 + s * 3 + i * i) % 257
    if b % 3 == 0:
        q.appendleft((b + i * 11) % 263)
    else:
        q.append(b)
    s = (s ^ (b + i)) % 251
    out.append([i, a, b, s])
print(out)
请直接输出print得到的完整嵌套列表，不要贴代码、不要省略元素。之后给出最终q中元素的顺序。'''),
    ('ledger_transform', '''背景材料已经结束，下面是本题完整账本，字段为id,amount,fee,tag：
k7,1837,41,B
p2,2941,67,A
m9,1723,29,C
v4,3829,83,B
a8,2153,47,A
r1,3467,71,C
c6,1289,37,B
h3,4513,89,A
t5,2671,53,C
n0,3191,61,A
e4,1597,43,C
w8,4093,79,B
b2,2381,59,C
s6,3761,73,A
j9,1879,31,B
f1,4933,97,C
z3,3259,65,B
d5,2749,57,A
u7,1423,23,C
g0,3623,69,B
对每行计算 net=amount-3*fee，然后计算 score=(net*7+fee*11)%1009。
先按score降序排列全部记录；score相同则按id字典序升序。
只输出一个JSON数组，每项依次含id、net、score、tag四个字段；不得遗漏记录。'''),
    ('transition_trace', '''此前材料结束，请只解答下面的离散状态机问题。
初始状态S=4、寄存器R=137。输入依次为：
8,3,17,6,21,4,13,9,2,19,7,25,11,5,23,16,1,14,27,12,20,10,26,15
对第i个输入a（i从1开始），先以旧S、R计算 t=(R+31*a+17*S)%509。
若t%4=0，新S=(S+a+3)%11；若t%4=1，新S=(2*S+5)%11；否则新S=(S*3+a)%11。
然后新R=(t*7+新S*13+i)%521。对全部24个输入逐个更新。
请直接输出24行，每行格式为 i|a|t|新S|新R；不要输出代码或省略中间步骤。最后给出所有新R的和。'''),
]
