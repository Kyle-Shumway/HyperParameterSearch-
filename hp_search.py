import itertools
import os
from math import exp, expm1
def comb_lists(b, l, m, ):
    print(list(itertools.product(b, l, m)))

bRange = list(range(20,1000,5))
bSize = list()
lRange = list()
lRate = list()
mRange = list(range(10000,100000,1000))
mEps = list()
envs = list("Pong-v0")
envKey = list()
deci = range(1,8)


for i in deci:
    lRange.append(.1**i)

for b in bRange:
    bSize.append("-b " + str(b))

for l in lRange:
    lRate.append("-l " + str(l))

for m in mRange:
    mEps.append("-m " + str(m))

comb_lists(bSize , lRate , mEps)
