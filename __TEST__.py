from utils import *
from qcommands import *

L = LoL([[]]*5,'int8')
#Q(L)

S = Stacks(L, 2, expand_ratio=1.0)

#######

N = 1000000
M = [[] for n in range(N)]
for k in range(2):
    for n in range(N):
        M[n].append(n)



        
##
