from fileformats import load_data
from utils import *

with open('P2020.csv','r') as file:
    C = file.read()


##
##print(C.head)
##
##x = [float(v) if v else 0 for v in C['lng']]
##y = [float(v) if v else 0 for v in C['lat']]
##
##
##print(C[21663])
##
##I = 21663
##P = Point(x[I],y[I])
##R = list(range(len(C)))
##D = [Point(x[n],y[n]).distance(P) for n in R]
##
##Z = [(D[n],n) for n in R]
##
##Z.sort()
##
##h = "dispatch_date_time location_block text_general_code".split()
##for n in range(100):
##    i = Z[n][1]
##    print(n,[C[k][i] for k in h])
##
##    
