from utils import *
from qcommands import *

# from literature [Lowry2016; Broach2012]
# new slope table: also, double outer limits in case of int rounding problems:
ten_slope = 800#10**22 
slope_table = Qf32([ten_slope]*2+9*[0]+\
    [0, 0, 37, 37, 120, 120, 320, 320, 320, 320, ten_slope, ten_slope])*0.01
# Extrapolation from literature, with maximal slope penalty equal to stairs (2kph) or infinite.
# at +/-10% slope: Abad2019 gives slope limit of 10%.


# TODO in LTS:
# - see if parking is 1 or 2-way
# - what about cycle_infra = 5 (crossing)?
# LTS>1 dismounts (value=6) - RARE! - unused for now.


def LTS_formula(D):
    BA = D['bike_access']
    if BA == 2: return 5 # forbidden
    if D['passageType'] == ord('f'): return 10 # ferry (transit)
    LTS = int(_core_LTS_BikeOttawa(D))
##    if LTS==0 and D['best_infra']>0:
##        print('*',end='')
##        return 1
    if LTS == 0 and (D['best_infra'] or BA==3):
        LTS = 1 # move back up fix: walking->biking
    if D['force_dismount'] or BA==1:
        if LTS == 1: LTS = 0 # move down fix: force_dismount dominates (is rare)
        #if LTS in [2,3,4]: LTS = 6#LTS? Force dismount on LTS>1
    if LTS == -1: LTS = 3 ### HOTWIRE, TO_DO: improve osmfilter to deal with ambiguous cases
    return LTS

def _core_LTS_BikeOttawa(D):
    BI = D['best_infra']
    if BI == 4: return 1 # track
    hcl = BI==3 # lane
    ct = D['cycle_type'] # osm way -> categ.
    if ct not in [2,3,4]: return ct
    ms = D['speedLimit_kph']
    ps = ms + 10*D['streetParking'] # perceived motor speed
    la = D['numLanes']
    isResidential = D['is_residential']
    if (hcl and ps>65) or (not hcl and (la>5  or
       (la>3 and ms<=50) or ms>50)): return 4 #m8,m11,m12
    if hcl and ps<=40 and isResidential: return 1
    # ELSE: decide between 2 and 3:
    if hcl:
        if la>2: return 3 #b2,c3
        if not isResidential: return 3
        if ms>50: return 3 #b6
        return 2
    else: # no cycle lane - mixed traffic:
        if ms<=50: #or ms<=40
            #####Service is now a separate category ### if D['is_service']: return 2 # m2, m3 & m4
            if la<3 and isResidential: return 2 # m5,m9
            return 3 #m6,m7,m10 (can m6+m7 be combined in BO.js?)
    return -1
