# int8(-128) - oneway: -1(reverse), 0, 1
# int8(-128) - speedLimit_kph: int8 - max=+100
# int8(-128) - numlanes: max = 4, 0 - unknown
# Binary: int8(-128) : 0,1
#  on street parking; isResidential; isService

from utils import *
from qcommands import *


def is_upgradeable(W):
    if W.get('highway') in ['service','track','steps']: return 0
    if W.get('route') == 'ferry': return 0
    if is_service(W): return 0
    if bike_access(W) == 2: return 0
    return 1


SPACE_CHAR = '$'
# This is an approximation of LTS based on higway type only.
# It can serve as an input for the more detailed LTS formulae
# LTS0  -> ped-path:dismount (may ==> LTS1 if cyclable tag)
# LTS0  -> steps:dismount+slow
# LTS5  -> forbidden for bikes, also triggered if bicycle=no
CYCLE_TYPES = """
1 cycleway
2 access
0 foot
"""
# note: way=bicycle might be special for seasonal routes (e.g. winter) - to implement later?
CYCLE_TYPES = str_to_inv_dict(CYCLE_TYPES,SPACE_CHAR)
CYCLE_HIGHWAY_TYPES = """
5 proposed planned abandoned disused
5 motorway motorway_link bus_guideway escape raceway bridleway highway
4 trunk trunk_link primary primary_link secondary secondary_link bus_stop corridor
3 tertiary tertiary_link
2 residential living_street service services
1 cycleway track
0 path pedestrian footway steps elevator platform rest_area
-1 access road unclassified
"""
CYCLE_HIGHWAY_TYPES = str_to_inv_dict(CYCLE_HIGHWAY_TYPES,SPACE_CHAR)


# 0: neutral, 1: dismount, 2: forbidden, 3: yes
def bike_access(W, fwd = True):
##    def fail(W): Qer('Unknown bike_access',W)
    # TODO: ignores fwd for now - not used in B.C. project
    if cycle_type(W) == 5: return 2
    b = W.get('bicycle')
    if b == 'dismount': return 1
    if b in ['yes','designated']: return 3
    f = W.get('foot')
    if b == 'no':
        if f in ['yes', 'designated', None]: return 1
        if f == 'no': return 2
        return 0 
    if b == 'private':
        if f == 'private': return 2
        return 1
    return 0

def cycle_type(way):
    def _road_type(way):
        if 'highway' in way:
            h = way['highway']
            if h == 'construction':
                if 'construction' in way:              
                    h = way['construction']#roads under construction are assumed finished.
                else: return 5
            if h in CYCLE_HIGHWAY_TYPES:
                return int(CYCLE_HIGHWAY_TYPES[h])
        else:
            w = None
            for t in CYCLE_TYPES:
                if t in way:
                    if w:
                        ###print('Ambiguous way type:',way)
                        return -1
                        break
                    w=t
            if w: return int(CYCLE_TYPES[w])
        ###print('Could not identify way type:',way)
        return -1
    if way.get('ferry'): return 0 # later 10 will be transit, only ferry for now
    if way.get('motor_vehicle') == 'no': return 1
    if has_cycle_track(way): return 1
    rt = _road_type(way)
    b = way.get('bicycle')
    if b==None:
        if way.get('access') in ['no','private']: rt = 5
    elif b in ['unknown', 'use_sidepath', 'lane', 'lanes']:pass
    elif b in ['yes','permissive']:
        if rt == 5:rt = 4
        if rt == 0:rt = 1
    elif b=='designated': rt = 1
    elif b=='dismount': rt = 0
    elif b=='private': rt = 5
    elif b=='dangerous': rt = 5
    elif b=='no': ### formerly rt=5 - but you can walk your bike - TO_DO:investigate further.
        if rt == 1: rt = 0
    else:
        ###print('Unknown "bicycle" tag:',b)
        return -1
    return rt


# 0 - nothing
# 1 - sharrows
# 2 - dotted:permissive
# 3 - painted lane (LTS)
# 4 - barrier:track (LTS)
# 5 - (? calming lanes?)
INFRA_TYPES = str_to_inv_dict("""0 no none share_sidewalk
1 share_lane shared shared_lane share_busway opposite_share_busway lane=pictogram shared_lane=pictogram lane=advisory
2 lane=permissive shoulder
3 yes Yes separate designated lane opposite_lane lane=exclusive
4 track opposite_track segregated
5 crossing
""")
def bike_infra(W, fwd = True):
    I = {'left':[],'right':[],'both':[]}
    bicycle = []
    for k,v in W.items():
        if k.startswith('cycleway'):
            c = set(k.split(':'))
            c.remove('cycleway')
            DIR = 'both'
            if c:
                for d in Qmx('both left right'):
                    if d in c:
                        DIR = d
                        c.remove(d)
                        break
            if not c and v!='no': I[DIR].append(v)
            else:
                for e in Qmx('oneway lane'):
                    if e in c:
                        I[DIR].append(e+'='+v)
                        c.remove(e)
                        break
            if c: return 1 #Qer('Could not parse key:\n',k,'\nin\n',W)
        if k == 'bicycle': bicycle.append(v)
        elif k.startswith('bicycle:'): bicycle.append(k[8:])
    #N = sum([bool(I[d]) for d in I])
    I['bicycle'] = bicycle
    # TODO: Left-side driving countries: change code here:
    f,b = [['left','right'][v] for v in [fwd, not fwd]]   
    CW = I.get(f)
    if not CW: CW = I['both']
    if not CW and 'oneway=no' in I[b]: CW = I[b]
    if not CW: return 0
    CW = set(CW)
    CW.discard('oneway=no')
    if len(CW) == 1: CW = CW.pop()
    elif len(CW)>2: return 1 #Qer("Too many items in cycelway:\n",W)
    else:
        q = []
        l = []
        for e in CW:
            if e.startswith('lane='): q.append(e[5:])
            else: l.append(e)
        if len(l) > 1 or len(q)>len(l): return 1#Qer("Conflicting items in cycleway:\n",I,CW)
        if len(q) == 0: CW = l[0]
        else: CW = l[0]+'='+q[0]
    out = INFRA_TYPES.get(CW)
    if out == None:
        print('Unknown bike lane type:',CW,'defaulting to infra=1.')
        print(W)
        print(bicycle)
        return 1 
    return out

##cycleway:left=lane  cycleway:right=lane  cycleway=lane
##cycleway=no
def cycle_infra(W):
    i = int(has_cycle_lane(W))
    i += 2*int(has_cycle_track(W))
    return i #0,1,2,3

def has_cycle_lane(way):
    if way.get('shoulder:access:bicycle')=='yes': return True
    for k in way:
        if k.startswith('cycleway'):
            if not way[k] == 'no' and not way[k].startswith('no_'):
                return True
    if str(way.get('bicycle')).lower().startswith('lane'):
        #print('!!!!!',way) # TO_DO - what is this!?!?!
        return True
    return False
            
def has_cycle_track(way):
    if way.get('bicycle') == 'use_sidepath': return True
    for k in way:
        if k.startswith('cycleway'):
            if way[k] in ['track','opposite_track']:
                return True
    return False































def osm_road_id(W): return int(W.get('osm_id'))
def has_name(W): return int('name' in W)
def is_area(W): return int(W.get('area')=='yes')
def __bicycle_road(W): return int('bicycle_road' in W)
def __sidewalk(W):
    for k in W:
        if 'sidewalk' in k: return 1
    return 0


    
def force_dismount(W, fwd = True):
    # TODO: ignores fwd for now - not used in B.C. project
    if W.get('bicycle') in 'no dismount'.split(): return 1
    C = W.get('bicycle:conditional')
    if C and (C=='no' or C.startswith('no ')): return 1
    return 0


def motor_allowed(W):
    return int(cycle_type(W)) not in [0,1] 






    
    
        
    
        
    
    
    

def testingA(W):
    return bool(sum([k.startswith('cycleway:') for k in W.keys()]))

def testingB(W):
    return W.get('highway') == 'track'

def testingC(W):
    return bool(sum(['bicycle' in k for k in W.keys()]))

def street_name(W):
    return W.get('name','')





def duration_s(W):
    d = W.get('duration')
    if d is None: return 0
    if d == 'PT10M': return 10*60 # Ottawa ON (x3)
    t = d.strip().split(':')
    s = 3600*int(t[0]) + 60*int(t[1])
    if len(t)==3: s+=int(t[2])
    if len(t) > 3 or not s: Qer("Bad duration:",t)
    return s
    


def oneway(W):
    cars = W.get('oneway')
    if cars is None:
        if W.get('junction') == 'roundabout': return +1
        return 0
    if cars == -1: return -1
    cars = cars.lower().strip()
    if cars in ['true','yes']: return +1
    if cars == 'no': return 0
    if cars == '-1': return -1
    return 0 #Qer("Unknown 'oneway' field:",cars)



# def bikeway_oneway(W):
#     bikes = W.get('oneway:bicycle')
#     if bikes == 'no': return 2 # 2-way bike
#     if bikes == 'yes': return 1 # 1-way bike
#     cars = W.get('oneway')
#     if cars == 'yes': return 1
#     if cars == 'no': return 2
#     return -111










######## ALL PERTINENT way TAGS AND PROBLEM VALUES:
# highway: 
# maxspeed: 
# surface:
# construction (see highway):
# cycleway*:
# access:
# foot:
# motor_vehicle:
# bicycle:
# parking:
# parking:lane*:
# shoulder:access:bicycle:
# lanes:
# oneway:bicycle:
# oneway: 
# bicycle_road: (only in Germany?)








TYPOS = make_dict(
"""pavedw paved
unpave unpaved
unpavedw unpaved
unpved unpaved
yes2 yes
yesd yes""")



def parse_osm_value(value, is_num=False, typos={}):
    s = value.lower()
    for c in ';/\\': s = s.replace(c,',')
    s = [x.strip().strip('_') for x in s.split(',')]
    if is_num:
        for n,x in enumerate(s):
            if 'mph' in x:
                try: s[n] = float(x.replace('mph','').strip())*1.609
                except: s[n] = '?'
            else:
                try: s[n] = float(x)
                except: s[n] = '?'
    else:
        if typos:
            for n,x in enumerate(s):
                r = typos.get(x)
                if r is not None: s[n] = r
    return s






################ TO_DO ################
# Other cycle-pertinent tags: seasonal?
#
#service: driveway gives 3, not 2!
#some links(1ary,2ary?) give 4, not 3.
#Check complex intersections on Woodroffe near Baseline.
#
########################################

##### converts everything to meters and seconds:
##def unit_convert(text):
##    text = text.lower().replace('\\','/')
##    #keep only letters and \
##    if 'kph' in text or 'kmph' in text or 'km/h' in text: return 1.0
##    if 'mph' in text or 'mi/h' in text: return 1.609
##    if 'mps' in text or 'm/s' in text: return 1.0/3.6
##    if 'min' in text: return 60.0
##    if 'hour' in text or 'hr' in text: return 3600.0
##    if 'second' in text or 'sec' in text: return 1.0
##    if 'km' in text: return 1000.0
##    if 'mi' in text: return 1609.0    



PASSAGE_EXCEPTIONS = str_to_dict("""
tunnel no culvert flooded
ferry no
bridge no""")
def passage(way):
    if way.get('route') == 'ferry': return ord('f')
    ans = ''
    for t in PASSAGE_EXCEPTIONS:
        g = way.get(t)
        if g and (g not in PASSAGE_EXCEPTIONS[t]): ans += t
    if not ans: return 0
    if ans not in PASSAGE_EXCEPTIONS: return 1
    return ord(ans[0])








##### BIKE SPEEDS #######
# https://github.com/Project-OSRM/osrm-backend/blob/master/profiles/bicycle.lua
clamp_surfaces = {"ice":0, "wood":10, "metal":10, "cobblestone:flattened":10, "paving_stones":10, "compacted":10,
            "cobblestone":7, "unpaved":6, "fine_gravel":10, "gravel":6,
            "pebblestone":6, "grass_paver":6, "ground":10, "dirt":8, "earth":6, "grass":6,
            "mud":3,"sand":3,"woodchips":3, "sett":9}
clamp_surfaces.update({s:100 for s in 'asphalt chipseal paved concrete concrete_lanes'.split()})
def clamp_speed(way):
    clamps = [100] # max 100kph
    # based on OSRM bicycle.lua:
    h = way.get('highway')
    if h:
        if h == 'track': clamps.append(12)
        if h == 'path': clamps.append(12)
        if h == 'steps': clamps.append(2)
        if h.startswith('parking'): clamps.append(10)
        if h == 'pier': clamps.append(6)
    if 1: #surface 
        s = way.get('surface')
        if s is not None:
            v = clamp_surfaces.get(s)
            if v is not None: clamps.append(v)
            ###else: print("!           unknown 'surface'=",s)
    return min(clamps)/3.6

def parse_profiles(text,fields):
    def say_error(l):
        print('Error parsing profiles at line:')
        print(l)
    lines = text.split('\n')
    profile = None
    out = {}
    for l in lines:
        s = l.strip()
        if s:
            s = s.replace(':',' ').split()
            s0 = s[0].lower()
            if s0 not in fields or (not profile and s0 != 'profile'):
                say_error(l)
                return False
            if s0 in ['note','notes']:
                pass
            elif not profile or s0 == 'profile':
                if len(s) > 1:
                    profile = ' '.join(s[1:])
                    out[profile] = {'profile':profile}
                else:
                    say_error(l)
                    return False
            else:
                try:
                    v = float(s[1])
                    if s0 == 'max_time': v=v*60.0
                    if s0 == 'min_time': v=v*60.0
                    if s0 == 'max_dist': v=v*1000.0
                    if s0.startswith('lts'): v=v/3.6
                    out[profile][s0] = v
                except:
                    say_error(l)
                    return False
    mandatory = {'max_time','max_dist','lts1','lts2','lts3','lts4','lts0','lts-1'}
    for p in out:
        missing = mandatory - set(out[p].keys())
        if missing:
            print('Error: profile \"'+p+'\" missing mandatory field'+'s'*(len(missing)>1)+':')
            print(', '.join(list(missing)))
            return False
    return out
            

PROFILES = """
profile: Super
max_time 30
min_time 10  
max_dist 9
LTS-1 18
LTS0 18
LTS1 18
LTS2 18
LTS3 18
LTS4 18

profile: Work
max_time 30
min_time 10
max_dist 9
LTS-1 10
LTS0 6
LTS1 18
LTS2 15
LTS3 10
LTS4 4

profile: School
notes: age 8-14
max_time: 15 (min)
max_dist: 2.5 (km)
LTS-1 0
LTS0 4
LTS1 10 (kph)
LTS2 5
LTS3 0
LTS4 0

profile: Highschool
notes: (15-18)
max_time: 30
max_dist: 6
LTS-1 12
LTS0 6
LTS1 16
LTS2 12
LTS3 6
LTS4 0

profile: University
max_time: 30
max_dist: 9
LTS-1 15
LTS0 6
LTS1 18
LTS2 15
LTS3 10
LTS4 4
"""
PROFILE_DEFAULT_UNITS = {'max_time':'min', 'max_dist':'km', 'LTS-1':'kph', 'LTS0':'kph', 'LTS1':'kph', 'LTS2':'kph', 'LTS3':'kph', 'LTS4':'kph', 'LTS5':'kph'}
PROFILE_FIELDS = 'profile max_time min_time max_dist LTS-1 LTS0 LTS1 LTS2 LTS3 LTS4 LTS5 note notes'.lower().split()
PROFILES = parse_profiles(PROFILES,PROFILE_FIELDS)
#disp(PROFILES)


def speed_map(profile):
    speeds = [6/3.6]
    for n in Qr(4): speeds.append(profile['lts'+str(n+1)])
    return speeds+[0.0]

def bike_speed(way, lts, profile):
    if lts == '?': lts = 4
    return min(speed_map(profile)[lts],clamp_speed(way))


##################################################################################
##################################################################################


SHP_TYPES = 'bikeways.shp trucks.shp streets.shp'.split()




##def is_cycling_certainly_forbidden(way):
##    try:
##        if int(cycle_type(way)) == 5: return True
##        return False
##    except: return False 


def hasHouses(way):
    types = ('residential','living_street','tertiary','tertiary_link')
    if way.get('highway') in types: return True
    if way.get('highway') == 'construction':
        if way.get('construction') in types: return True
    return False

    

def has_onstreet_parking(way, fwd=True):
    # TO_DO: make 2way
    if way.get('parking') == 'yes':return 1
    for k in way:
        if k.startswith('parking:lane'):
            if way[k] == 'no': return 0
            if way[k].startswith('no_'): return 0
            return 1
    return 0


def get_lanes(way, fwd = True):
    def parse_lanes(lanes, _fwd):
        if ';' in lanes:
            L = [s.strip() for s in lanes.split(';')]
            if len(L) != 2 or _fwd == -1: Qer('Too many lanes: '+str(L))
            return float(L[1-int(_fwd)])
        return float(lanes) #is ';' for two directions?
        #float(sum([float(n) for n in lanes.split(';')]))
    for k in ['lanes:'+['forward','backward'][1-int(fwd)], 'lanes', 'numlanes']:        
        L = way.get(k)
        if L is not None: return min(4.0, parse_lanes(L, fwd if k.endswith('lanes') else -1))
    return 2.0#default - BO.js

# maybe redo maxspeed according to getlanes() pattern?
def get_maxspeed(way, fwd = True):
    try:
        def parse_speed(speed):
            sp = speed.lower()
            if sp == 'national': sp = 40 #BO.js#TO_DO: find for Canada
            if 'mph' in sp: sp = int(sp.replace('mph','').strip())*1.609 # can crash
            return min(int(sp),100)
        ms = [way.get('maxspeed'+t) for t in ['',':forward',':backward']]
        form = ''.join(str(int(v is not None)) for v in ms)
        if form == '000': src = 'XX'
        elif form == '100': src = '00'
        elif form == '011': src = '12'
        elif form == '101': src = '02'
        elif form == '001': src = 'X2'
        elif form == '010': src = '1X'
        else: Qer(way,'\n',form,'\n','conflicting maxspeed keys!')
        S = src[1-int(fwd)]
        if S != 'X': return parse_speed(ms[int(S)])
        #else:
        h = way.get('highway')
        if h:
            if h.startswith('motorway'): return 100#BO.js
            if h.startswith('trunk'): return 60
            if h.startswith('primary'): return 80#BO.js
            if h.startswith('secondary'): return 80#BO.js
        return 50 #Default - BO.js
    except: return 50

### From OSM wiki:
####Canada
####highway=motorway - 100 km/h (some motorways have 90 to 110 km/h limit)
####highway=primary - 80 to 90 km/h
####highway=secondary - 50 to 70 km/h
####highway=tertiary - 50 km/h
####highway=residential - 40 to 50 km/h, often 30 km/h in school zones
##default_maxspeed = {'motorway':100, 'motorway_link':100,
##                    'trunk':60, 'trunk_link':60,
##                    'primary':80, 'primary_link':80,
##                    'secondary':60, 'secondary_link':60,
##                    'tertiary':50, 'tertiary_link':50,
##                    'residential':40, 'living_street':30}
##def guess_maxspeed(way):
##    ms = way.get('maxspeed')
##    try: return int(ms)
##    except:
##        hw = way.get('highway')
##        if hw in default_maxspeed: return default_maxspeed[hw]
##        if ms == 'national': return 40
##        return 50

def draw_speed(way):
    profile = PROFILES['Work']
    lts = LTS_BikeOttawa(way)
    return (bike_speed(way, lts, profile)-0.001)/PROFILES['Work']['lts1']


def draw_highway_type(way):
    ct = cycle_type(way)
    if ct == '?': return 2
    if ct<0: return (-ct)/8.00001
    if ct == 1: return 0
    return -1


def drawBO_LTS(way):
    lts = LTS_BikeOttawa(way)
    if False and lts == 2: # 2->1 ?
        wayL = way.copy()
        wayL['cycleway'] = 'yes'
        if LTS_BikeOttawa(wayL) == 1:
            lts = 21
    colour = {0:0, 1:0.25, 2:0.5, 3:0.875, 4:1, 5:-1,'?':2}
    return colour.get(lts,-1)



def HamiltonProjectFilter(way):
    shp = way.get('.shp')
    if shp == 'bikeways':
        t = way.get('TYPE')
        BC = {'PMURT':21,'BL':22,'HCOS':23,'MCOS':23,'LCOS':23,'SBR':23,'HSBR':23}
        return BC.get(t)
    if shp == 'trucks':
        adt = float(way.get('VOLUME_24H'))
        if adt<=1500: return 11
        if adt<=3000: return 12
        if adt<=8000: return 13
        return 14
    return None
##-1 HCOS: High Auto Volume Connection Route
##-2 SBR: Signed Bike Route
##-3 PS: Paved shoulder (no LTS imporvement)
##-4 LCOS: Low Auto Volume Connection Route
##-5 BL: Bicycle Lane (may improve LTS rating to 3 or 2 from higher)
##-6 PMURT: Paved Multi-Use (Trail?) (LTS = 1)
##-7 MCOS: Medium Auto Volume Connection Route
##-8 HSBR (High-volume) Signed Bike Route


    
def is_service(way):
    S = ['alley','parking_aisle','driveway']
    # construction?
    if way.get('highway') == 'construction':
        if way.get('construction') == 'service':
            if way.get('service') in S:
                return 1
    if way.get('highway') == 'service':
        if way.get('service') in S:
            return 1
    return 0

def is_residential(way):
    T = ['residential','living_streets']
    if way.get('highway') in T: return 1
    if way.get('highway') == 'construction':
        if way.get('construction') in T: return 1
    return 0
                
def LTS_BikeOttawa(way, overrides = {}):
    CI = overrides.get('cycleInfra',0)
    if CI > 1: return 1 # imported track
    hcl = (CI == 1) or has_cycle_lane(way)
##    hpf = HamiltonProjectFilter(way)
##    if hpf is not None: return hpf
    ct = cycle_type(way)
    if ct not in [2,3,4]: return ct
    ms = get_maxspeed(way)
    osp = has_onstreet_parking(way)
    ps = ms + 10*osp # perceived motor speed
    la = get_lanes(way)
    isResidential = is_residential(way)
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
            if is_service(way): return 2 # m2, m3 & m4
            if la<3 and isResidential: return 2 # m5,m9
            return 3 #m6,m7,m10 (can m6+m7 be combined in BO.js?)
    return -1

def __LTS_BikeOttawa___ORIGINAL___(way):
    ct = cycle_type(way)
    if ct not in [2,3,4]: return ct
    ms = get_maxspeed(way)
    la = get_lanes(way)
    isResidential = way.get('highway') in ['residential','living_streets']
    width = 99999.9 #not implemented
    hasSeparatingMedian = False # not implemented
    if has_cycle_lane(way):
        if has_onstreet_parking(way):
            if ms>55: return 4 #b8
            if ms<=40 and isResidential and la<=2: return 1
            lts = 1
            if la>=3: lts=max(lts,3) #b2
            if width<=4.1: lts=max(lts,3) #b3
            elif width<=4.25: lts=max(lts,2) #b4
            elif width<=4.5 and (ms<40 or isResidential): lts=max(lts,2) #b5
            if ms>40:
                if ms<=50: lts=max(lts,2) #b6
                elif ms<55: lts=max(lts,3) #b7 (note: BO.js code erroneously has <65!!!) 
            if not isResidential: lts=max(lts,3)
            return lts
        else: # no onstreet parking (for most rules, simply relaxes maxspeed effect by 10kph)
            if ms>65: return 4 #c6
            if ms<=50 and isResidential and la<=2: return 1
            lts = 1
            if la == 3 and hasSeparatingMedian: lts=max(lts,2) #c2
            elif la >= 3: lts=max(lts,3) #c3
            if width<=1.7: lts=max(lts,2) #c4
            if ms>50 and ms<65: lts=max(lts,3) #c5
            if not isResidential: lts=max(lts,3) #c7
            return lts
    else: # no cycle lane - mixed traffic:
        if (la>5 and ms<=40) or (la>3 and ms<=50) or ms>50: return 4 #m8,m11,m12
        if way.get('highway') == 'service':
            if ms<=50 and way.get('service') in ['alley','parking_aisle','driveway']: return 2 # m2, m3 & m4
        if ms<=50: #or ms<=40
            if la<3 and isResidential: return 2 # m5,m9
            return 3 #m6,m7,m10 (can m6+m7 be combined in BO.js?)
    return '?'



def __NOTUSED__highway_mode(W):
    BI = 2
    WK = 1
    CN = 0.75
    NO = [None, 'no']
    if 'lcn' in W: return CN
    if 'rcn' in W: return CN
    if 'ncn' in W: return CN
    if W['highway'] == 'cycleway': return BI # cycle mode
    bicycle = W.get('bicycle')
    if bicycle in "yes designated".split(): return BI
    if bicycle in "permissive dismount".split(): return WK
    if W.get('cycleway') not in NO: return BI
    if W.get('cycleway:buffer') not in NO: return BI
    if W.get('cycleway:left') not in NO: return BI
    if W.get('cycleway:right') not in NO: return BI
    if W.get('cycleway:seasonal') not in NO: return BI
    if 'oneway:bicycle' in W: return BI
    if W.get('shoulder:access:bicycle') == 'yes': return BI
    # paved_shoulder?
    if W['highway'] in "footway path pedestrian".split(): return WK # ped mode
    if W.get('foot') not in NO: return WK
    if W.get('footway') not in NO: return WK
    if W.get('path') not in NO+['desire']: return WK
    return -111 # car mode

def highway_lit(W):
    val = W.get('lit')
    if val == None: return -111
    if val == "yes": return 2
    if val == "no": return 1
    
def bikeway_oneway(W):
    bikes = W.get('oneway:bicycle')
    if bikes == 'no': return 2 # 2-way bike
    if bikes == 'yes': return 1 # 1-way bike
    cars = W.get('oneway')
    if cars == 'yes': return 1
    if cars == 'no': return 2
    return -111


def highway_layered(W):
    if 'layer' in W: return 100#+int(W['layer'])
    return -1







##def LTS(way, formula):
##    call = {'R5simple':LTS_R5simple,
##            'BikeOttawa':LTS_BikeOttawa}
##    if formula in call:
##        return call[formula](way)
##    else:
##        print('Unknown LTS formula:',formula)
##        return False



##def LTS_R5simple(way):
##    if has_cycle_track(way): return 1
##    ct = cycle_type(way)
##    ms = get_maxspeed(way)
##    if ms and ms<41: under41 = True
##    else: under41 = False
##    if ct == '?': return '?'
##    # larger than 3ary:
##    if ct == 4: return ct - int(has_cycle_lane(way))
##    # 3ary:
##    if ct == 3:
##        if has_cycle_lane(way): return 2
##        if get_lanes(way) == None and under41: return 2
##        return 3
##    if ct == 2:
##        pass # TO_DO ...
##     # TO_DO ...
##    # service ==> LTS2
##    return ct

##    Source: blog.conveyal.com/better-measures-of-bike-accessibility-d875ae5ed831
##    Does not allow cars: LTS 1
##    Is a service road: Unknown LTS
##    Is residential or living street: LTS 1 - should be 3?
##    Has 3 or fewer lanes and max speed 25 mph or less: LTS 2
##    Has 3 or fewer lanes and unknown max speed: LTS 2
##    Is tertiary or smaller road:
##     Has unknown lanes and max speed 25 mph or less: LTS 2
##     Has bike lane: LTS 2
##     Otherwise: LTS 3
##    Is larger than tertiary road
##     Has bike lane: LTS 3
##     Otherwise: LTS 4
##      25mph = 40.225kph.

    


#TO_DO: Implement BO_LTS, at least in part.
##try: import BikeOttawa_LTS_stressmodel
##except:
##    import js2py
##    js2py.translate_file('BikeOttawa_LTS_stressmodel.js',
##                         'BikeOttawa_LTS_stressmodel.py')
##    import BikeOttawa_LTS_stressmodel
##
##def LTS_BikeOttawa(way_tags):
##    return BikeOttawa_LTS_stressmodel.PyJsHoisted_evaluateLTS_({'tags':way_tags})
##    
##print(LTS_BikeOttawa({'highway': 'residential', 'maxspeed':'40', 'lanes':'2'}))

############ OSM TAG CATEGORISATION ######################


        ## Level tags in Ottawa (? find also height tags)
        ##building:levels:underground ---> {'1'}
        ##Levels ---> {'2', '3'}
        ##level ---> {'0', '0,1', '1', '0;1', '2', '3', '-1', '6'}
        ##building:levels ---> {'12.5', '1', '4.5', '22', '2', '28', '20', '23', '17', '45', '16', '1.5', '4', '33', '7', '11', '3.5', '19', '5.5', '10', '29', '0', '2.5', '24', '12', '3', '25', '27', '30', '26', '18', '21', '11.5', '8', '15', '9', '14', '5', '6', '13.5', '13'}
        ##isced:level ---> {'1', '2', '3', '2-3'}
        ##building:level ---> {'1'}
        ##min_level ---> {'0'}
        ##roof:levels ---> {'0', '2', '1'}
        ##max_level ---> {'2'}
def get_building_levels(building):
    if not 'building:levels' in building: return 1.0   
    try: return float(building['building:levels'])
    except:
        try: return min(float(s) for s in re_findall(r"[-+]?\d*\.\d+|\d+", building['building:levels']))
        except:
            # This happens 0 times in Ottawa, once in Toronto ('building:levels': 'steak, seafood' - :) )
            print('Cannot parse building:levels in:',building)
            return 1.0




# Misspelled: Ottawa:detatched(R)
BUILDING_TYPES = """
R nursing_home townhouse retirement_home houseboat shelter_type boathouse shelter mcmansion apartments dormitory residential terrace house detached detatched semi-detached static_caravan cabin bungalow block semidetached_house
W fire_station research_centre townhall factory research_institute government industrial farm office farm_auxiliary radio_station manufacture
C fuel funeral_hall vehicle_inspection pub cinema pharmacy bank mall car_wash post_office marketplace diner food_court cafe fast_food bar studio casino nightclub pottery_studio car_rental concession_stand crematorium delivery cuisine club brand restaurant shop retail commercial supermarket art_and_craft_store kiosk
P dentist museum doctors arts_centre sports clinic courthouse theatre public_building recreational music_school stadium library events_venue animal_shelter culture_center police veterinary conference_centre toilets childcare community_centre tourism takeaway sport social_facility social_facility:for leisure healthcare:speciality healthcare public_library gallery sports_centre recreation_centre gym public school university civic hospital hotel college kindergarten social_facility prison social_centre motel train_station
S post_depot bicycle_parking gatehouse recycling waste_transfer_station aeroway storage utility railway_station hangar warehouse shed service barn parking greenhouse garage bleachers bunker hut stable storage_tank pavilion transportation construction garages gazebo silo
0 shrine place_of_worship parish_hall monastery denomination church convent mosque cathedral roof military embassy temple canopy synagogue chapel
? carport porch patio bandstand tower riding_hall user$defined ter3 grandstand religious
X no vacant portable abandoned collapsed ruins construction recreation_ground quarry reservoir allotments landfill churchyard railway garages #what-to-do-with-these?
"""
BUILDING_TYPES = str_to_inv_dict(BUILDING_TYPES, SPACE_CHAR)
def building_type(Bway):
    bt = Bway.get('building',Bway.get('building:part','no')).lower()
    if bt == 'no': return 'X'
    if bt == 'yes':
        am = Bway.get('amenity')
        if am: bt = am
        else:
            ks = set(BUILDING_TYPES.get(k) for k in Bway)
            ks.discard(None)
            if len(ks)==1: return ks.pop()
            if len(ks)==0: return 'R' # sic:insufficient info ==> Residential
            return '?'
    T = BUILDING_TYPES.get(bt)
    if T is None:
        #print('Unknown OSM building type: ', bt)
        return '?'
    return T
def building_height_unit(Bway):
    h = Bway.get('height',Bway.get('building:height',''))
    h = h.replace(' ','')
    if h:
        #parse 'm',' m',"'"(feet)
        if h[-1]=='m': return (float(h[:-1]),'m')
        if h[-1]=="'": return (0.304*float(h[:-1]),'m')
        return (float(h),'m')
    f = Bway.get('building:levels',Bway.get('building:level',''))
    # eliminate negative floors==> 0
    if f: return (min(0,float(f)),'floors')
    return (0,None)
# NEED TO ADD BUILDING TAGS TO DECODE TYPE:
# 'office': 244
# leisure 124 sport 51 club 7 historic 51 seasonal 5 emergency 14 'social_facility': 23,
# residential 3 'building:use': 8,
# 'healthcare': 9  'power': 21 'military': 4,

# height 'building:level': 'building:flats': 18 'capacity': 17,'building:height': 8,
#'bicycle': 3 'public_transport': 1 'building:part': 145, 'building:levels'




WAY_TYPES_POLYGON = set('building building:part landuse natural'.split())
# ncn:national cycling NW; lcn:local"", rcn:regional""
##WAY_TYPES = """
##building office building:part height building:level building:levels building:flats
##building:part
##""" + \
##"highway bicycle cycleway ncn lcn rcn" +\
##" maxspeed layer oneway" +\
##" surface smoothness lit condition" +\
##" foot footway path" +\
##" cycleway:buffer cycleway:left cycleway:right cycleway:seasonal" +\
##" oneway:bicycle shoulder:access:bicycle paved_shoulder" +\
##"\n" + """
##cycleway
##access
##bicycle
##foot
##oneway
##landuse
##natural
##"""
##WAY_TYPES = str_to_dict(WAY_TYPES,SPACE_CHAR)
WAY_TYPES = 'highway cycleway access bicycle foot'.split()


##CYCLEWAY_TYPES = """
##X no
##Y yes lane
##T track
##S share_busway shared_lane
##O opposite_lane
##+ crossing
##"""
##CYCLEWAY_TYPES=str_to_inv_dict(CYCLEWAY_TYPES,SPACE_CHAR)


##def CycleStreetsTable():
##    print('WARNING - CycleStreetsTable may malfunction due to SPACE_CHAR - TO_DO - test!')
##    import csv, re
##    BIKE_HIGHWAY_TABLE = {}
##    FOR_PRINTING = []
##    with open("data/osmHighwayTranslation from CycleStreets.csv", 'r') as f:
##        for e in list(csv.reader(f))[1:]:
##            vals = [int(n) for n in re.findall(r'\d+',e[9])]
##            vals[0] *= int(e[9][0:3] == 'yes')
##            #FOR_PRINTING += '\t'.join(str(v) for v in [e[3],e[4], vals[0]*vals[1],vals[0]*vals[2],vals[3]])
##            BIKE_HIGHWAY_TABLE[e[3]+SPACE_CHAR+e[4]] = vals
##    #disp(BIKE_HIGHWAY_TABLE,1000000)
##    CycleStreetsBikeSafetyIndicator = {k: v[0]*v[1] for k,v in BIKE_HIGHWAY_TABLE.items() if len(v)==4}
##    #disp(FOR_PRINTING,10000000)
##    return BIKE_HIGHWAY_TABLE
##T = CycleStreetsTable()
###for r in T:
###    print(r.replace(SPACE_CHAR,',')+','+','.join([str(v) for v in T[r]]))



