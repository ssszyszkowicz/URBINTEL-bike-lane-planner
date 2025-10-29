from utils import *
from qcommands import *
from fileformats import *
from earth import *
from math import atan, sin, cos, pi
from geometry2 import dist
from numpy import cumsum
import cProfile, pstats, sys###

#Kamloops: park place (@327, 328, 389) is unknown
TEMP_BIKEWAYS_TYPOS ={# Kamloops:
    "OVERLANDER BRIDGE SOUTHWEST":"OVERLANDER BRIDGE",
    "DALLAS/BARNHARTVALE NATURE PARK":"DALLAS",
    "HUGH ALLEN DR":"HUGH ALLAN DR",
    "HUGH ALLEN RD":"HUGH ALLAN DR",
    "HUGH ALLAN RD":"HUGH ALLAN DR",
    "BEACH RD":"BEACH AVE"}


def load_shp_bikeways(shp_file, MAP, TYPE):
    type_dict = {}
    for k in TYPE:
        if type(k) is int:
            for v in TYPE[k]:
                type_dict[v] = k
                if v.isdigit():
                    type_dict[int(v)] = k
    print(type_dict)
    SHP = read_shp_to_dict(shp_file)
    polylines = []
    print(len(SHP['polylines']),'bike polylines, with',
          sum([len(v) for v in SHP['polylines']]),'lines.')
    indices = []
    for n, pl in enumerate(SHP['polylines']):
        indices += [n]*len(pl)
        polylines += pl
    records = SHP['records']
    MAP['bikeways'] = Geometry(polylines) # mount G3
    for k,v in [('type',TYPE['column']),
                ('streetname','STREETNAME LOCATION')]:
        for f in Qmx(v):
            if f in records.head:
                r = records[f]
                if k=='type':
                    r = [type_dict.get(x,0) for x in r] # unknown types default to infra=0
                elif k=='streetname':
                    r = [TEMP_BIKEWAYS_TYPOS.get(x,x) for x in r]
                    r = [parse_street_name(x) for x in r]
                #r = [merge([filt(y) for y in x]) for x in zip(*[records[j] for j in f])]
                MAP['bikeways'][k] = [r[i] for i in indices]
                break
    SN = MAP['bikeways'].get('streetname')
    if SN:
        named = Qbool(not sn.endswith('_?') for sn in SN)
        MAP['bikeways_named'] = Geometry(Qfilter(polylines, named))
        MAP['bikeways_unnamed'] = Geometry(Qfilter(polylines, ~named))
    

# TYPES:...        
##    TYPES = MAP['bikeways'].get('type')
##    if TYPES:
##        print('Bikeways types:')
##        #histo(TYPES)
##        t_dict = {}
##        for n,t in enumerate(TYPES):
##            if t not in t_dict: t_dict[t] = [n]
##            else: t_dict[t] += [n]
##        t_lens = [(k, len(v)) for k,v in t_dict.items()]
##        t_lens.sort(key = lambda x: x[1], reverse=True)
##        print(t_lens)
        
        
def main_shp_bikeways(GLOBAL, VARS):
    if 1: #Q-VARS
        MAP = GLOBAL['MAP']
        TRs = GLOBAL['NWK'].TRs
        COMPUTE = GLOBAL['COMPUTE']
        MAP_HASH_GRID_m = GLOBAL['ALGORITHM']['MAP_HASH_GRID_m']
        BIKEWAYS_FILE = GLOBAL['PATHS'].get('BIKEWAYS_SHP')
        #osmSN = MAP.roads.property.get('streetName')        
    if BIKEWAYS_FILE: load_shp_bikeways(BIKEWAYS_FILE, MAP, GLOBAL['BIKEWAYS_SHP'])
    BW = MAP.get('bikeways')
    if not BW: return False
    # physical consts:
    q_m = 5.0 # quantization step in meters
    r_m = 40 # search radius in meters, should be at least 2x q_m
    max_angle = 0.523598776 # 30 deg.
    lines = TRs.lines
    rx = TRs.roads.node_x[lines.data]
    ry = TRs.roads.node_y[lines.data]
    ro = lines.offsets
    rslm = TRs.seg_len_m.data
    TYPE = BW['type']
    CONF = conflate(BW,TYPE,(rx,ry,ro,rslm),TRs['oneway'],
        (COMPUTE,MAP_HASH_GRID_m), q_m, r_m, max_angle)
    TRs['fwd_shp_infra'] = CONF
    TRs['bwd_shp_infra'] = CONF
    ###GLOBAL['B'] = locals()


# angles a and b must be in [-pi/2,pi/2]
def line_angle(a,b):
    return min(abs(a-b),abs(pi+a-b),abs(-pi+a-b))

# src must be a Geometry3, dest must be TRs
def conflate(src, infra_value, dest , oneway, HGconfig, q_m, r_m, max_angle):

    print('Conflation of',len(src),'lines.')
    COMPUTE,MAP_HASH_GRID_m = HGconfig
    HG = HashGrid(COMPUTE, src.get_lrbt('m'), MAP_HASH_GRID_m)
    rx, ry, ro, rslm = dest
    HG.hash_lines('TRs', rx, ry, ro, rslm)
    infraD = {}
    src_xy = src.xy()
    src_seglen = []; src_angle = []

    print('Finding angles & lengths of src segments.')
    for nsrc, c in enumerate(src_xy):
        seglen = []
        angle = []
        for n in range(len(c)-1):
            p = c[n]
            q = c[n+1]
            seglen.append(dist(p,q))
            dx = q[0]-p[0]
            dy = q[1]-p[1]
            if dx: angle.append(atan(dy/dx))
            else: angle.append(pi/2)
        src_seglen.append(seglen)
        src_angle.append(angle)

    print('Sampling src lines.')
    src_points = []
    src_segment = []
    for nsrc, c in enumerate(src_xy):
        seglen = src_seglen[nsrc]
        q_offset = (sum(seglen)%q_m)/2
        segment = []
        qo = q_offset
        for n in range(len(c)-1):
            nq = int((seglen[n]-qo)/q_m)+1
            segment += [n]*nq
            qo = nq*q_m+qo-seglen[n]
        src_segment.append(segment)
        SLS = LineString(c)
        points = []
        for i in range(len(segment)):
            P = SLS.interpolate(i*q_m+q_offset)
            points.append((P.x,P.y))
        src_points.append(points)

    print('Hashing samples to find nearby dest.')
    dest_seg_sets = []
    dest_seg_angle = {}
    for nsrc in range(len(src_xy)):
        seg_sets = []
        for P in src_points[nsrc]:
            x,y = P
            segs = HG.find_all_in_lrbt((x-r_m,x+r_m,y-r_m,y+r_m),
                     'TRs', with_subindex = True)
            seg_sets.append(segs)
            for seg in segs:
                dest_seg_angle[seg] = None
        dest_seg_sets.append(seg_sets)

    print('Making angles and shapely objects of found dest segments.')
    dest_seg_LS = {}
    dest_seg_origin_P = {}
    for seg in dest_seg_angle:
        tr = int(seg//(2**32))
        s = int(seg%(2**32))
        ix = ro[tr]+s
        x = rx[ix]
        y = ry[ix]
        x_ = rx[ix+1]
        y_ = ry[ix+1]
        dx = x_-x
        dy = y_-y
        if not dx: dest_seg_angle[seg] = pi/2
        else: dest_seg_angle[seg] = atan(dy/dx)
        dest_seg_LS[seg] = LineString(((x,y),(x_,y_)))
        dest_seg_origin_P[seg] = Point(x,y)
        
    print('Finding intersections between src "antennas" and dest TRs.')
    src_intersect = []
    for nsrc, c in enumerate(src.xy()):
        seg_sets = dest_seg_sets[nsrc]
        #seglen = src_seglen[nsrc]
        angle = src_angle[nsrc]
        segment = src_segment[nsrc]
        points = src_points[nsrc]
        intersect = []
        for i,P in enumerate(points):
            s = segment[i]
            an = angle[s]
            v = r_m*sin(an)
            w = r_m*cos(an)
            x,y = P
            A1 = LineString(((x-v,y+w),(x+v,y-w))) #"antenna"
            a1 = {}
            for seg in seg_sets[i]:
                if line_angle(dest_seg_angle[seg],an)<max_angle:
                    I1 = dest_seg_LS[seg].intersection(A1)
                    if I1.geom_type == 'Point': a1[seg] = I1
            # format is:(distance, 64-bit 2-part seg, intersection Point object)
            Pt = Point(x,y)
            X1 = [(I.distance(Pt),seg,I) for seg, I in a1.items()]
            X1.sort() # get closest intersections
            if not X1: intersect.append([])
            elif len(X1)==1: intersect.append(X1)
            else:
                # if both TRs are oneway, keep them both, else only keep the closest one:
                tr1 = int(X1[0][1]//(2**32))
                tr2 = int(X1[1][1]//(2**32))
                if oneway[tr1] and oneway[tr2]:
                    intersect.append(X1[0:2])
                else:
                    intersect.append([X1[0]])
        src_intersect.append(intersect)
        
    print('Updating TRs with best infra values.') 
    for nsrc, c in enumerate(src.xy()):
        intersect = src_intersect[nsrc]
        IV = infra_value[nsrc]
        points = src_points[nsrc]
        for i,P in enumerate(points):
            for Y in intersect[i]:
                seg = Y[1]
                tr = int(seg//(2**32))
                s = int(seg%(2**32))
                if tr not in infraD:
                    cs = cumsum(rslm[(ro[tr]-tr):(ro[tr+1]-tr-1)])
                    cs = [0]+cs.tolist()
                    ln = int(cs[-1]//q_m)+1
                    infraD[tr] = (cs, np_zeros(ln,dtype='uint16'))
                ix = int((dest_seg_origin_P[seg].distance(Y[2])+\
                          infraD[tr][0][s])//q_m)
                vi = infraD[tr][1]
                vi[ix] = max(IV,vi[ix]) # set value to best infra
                    
    print('Counting majority conflation value for each dest line.')
    out = np_zeros(len(ro)-1,dtype='uint16')
    for tr,cs_V in infraD.items():
        V = cs_V[1]
        h = {}
        for v in V:
            if v not in h: h[v] = 1
            else: h[v] += 1
        j = [(v,k) for k,v in h.items()]
        j.sort(reverse=True)
        if j:
            if j[0][1] != 0: # zeros don't dominate: take most frequent value:
                out[tr] = j[0][1]
            else:
                if j[0][0] >= len(V)/2: # more than half the vector is zeros:
                    out[tr] = 0
                else: #take most frequent value, except zero:
                    out[tr] = j[1][1]
    return out
    
def __old__main_shp_bikeways(GLOBAL, VARS):
    if 1: #Q-VARS
        MAP = GLOBAL['MAP']
        TRs = GLOBAL['NWK'].TRs
        COMPUTE = GLOBAL['COMPUTE']
        MAP_HASH_GRID_m = GLOBAL['ALGORITHM']['MAP_HASH_GRID_m']
        BIKEWAYS = GLOBAL['PATHS'].get('BIKEWAYS_SHP')
        osmSN = MAP.roads.property.get('streetName')
    if BIKEWAYS: load_shp_bikeways(BIKEWAYS, MAP)
    #
    BW = MAP.get('bikeways')
    if not BW: return False
    # physical consts:
    q_max_m = 5.0 # quantization step in meters (maximum)
    q_N_min = 10 # minimum number of quantization points
    r_m = 40 # search radius in meters, should be at least 2x q_max_m
    # Hash TRs:
    HG = HashGrid(COMPUTE, BW.get_lrbt('m'), MAP_HASH_GRID_m)
    if 1:
        print('Conflation: is_area filter start...')
        lines = TRs.lines.filter(~TRs['is_area']) # remove is_area's with LoL.filter
        print('Conflation: is_area filter end...')
        rx = TRs.roads.node_x[lines.data]
        ry = TRs.roads.node_y[lines.data]
        ro = lines.offsets
        rslm = TRs.seg_len_m.data
        HG.hash_lines('TRs', rx, ry, ro, rslm)
    MATCHES = []
    # Name matching engine:
    if not osmSN: shpSN = None
    else:
        map_name_hash = {}
        for tr, oR in enumerate(TRs.osm_road):
            sn = osmSN[oR]
            if sn in map_name_hash: map_name_hash[sn].append(tr)
            else: map_name_hash[sn] = [tr]
        has_name = Qu8(len(BW))
        shpSN = BW.get('streetname')
        if shpSN: has_name = Qu8([not sn.endswith('_?') and sn in map_name_hash for sn in shpSN])
    # TO_DO: how to ensure oneway exists? Necessary!
    oneway = abs(TRs['oneway']) # -1 is reverse one-way.
    Q_M = []
    print('Bikeway SHP LineString:',end='')
    for bls, bikeLS in enumerate(BW.xy()):
        if bls%10==0: print('',bls,end='')
        LS = LineString(bikeLS)
        N = max(int(LS.length/q_max_m), q_N_min)
        q_m = LS.length/N
        Q_M.append(q_m)
        #print(q_m, N, LS.length)
        MATCH = []
        nm = set(map_name_hash.get(shpSN[bls])) if shpSN and has_name[bls] else None
        for n in Qr(N):
            i = LS.interpolate(q_m*(n+0.5))
            t = HG.find_one_line((i.x,i.y), 'TRs', r_m,
                values = None) #values=None gives all; failure returns None
            if t is None: t = []
            else:
                if nm and oneway[t]: # search for the other side of the one-way pair
                    lrbt = [i.x-r_m, i.x+r_m, i.y-r_m, i.y+r_m]
                    m = nm & set(HG.find_all_in_lrbt(lrbt, 'TRs', False))
                    m = [a for a in m if oneway[a] and a!=t]
                    if len(m) == 0: t = [t]
                    elif len(m) == 1: t = [m[0],t]
                    else: t = [t, HG.find_one_line((i.x,i.y), 'TRs', r_m, values=m)]
                else: t = [t]
            MATCH.append(t)
        MATCHES.append(MATCH)
    # TO_DO: It will be important here to separate LSs by TYPE: lane, track, null.
    hits = Qf32(len(TRs),0)
    for n,a in enumerate(MATCHES):
        q_m = Q_M[n]
        for b in a:
            for c in b:
                if c is not None: hits[c] += q_m # !!not ints, maybe numpy ints?
    coverage = hits/TRs.len_m
    E = {}; k = '_shp_infra'
    E['shp_infra_coverage'] = (np_clip(coverage,a_min=0.0,a_max=1.11)*10).astype('uint8') #0 - 11
    E['fwd'+k] = (coverage>0.5).astype('int8') * 4 # 4:track, for now.
    E['bwd'+k] = E['fwd'+k]
    E['equ'+k] = E['fwd'+k] == E['bwd'+k] # no directivity, === True for now.
    E['vis'+k] = E['fwd'+k] # no directivity
    TRs.extra.update(E)
    #GLOBAL['B'] = locals()
    # OLD:
##    covers = [{t:M[t]/lm[t] for t in M} for M in MATCHES]
##    extra = Qf32(len(TRs),0)
##    for c in covers:
##        for t in c: extra[t] += c[t]
##    ### Wrong, should augment, not override:
##    #TRs.extra['cycleInfra'] = (extra>0.5).astype('uint8')*2### *2 converts lanes to tracks.
