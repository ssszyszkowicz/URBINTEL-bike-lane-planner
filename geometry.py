# This is old code - imports are not in line with new style.
### !!! Imports not cleaned here!


# fractum fortior recrescit
import Polygon, random, time
from Polygon.Utils import fillHoles, prunePoints, reducePointsDP, convexHull
from shapely.geometry import LinearRing, LineString, MultiPolygon, Point, MultiPoint
from shapely.geometry import Polygon as sPolygon
from shapely.ops import cascaded_union
# shapely.speedups makes things crash :(

from math import hypot, ceil, pi, cos, sin, atan2
from numpy import argmin, argmax, cumsum, diff, nonzero

# ! When using exterior.coords, always add [0:-1] to remove last redundant point.

######################
# NEW/IN DEVELOPMENT #
######################
def all_intersect(contours):
    if not contours: return []
    I = Polygon.Polygon(contours[0])
    for c in contours[1:]:
        I &= Polygon.Polygon(c)
        print(I.area())
    return {'contours':[c for k,c in enumerate(I) if I.isSolid(k)],
            'holes':[c for k,c in enumerate(I) if not I.isSolid(k)]}
            


    
##    # Implementation with Polygon.py sometimes causes Python to restart!
##    HH = Polygon.Polygon()
##    for h in holes: HH.addContour(h)
##    CC = Polygon.Polygon()
##    for c in contours: CC.addContour(c)
##    II = CC - HH
##    return [c for k,c in enumerate(II) if II.isSolid(k)]

#####################################
# OBJECTS TOUCHING/DISTANCE QUERIES #
#####################################
# Objects: Filled contours, 2-point lines, or Points
def pointlists_to_objects(pointlists):
    obj = []
    for pl in pointlists:
        n = len(pl)
        if n >= 3: obj.append(sPolygon(pl))
        elif n == 2: obj.append(LineString(pl))
        elif n == 1: obj.append(Point(pl[0]))
        else: obj.append(None)
    return obj

# Return lower-tri list of lists 
def mutual_distances(pointlists):
    if len(pointlists) == 1: return []
    if len(pointlists) == 0: return None
    OBJ = pointlists_to_objects(pointlists)
    D = []
    for n in range(len(OBJ)):
        row = []
        for m in range(n):
            row.append(OBJ[n].distance(OBJ[m]))
        D.append(row)
    return D

# Return rectangular lists of lists of distances 
def two_sets_distances(pointlists1, pointlists2):
    O1 = pointlists_to_objects(pointlists1)
    O2 = pointlists_to_objects(pointlists2)
    N1,N2 = len(O1),len(O2)
    DD = [[0.0]*N2 for n in range(N1)]
    for n1 in range(N1):
        for n2 in range(N2):
            DD[n1][n2] = O1[n1].distance(O2[n2])
    return DD
    

# Returns the list of pointlists that touch (have distance <eps)
# Returns None if pointlists is empty
# Else, returns [] - which should happen for islands and such
# Overlapping islands should be merged properly with the rest of the environment
def mutual_touching(pointlists, eps = 1e-10):
    if not pointlists: return None
    N = len(pointlists)
    touching = set()
    DD = mutual_distances(pointlists)
    for n in range(N):
        for m in range(n):
            if DD[n][m]<eps:
                touching.add(n)
                touching.add(m)
    return sorted(list(touching))

# Returns a list of N1 lists of objects in pl2 that touch object in pl1
def two_sets_touching(pointlists1, pointlists2, eps = 1e-10):
    DD = two_sets_distances(pointlists1, pointlists2)
    return [[n for n,d in enumerate(D) if d<eps] for D in DD] 

# Return indices of all bridges that connects side1 to side2
# inputs are pointlists
def find_bridges(bridges, side1, side2, eps = 1e-10):
    DD = two_sets_distances(bridges, [side1,side2])
    return [n for n,D in enumerate(DD) if D[0]<eps and D[1]<eps] 




def contours_linking_contours(A,B,contours, eps = 1e-6):
    linking = []
    for n, C in enumerate(contours):
        print(len(A),len(B),len(C))
        DD = contour_distances([A,B,C])
        if DD[2][0] < eps and DD[1][2] < eps: linking.append(n)
    return linking



##devanet_orientation(C) == 'ccw': CC[n] = C
##        else:
##            print('Error, convex_space() only takes ccw contours!')
##            return False
##    if mutual_touching(CC): return False
##    CH = ensure_orientation(convex_hull(sum(CC,[])),'ccw')
##    CH_Pts = [[p] for p in CH]
##    on_CH = [s[0] for s in two_sets_touching(CH_Pts,CC)]
##    print(on_CH)
##    N_CH = len(on_CH)
##    d = list(diff(on_CH+[on_CH[0]]))
##    print(d)
##    down = d.index(-1)%N_CH
##    up = d.index(1)%N_CH
##    on_CH = [s[0] for s in two_sets_touching(CH_Pts,CC)]
##    print(on_CH)
##    print(up,down)
##    Touch = [0,0]
##    Touch[0] = [[CH[up]],[CH[(down+1)%N_CH]]]
##    Touch[1] = [[CH[down]],[CH[(up+1)%N_CH]]]
##    for n in [0,1]:

##    on_CH = [s[0] for s in two_sets_touching(CH_Pts,CC)]
##    print(on_CH)
##        print(two_sets_touching(Touch[n]))









##def convex_bridge(contour1, contour2):
##    CC = [0,0]
##    for n,C in enumerate([contour1,contour2]): 
##        if get_orientation(C) == 'ccw': CC[n] = C
##        else:
##            print('Error, convex_space() only takes ccw contours!')
##            return False
##    if mutual_touching(CC): return False
##    CH = ensure_orientation(convex_hull(sum(CC,[])),'ccw')
##    CH_Pts = [[p] for p in CH]
##    on_CH = [s[0] for s in two_sets_touching(CH_Pts,CC)]
##    print(on_CH)
##    N_CH = len(on_CH)
##    d = list(diff(on_CH+[on_CH[0]]))
##    print(d)
##    down = d.index(-1)%N_CH
##    up = d.index(1)%N_CH
##    print(up,down)
##    Touch = [0,0]
##    Touch[0] = [[CH[up]],[CH[(down+1)%N_CH]]]
##    Touch[1] = [[CH[down]],[CH[(up+1)%N_CH]]]
##    for n in [0,1]:
##        print(two_sets_touching(Touch[n]))

# returns [[A,B],[C,D]] such that A,B are indices on contour1
# C,D are indices on contour2, and A-C and B-D are links of the bridge
# ACDB should be CCW
def convex_bridge(contour1, contour2):
    CC = [contour1,contour2]
    CH = ensure_orientation(convex_hull(CC[0]+CC[1]),'ccw')
    CH_Pts = [[p] for p in CH]
    on_CH = [s[0] for s in two_sets_touching(CH_Pts,CC)]
#    print(on_CH)
    N_CH = len(on_CH)
    d = list(diff(on_CH+[on_CH[0]]))
    down = d.index(-1)%N_CH
    up = d.index(1)%N_CH
#    print(up,down)
    Touch = [[[CH[up]],[CH[(down+1)%N_CH]]], [[CH[(up+1)%N_CH]],[CH[down]]]]
    return [[k[0] for k in two_sets_touching(Touch[n],[[P] for P in CC[n]])] for n in [0,1]]

    



##def convex_space(contour1, contour2):
##    CC = [Polygon.Polygon(C) for C in [contour1,contour2]]
##    CH = Polygon.Polygon(convex_hull(contour1+contour2))
##    DIFF = CH - (CC[0] + CC[1])
##    CAND = [C for k,C in enumerate(DIFF) if DIFF.isSolid(k)]
##    BRIDGES = find_bridges(CAND,contour1,contour2)
##    if len(BRIDGES) != 1:
##        print('Error in convex_space()!!!')
##        return False
##    return CAND[BRIDGES[0]]




# is {contours} - {holes} without any holes.
def solid_difference(holes, contours, eps = 0.001):
    HH = cascaded_union([sPolygon(h) for h in holes])
    CC = cascaded_union([sPolygon(c) for c in contours])
    DD = CC.difference(HH)
    print(DD.geom_type)
    if DD.geom_type == 'Polygon': DD = [DD]
    return [tuple(D.exterior.coords)[0:-1] for D in DD]
    
    
##    # Implementation with Polygon.py sometimes causes Python to restart!
##    HH = Polygon.Polygon()
##    for h in holes: HH.addContour(h)
##    CC = Polygon.Polygon()
##    for c in contours: CC.addContour(c)
##    II = CC - HH
##    return [c for k,c in enumerate(II) if II.isSolid(k)]



# This takes forever and a day
# Based on visibility from a large number of points around the contours:
# Returns a list for each contour of the indices of visible contours. 
##def contours_mutual_visibility(contours):
##    if mutual_touching(contours): return False
##    l,r,b,t = bounding_box_lrbt([item for sublist in contours for item in sublist])
##    boundary = ensure_frame_is_contour([l-1,r+1,b-1,t+1])
##    RN = range(len(contours))
##    for n in RN:
##        points = dilate_and_quantize(contours[n], dilate=0.001, quant=1.0)
##        print(len(points))
##        for p in points:
##            print(len(visibility(p,boundary,contours)))
##        #FOVs = [visibility(p,boundary,contours) for p in points]
##        touch = what_touches(FOV, contours, items = 'contours')
##        print(touch)
##        input('...')
            



def iterated_convex_merge(contours,iter=None):
    if not contours: return []
    convex = [convex_hull(c) for c in contours if c]
    count = 0
    while True:
        count += 1
        merged = merge(convex)['islands']
        change = len(convex) - len(merged)
        print('merged %d convexes'%change)
        if not iter and not change: break
        if count == iter: break
        convex = [convex_hull(c) for c in merged]
    return merged # should return a lits of contours, all convex and disjoint





#MV matrix (lower triangular) takes values:
#    True: visible for sure,
#    False: not visible for sure
#    None: not yet determined
def contours_mutual_visibility(contours, nearest_neighbours = []):
    def PPVrays(A, B, Others, debug=False):
        # Test all corner-to-corner rays for mutual visibility:
        CH = LinearRing(convex_hull(A+B))
        POs = [ ]
        for O in Others:
            P = sPolygon(O)
            if P.intersects(CH): POs.append(P)
        for a in A:
            for b in B:
                LS = LineString((a,b))
                clear = True
                for PO in POs:
                    if LS.intersects(PO): clear = False
                if clear: return True # found visibility line
        return False# strictly, should be None
    def PPV(A, B, Others, debug=False): # Contour-to-Contour Visibility
        BRIDGE = convex_bridge(A,B)
        LINKS = ((A[BRIDGE[0][1]],B[BRIDGE[1][1]]),
                 (B[BRIDGE[1][0]],A[BRIDGE[0][0]]))
        INNER = [[],[]]
        for n in [0,1]:
            C = [A,B][n]
            N = len(C)
            s = BRIDGE[n][n]
            e = BRIDGE[n][1-n]
            if s<e: R = range(s,e+1)
            else: R = list(range(s,N))+list(range(e+1))
            INNER[n] = [C[k] for k in R]
        AREA = INNER[0]+INNER[1]
        OBSTACLES = contours_in_contours(Others,[AREA])
        NO = len(OBSTACLES)
        TST = [set(t) for t in two_sets_touching(LINKS,OBSTACLES,eps=0.01)]
        if not TST[0] or not TST[1]: return True # visibility by LINKS
        if TST[0] & TST[1]: return False # obstacle blocks all visibility
        OBS_SIDE = [[],[]]
        for s in [0,1]:
            SO  = sum([OBSTACLES[n] for n in TST[s]],[])
            OBS_SIDE = [convex_hull(SO) if SO else []]
        OBS_IN = [OBSTACLES[n] for n in (set(range(NO))-TST[0]-TST[1])]
        if debug: print(len(OBS_IN),len(OBS_SIDE))
        NEW_OBSTACLES = iterated_convex_merge(OBS_SIDE+OBS_IN, 2)
        TST = [set(t) for t in two_sets_touching(LINKS,NEW_OBSTACLES)]
        if TST[0] & TST[1]: return False
        # Test all corner-to-corner rays for mutual visibility:
        POs = [sPolygon(O) for O in NEW_OBSTACLES]
        N1, N2 = len(INNER[0]),len(INNER[1])
        for n in range(N1):
            for m in range(N2):
                if (n==0 and m==0) or (n==N1 and m==N2): pass
                else:
                    LS = LineString((INNER[0][n],INNER[1][m]))
                    clear = True
                    for PO in POs:
                        if LS.intersects(PO): clear = False
                    if clear: return True # found visibility line
        return None # Visibility is uncertain: should continue the algorithm here
        # to see if there exists a visibility line connecting A and B 
    # clean contours: make sure they are CCW and non-touching
    #if mutual_touching(contours): return False
    contours = [ensure_orientation(C,'ccw') for C in contours]
    RN = range(len(contours))
    MV = [[None for m in range(n)] for n in RN]
    for n in RN:
        print(n)
        for m in range(n):
            if nearest_neighbours and m in nearest_neighbours[n]:
                VIS = True
            else:
                VIS = PPVrays(contours[n],contours[m],
                    [C for k,C in enumerate(contours) if k!=m and k!=n],
                    (n==39 and m == 14))
            print(n,m,VIS)
            MV[n][m] = VIS
    # convert format to list of (possibly) visible island indices:
    sure = [[] for n in RN]
    maybe = [[] for n in RN]
    for n in RN:
        for m in range(n):
            if MV[n][m] is True:
                sure[m].append(n)
                sure[n].append(m)
            if MV[n][m] is None:
                maybe[m].append(n)
                maybe[n].append(m)
    return {'IMVsure':sure,'IMVmaybe':maybe}
    






###################
# CONTOUR TOOLBOX # 
###################

def contours_in_contours(contours, boundaries):
    CC = Polygon.Polygon()
    for c in contours: CC.addContour(c)
    BB = Polygon.Polygon()
    for b in boundaries: BB.addContour(b)
    II = CC & BB
    return [c for k,c in enumerate(II) if II.isSolid(k)]
    
    
def remove_flat_angles(contour, tol_deg=0.01):
    if get_orientation(contour) is False:
        return False
    tol = cos(pi/2-(tol_deg*pi/180.0)) - 1
    N = len(contour)
    keep = [True]*N
    for n in range(N):
        if cos_abc(contour[n-1],contour[n],contour[(n+1)%N]) < tol:
            keep[n] = False
    return [contour[n] for n in range(N) if keep[n]]

# returns [] if polygon is convex :)
def get_concave_corners(contour): 
    N = len(contour)
    orientation = get_orientation(contour)
    if orientation == 'ccw': rot = -1
    elif orientation == 'cw': rot = 1
    else: return False
    concave_corners = []
    for n in range(N):
        if rot * sin_abc(contour[n-1],contour[n],contour[(n+1)%N]) < 0.0:
            concave_corners.append(n)
    return concave_corners

def convex_hull(pointlist):
    if len(pointlist) < 2: return pointlist
    CH = MultiPoint(pointlist).convex_hull
    if CH.geom_type == 'LineString': return tuple(CH.coords)
    return tuple(CH.exterior.coords)[0:-1]

# Must erode contour so as to produce another contour.
# If this fails, return False
def erode_slightly(contour, eps):
    CP = sPolygon(contour)
    ER = CP.buffer(distance=-eps, resolution=1, cap_style = 3, join_style=2, mitre_limit=2.0)
    if ER.geom_type != 'Polygon': return False
    if not ER.exterior: return False
    return list(ER.exterior.coords)[0:-1]
### TEST:
### U-shape, should break into two for eps>=0.5 and disappear for eps>=1
##C = [(0,0),(0,5),(2,5),(2,1),(3,1),(3,5),(5,5),(5,0)]
##print(erode_slightly(C, 0.1))

def get_orientation(contour):
    P = Polygon.Polygon(contour)
    O = P.orientation(0)
    if O == 1: return 'ccw'
    if O == -1: return 'cw'
    print('Invalid contour passed to geometry.get_orientation():')
    print(contour)
    return False

def ensure_orientation(contour, orientation = 'ccw'):
    O = orientation.lower()
    C = get_orientation(contour) 
    valid = {'ccw','cw'}
    if O in valid and C in valid:
        if O == C: return contour
        else: return contour[::-1]
    else:
        print('Invalid orientations in geometry.ensure_orientation():')
        print('O: ', O)
        print('C: ', C)
        return False

def outer_normals(contour):
    O = get_orientation(contour)
    if O == 'ccw': rot = -pi/2
    elif O == 'cw': rot = pi/2
    else: return False
    N = len(contour)
    normals = []
    for n in range(N):
        vec = vec_diff(contour[(n+1)%N],contour[n])
        angle = atan2(vec[1],vec[0])#note order: y,x
        normals.append([cos(angle+rot),sin(angle+rot)])
    return normals

def bounding_box_lrbt(points): # e.g., contour
    X,Y = zip(*points)
    return [min(X),max(X),min(Y),max(Y)]


def count_points(contours):
    return sum([len(c) for c in contours])

def get_area(contour):
    return 0 if len(contour) < 3 else sPolygon(contour).area 

def get_union_area(contours):
    return cascaded_union([sPolygon(c) for c in contours]).area

def overlap_area(c1, c2):
    if len(c1)<3 or len(c2)<3: return 0
    P1 = sPolygon(c1)
    P2 = sPolygon(c2)
    return P1.intersection(P2).area

def widest(contour):
    R = range(len(contour))
    return max(max((dist(contour[n],contour[m]) for n in R if n<m)) for m in R if m>0)

def merge_and_fill(contours,engine='Polygon'):
    if len(contours) < 2: return contours
    if engine == 'shapely':
        print('geometry.merge_and_fill() not yet implemented with shapely engine')
        return False
        CU = cascaded_union([sPolygon(c) for c in contours])
        if CU.geom_type == 'Polygon': PP = [CU]
        else:
            assert CU.geom_type == 'MultiPolygon'
            PP = [P for P in CU]
        # not finished - see merge()
    if engine == 'Polygon':    
        pp = Polygon.Polygon()
        for c in contours:
            pp += Polygon.Polygon(c)
            if not random.randint(0,19): # Fill holes about once in 20 
                pp = fillHoles(pp)
        pp = fillHoles(pp)
        pp.simplify()
        pp = prunePoints(pp)
        return [p for p in pp] # needed to iterate over Polygon object

def group_contours_by_wrappers(contours, wrappers):
    grouping = [[] for n in wrappers]
    W = [sPolygon(w) for w in wrappers]
    C = [sPolygon(c) for c in contours]
    for ic, c in enumerate(C):
        for iw, w in enumerate(W):
            if c.intersection(w).area > 0.99 * c.area:
                grouping[iw].append(ic)
    return grouping





### Dilation operations
dilation_style = {'resolution':1, 'cap_style':3, 'join_style':2, 'mitre_limit':2.0}#mitred join

def shapely_get_interiors(sObject):
    print('shapely_get_interiors() not implemented yet!')
    return False

def shapely_get_exteriors(sObject):
    if sObject.geom_type is 'Polygon': PP = [sObject]
    elif sObject.geom_type is 'MultiPolygon': PP = sObject
    else:
        print('Unimplemented shapley object type <'+sObject.geom_type+'> passed to shapely_get_exteriors()')
        return False
    return [tuple(P.exterior.coords)[0:-1] for P in PP] # exterior is a LineString, which has first and last point the same

def shapely_MultiPolygon(contours, interiors=False):
    if interiors:
        print('interiors not implemented in shapely_MultiPolygon()')
        return False    
    return MultiPolygon([sPolygon(c) for c in contours])

# Dilation+Erosion = Closing in Comp Geo
def closing(contours, radius, interiors = False):
    if interiors:
        print('interiors not implemented in closing()')
        return False
    if not contours: return []
    if False:# This doesnt work:
        MP = shapely_MultiPolygon(contours)
        DI = MP.buffer(distance=radius, **dilation_style)
        ME = cascaded_union(DI)
        ER = ME.buffer(distance=-radius, **dilation_style)
        return shapely_get_exteriors(ER)
    else: #Instead, do step by step:
        CC = dilate(contours, radius)# dilate
        CC = merge_and_fill(CC)# merge
        return dilate(CC, -radius) # erode        


def dilate_and_quantize(contour, dilate, quant):
    LS = LineString(contour+[contour[0]]) # Must be LineString to support interpolation
    if dilate: dilated = LineString(LS.buffer(dilate,1).exterior) # use default(rounded) dilation
    else: dilated = LS
    N_cand = ceil(dilated.length/quant)
    return [tuple(dilated.interpolate(n*quant).coords)[0] for n in range(N_cand)]





## Reduce number of points in a set of islands

def simplify_preserve_topology(contours, wall_tol, interiors = False):
    if interiors:
        print('interiors not implemented in simplify_preserve_topology()')
        return False
    if not contours: return []
    MP = shapely_MultiPolygon(contours)
    MP = MP.simplify(wall_tol, preserve_topology=True)
    return shapely_get_exteriors(MP)

    
    
##########################################
### DEPRECTATED merge function - now upgraded in nbh.make_islands()
##########################################
### TO IMPLEMENT: 'closing', and careful recombining potential overlaps.
##
##
##def merge(contours, wall_tol = 0.0, min_area = 0.0, frame = None): # maybe implement, closing_r = 0.0): 
##    if not contours: return {'islands':()}
##    # Merge all buildings.
##    try:
##        MP = cascaded_union([sPolygon(c) for c in contours])
##    except:
##        PP = Polygon.Polygon()
##        for c in contours: PP += Polygon.Polygon(c)
##        outsides = [sPolygon(PP[i]) for i in range(len(PP)) if PP.isSolid(i)]
##        MP = MultiPolygon(outsides)
##    # Simplify buildings:
##    MP = MP.simplify(wall_tol,preserve_topology=True)
##    # Crop by frame:
##    if frame:  MP = MP.intersection(sPolygon(ensure_frame_is_contour(frame)))
##    # If there is only one island
##    if MP.geom_type is 'Polygon':
##        if MP.area>min_area: MP = [MP]
##        else: MP = []
##    else: MP = [P for P in MP if sPolygon(P.exterior).area > min_area]
##    # Separate Polygons in MP into islands and insides
##    II = []
##    for P in MP:
##        for i in P.interiors:
##            II.append(sPolygon(i))
##    MI = cascaded_union(II)
##    if MI.geom_type is 'Polygon':
##        largest_courtyard_area = MI.area
##    else:
##        largest_courtyard_area = max([i.area for i in MI]+[0.0])
##    islandPP = []
##    insidePP = []
##    for P in MP:
##        if sPolygon(P.exterior).area < largest_courtyard_area and MI.contains(P):
##            insidePP.append(P)
##        else:
##            islandPP.append(P)    
##    # Assign islands, courtyards, and insides
##    out = {}
##    out['islands'] = tuple([tuple(P.exterior.coords)[0:-1] for P in islandPP]) # exterior duplicates last point - remove it!
##    #out['insides'] = tuple([tuple(P.exterior.coords)[0:-1] for P in insidePP])
##    #out['courtyards'] = tuple([tuple(P.exterior.coords)[0:-1] for P in MI])
##    # - should also implement multipolygon relations in OSM for proper courtyard treatment
##    return out

### Uses simplify from shapely.py - topology preservation = True
### Do not use Polygon.py, as it does not preserve topology: it may cut off big chunks of contours.
### !!! This may cause nearby islands to overlap after simplification. Use shapley's simplify instead.




##################
# VECTOR TOOLBOX # 
##################
   

def scale_vec(v,a):
    return (v[0]*a,v[1]*a)

def dot_product(a,b):
    return a[0]*b[0]+a[1]*b[1]

def vec_sum(p,q):
    return (p[0]+q[0], p[1]+q[1])

def vec_diff(p,q):
    return (p[0]-q[0], p[1]-q[1])

def dist(p,q):
    return hypot(p[0]-q[0],p[1]-q[1])

def cos_abc(a,b,c):
    return dot_product(diff(b,a), diff(b,c))/dist(b,a)/dist(b,c)


def vec_product(U,V):
    return U[0]*V[1]-U[1]*V[0]
    
def sin_abc(A, B, C): 
    U = (A[0]-B[0],A[1]-B[1])
    V = (C[0]-B[0],C[1]-B[1])
    return (U[0]*V[1]-U[1]*V[0])/hypot(*U)/hypot(*V)

def unit_vec(p):
    l = hypot(*p)
    return (p[0]/l,p[1]/l)


#######################
# OPERATIONS ON LINES #
#######################

def line_lengths(line):
    if len(line)<2: return 0
    return [dist(line[n],line[n+1]) for n in range(len(line)-1)]











    











#########################
# DISTANCES TO CONTOURS #
#########################




def get_nearest_point_idx_idx(xy, points_lists): # do maybe later
    #dists = [[dist(xy, p) for p in points] for points in points_lists]
    #minargs = [min_arg]
    pass

def get_nearest_point_idx(xy, points):
    dists = [dist(xy, p) for p in points]
    return argmin(dists)

def get_nearest_contour_idx(xy, contours):
    PP = [sPolygon(c) for c in contours]
    XY = Point(xy)
    dists = [P.distance(XY) for P in PP]
    return argmin(dists)

# Largest circle in negative space
# returns radius = distance to nearest contour
def largest_circle(centre, obstacles):
    if not points_outside([centre], obstacles):
        return 0.0
    if not obstacles:
        return float('inf')
    C = Point(*centre)
    return min([LinearRing(O).distance(C) for O in obstacles])

















############################
# VISIBILITY AND ADJACENCY #
############################


        
        
        

def obs_corn_to_int(obs_corn, contours, offset = 0):
    L = [0] + list(cumsum([len(C) for C in contours]))[:-1]
    return [L[o]+c+offset for o,c in obs_corn]


def what_touches(contour, obstacles, items = 'all', tol = 1e-10):
    valid_items = {'corners','lines','contours'}
    if type(items) is str:
        items = set([items])
    if 'all' in items:
        items = valid_items
    invalid_items = items - valid_items
    if invalid_items:
        print('Invalid items passed to geometry.what_touches():')
        print(invalid_items)
        print('valid items are:')
        print(valid_items)
        return False
    OBJ = sPolygon(contour)
    OBSTACLES = [sPolygon(o) for o in obstacles]
    N = len(obstacles)
    TP = [n for n in range(N) if OBJ.distance(OBSTACLES[n])<tol]
    if items == {'contours'}:
        return TP
    if 'lines' in items:
        TL = []
        for tp in TP:
            O = obstacles[tp]
            M = len(O)
            LINES = [LineString((O[m],O[(m+1)%M])) for m in range(M)]
            TL += [(tp,m) for m in range(M) if OBJ.distance(LINES[m])<tol]
        if items == {'lines'}:
            return TL
    if 'corners' in items:
        TC = []
        for tp in TP:
            O = obstacles[tp]
            M = len(O)
            CORNERS = [Point(O[m]) for m in range(M)]
            TC += [(tp,m) for m in range(M) if OBJ.distance(CORNERS[m])<tol]
        if items == {'corners'}:
            return TC
    # at this point, there is more than one item to return:
    out = {}
    for item in items:
        if item == 'corners': out['corners'] = TC
        elif item == 'lines': out['lines'] = TL
        elif item == 'contours': out['contours'] = TP
    return out

##def visibility_from_wall(obs_wall_frac,boundary,obstacles, facets = None):
##    no, nw, fr = obs_wall_frac
##    

def visibility_from_corner(obs_corn,boundary,obstacles, facets = None):
    no, nc = obs_corn
    N = len(obstacles)
    obstacle = obstacles[no]
    M = len(obstacle)
    obstacles_reduced = [obstacles[n] for n in range(N) if n!=no] + \
                        [[obstacle[m] for m in range(M) if m!=nc]]
    triangle = [obstacle[nc-1],obstacle[nc],obstacle[(nc+1)%M]]
    centre = obstacle[nc]
    print(centre)
    if facets is None: augmented = visibility(centre,boundary,obstacles_reduced)
    else: augmented = visibility(centre,boundary,obstacles_reduced,facets)
    P = Polygon.Polygon(augmented) - Polygon.Polygon(triangle)
    if len(P) == 1:
        return P[0]
    return False

# Nov 21, 2015: fixed transparent walls (longer than 2R): use divs=0.5 (rather than 1) for points farther than 
# use Polygon.py - seems robust and faster than with shapely
def visibility(centre, boundary, obstacles, facets = 100):
    old_tol = Polygon.getTolerance()
    Polygon.setTolerance(10**-7)#10**-7 seems stable
    xc, yc = centre
    if type(boundary) is float or type(boundary) is int:
        radius = boundary
        R = radius * 1.1
        frame = Polygon.Polygon([(xc+R, yc+R),(xc-R, yc+R),(xc-R, yc-R),(xc+R, yc-R)])
        clip = False
    else:
        boundary_contour = ensure_frame_is_contour(boundary)
        if get_concave_corners(boundary_contour):
            Polygon.setTolerance(old_tol)
            return 'ERROR: boundary contour must be convex!'
        frame = Polygon.Polygon(boundary_contour)
        radius = widest(boundary_contour)
        clip = True
    obs = []
    for c in obstacles:
        o = Polygon.Polygon(c)
        if o.isInside(xc, yc):
            Polygon.setTolerance(old_tol)
            return []
        if o.overlaps(frame):
            obs.append(c)
    # calculate shadows
    R = radius + 0.5
    shadows = []
    for cont in obs:
        # all polygons are shifted to (0,0):
        vecs = [(xy[0]-xc,xy[1]-yc) for xy in cont]
        divs = [0.5*min(hypot(*v),R)/R for v in vecs]
        N = len(vecs)
        vxts = [(vecs[n][0]/divs[n],vecs[n][1]/divs[n]) for n in range(N)]
        sh = Polygon.Polygon([vecs[N-1],vecs[0],vxts[0],vxts[N-1]])
        for n in range(N-1):
            sh += Polygon.Polygon([vecs[n],vecs[n+1],vxts[n+1],vxts[n]])
        shadows.append(sh)
    # remove shadows
    FOV = Polygon.Polygon(make_circle((0,0), radius, facets))
    for sh in shadows:
        FOV -= sh
    FOV.shift(xc,yc) # Shift back to global coordinates
    if clip:
        FOV &= frame
    outsides = [n for n in range(len(FOV)) if FOV.isSolid(n)]
    Polygon.setTolerance(old_tol)
    if len(outsides) == 0:
        error('no contours found in FOV!')
    for n in outsides:
        if FOV.isInside(xc,yc,n):
            return FOV[n]
    return 'ERROR! Something went very wrong!'


# use Polygon.py - accurate representation of algortihm in my paper (50% slower than visibility_old() ).
# Long walls leak :(
# This algorithm is currently unpublished (see old versions of autoRS) and out of use.
def visibility_leaky_long_walls(centre, radius, obstacles, facets = 100):
    def Pentagon(p,q,r):
        def proj(p, d):
            m = 1.5*r/min(hypot(*p),1.5*r)
            return (p[0]*m,p[1]*m)
        return [p,q,proj(q,r), proj(vec_sum(unit_vec(p),unit_vec(q)),r),proj(p,r)]
    old_tol = Polygon.getTolerance()
    Polygon.setTolerance(10**-6)#10**-6 seems stable
    R = radius + 1
    xc, yc = centre
    frame = Polygon.Polygon(make_circle((xc,yc), R, facets))
    obs = []
    for c in obstacles:
        o = Polygon.Polygon(c)
        if o.isInside(xc, yc):
            return []
        if o.overlaps(frame):
            obs.append(c)
    # calculate shadows
    R = radius + 0.5
    shadows = []
    for cont in obs:
        # all polygons are shifted to (0,0):
        vecs = [(xy[0]-xc,xy[1]-yc) for xy in cont]
        mults = [1.5*R/min(hypot(*v),1.5*R) for v in vecs]
        N = len(vecs)
        vxts = [(vecs[n][0]*mults[n],vecs[n][1]*mults[n]) for n in range(N)]
        sh = Polygon.Polygon()
        Pcentre = Point(0,0)
        for n in range(N):
            n_ = (n+1)%N
            #if Pcentre.distance(LineString((vecs[n],vecs[n_])))<R:
            if vecs[n][0]*vecs[n_][0]+vecs[n][1]*vecs[n_][1] >= 0: # acute angle
                if mults[n]>1.0 and mults[n_]>1.0:
                    sh += Polygon.Polygon([vecs[n],vecs[n_],vxts[n_],vxts[n]])
            else: # obtuse angle
                sh += Polygon.Polygon(Pentagon(vecs[n],vecs[n_],R))
        shadows.append(sh)
    # remove shadows
    FOV = Polygon.Polygon(make_circle((0,0), radius, facets))
    for sh in shadows:
        FOV -= sh
    outsides = [n for n in range(len(FOV)) if FOV.isSolid(n)]
    Polygon.setTolerance(old_tol)
    if len(outsides) == 0:
        error('no contours found in FOV!')
    for n in outsides:
        if FOV.isInside(0,0,n):
            return [(xy[0]+xc,xy[1]+yc) for xy in FOV[outsides[n]]]




    




########################
# POINT SET OPERATIONS #
########################

def generate_random_points_outside(N, lrbt, obstacles):
    # generate N points in the negative space
    density = contours_density(lrbt, obstacles)
    N_PPP = int(N*1.1/(1-density))
    pts=[]
    while len(pts)<N:
        pts = random_points(N_PPP,lrbt)
        pts = points_outside(pts,obstacles)
    return pts[:N]
    
def random_points(N,lrbt):
    return [(random.uniform(lrbt[0],lrbt[1]),random.uniform(lrbt[2],lrbt[3])) for i in range(N)]

def points_outside(points,contours):
    pointsOutside=[]
    polyContours=[Polygon.Polygon(c) for c in contours]
    for i in points:
        isOutside=True
        for j in polyContours:
            if j.isInside(i[0],i[1])==True:
                isOutside=False
        if isOutside==True:
            pointsOutside.append(i)
    return pointsOutside


def points_in_frame(points, frame):
    F = Polygon.Polygon(ensure_frame_is_contour(frame))
    return [p for p in points if F.isInside(*p)]
    
def are_points_in_contour(points,contour):
    C = Polygon.Polygon(contour)
    return [C.isInside(p[0],p[1]) for p in points]

def points_outside_inside(points,contours):
    pointsOutside=[]
    pointsInside=[]
    polyContours=[Polygon.Polygon(c) for c in contours]
    for i in points:
        isOutside=True
        for j in polyContours:
            if j.isInside(i[0],i[1])==True:
                isOutside=False
        if isOutside==True:
            pointsOutside.append(i)
        else:
            pointsInside.append(i)
    return (pointsOutside, pointsInside)



def make_grid(centre, dxdy, nxny):
    xy = [[],[]]
    for i in [0,1]:
        xy[i] = [centre[i] + (n-(nxny[i]-1)/2)*dxdy[i] for n in range(nxny[i])] 
    return [(x,y) for x in xy[0] for y in xy[1]]




######################
# FRAMING OPERATIONS #
######################


def reduce_lrbt(lrbt, d):
    l,r,b,t = lrbt
    if r-l < 2*d or t-b < 2*d:
        print('Bad parameters to geometry.reduce_lrbt:',lrbt,d)
        return None
    else:
        return [l+d,r-d,b+d,t-d]


def ensure_frame_is_contour(frame, cw = None):
    if type(frame[0]) is int or type(frame[0]) is float:
        frame = [(frame[0],frame[2]),(frame[1],frame[2]),(frame[1],frame[3]),(frame[0],frame[3])]
    if cw != None:
        orientation = Polygon.Polygon(frame).orientation()[0]
        if (cw == True and orientation == -1.0) or (cw == False and orientation == 1.0):
            frame = frame[::-1]
    return frame

def contours_density(frame,contours):
    Pin = Polygon.Polygon()
    for c in contours:
        Pin.addContour(c)
    Pout = Polygon.Polygon(ensure_frame_is_contour(frame))
    Aoverlap = (Pin & Pout).area()
    Aout = Pout.area()
    return Aoverlap/Aout

def subtile_min_density(div, lrbt, contours):
    if type(div) is not int or div<1:
        print('div = '+str(div)+' not supported in geometry.subtile_max_density().')
        return False
    if div == 1:
        return contours_density(lrbt,contours)
    dx = (lrbt[1]-lrbt[0])/div 
    dy = (lrbt[3]-lrbt[2])/div
    densities = []
    for x in range(div):
        for y in range(div):
            subtile = [lrbt[0]+dx*x, lrbt[0]+dx*(x+1), lrbt[2]+dy*y, lrbt[2]+dy*(y+1)]
            densities.append(contours_density(subtile,contours))
    return min(densities)
































###########################################
# NATURAL NEIGHBOUR LINKS BETWEEN ISLANDS # 
###########################################

# Older versions of this code (Mar 27 2015) sometimes return incorrect projection indices (wrong orientation)
# New code makes 'LCIs', not 'link_contact_indices'
def shortest_link(c1, c2):
    def makeLSs(c):
        N = len(c)
        return [LineString([c[i],c[(i+1)%N] ]) for i in range(N)]  
    # find link in c2 that is closest to c1: ls2 == LineString(c2[n2],c2[n2_]) 
    lsls = makeLSs(c2)
    n2 = argmin([LinearRing(c1).distance(ls) for ls in lsls])
    n2_ = (n2+1)%len(c2)
    ls2 = lsls[n2]
    # find link in c1 closest to ls2: ls1 == LineString(c1[n1],c1[n1_]) 
    lsls = makeLSs(c1)
    dd = [ls2.distance(ls) for ls in lsls]
    n1 = argmin(dd)
    n1_ = (n1+1)%len(c1)
    ls1 = lsls[n1]
    # At this point, the two nearest segments (ls1, ls2) are known.
    # See if shortest link is between endpoints:
    dmin = dd[n1]
    combos = [(n1, n2), (n1, n2_), (n1_, n2), (n1_,n2_)]
    dd = [dist(c1[nn[0]],c2[nn[1]]) for nn in combos] + [dmin+0.001]
    case = argmin(dd)
    if case < 4:
          nn = combos[case]
          return (nn, (c1[nn[0]],c2[nn[1]]), dd[case])
    # Otherwise: what is the shortest of the four possible point-to-segment links:
    else:
        p1 = Point(c1[n1])
        p1_ = Point(c1[n1_])
        p2 = Point(c2[n2])
        p2_ = Point(c2[n2_])
        case = argmin([p1.distance(ls2), p1_.distance(ls2), p2.distance(ls1), p2_.distance(ls1)])
        if case == 0:
            pr = ls2.project(p1,normalized=True)
            nn = (n1, n2+pr)
            pp = (c1[n1], ls2.interpolate(pr,normalized=True).coords[0])
        elif case == 1:
            pr = ls2.project(p1_,True)
            nn = (n1_, n2+pr)
            pp = (c1[n1_], ls2.interpolate(pr,True).coords[0])
        elif case == 2:
            pr = ls1.project(p2,True)
            nn = (n1+pr, n2)
            pp = (ls1.interpolate(pr,True).coords[0],c2[n2])
        else: # case == 3:
            pr = ls1.project(p2_,True)
            nn = (n1+pr, n2_)
            pp = (ls1.interpolate(pr,True).coords[0],c2[n2_])
        return (nn, pp, dist(*pp))
    # (num,num) of (potentially float) indices of location of segment on contours c1 and c2
    # (point, point): coords connecting of segment
    # length of connecting segment

def neighbour_links(cc, max_len = False, allow_crosses = False):
    # links are tuples of 4 elements:
    # 0) (i,j) - indices of polygons linked
    # 1) (fi, fj) - floating indices of contact points on polygons
    # 2) ((xi,yi),(xj,yj)) - coords of link segment
    # 3) d - link segment length
    polys = []
    cps = []
    links = []
    LSlinks = []
    dists = []
    N = len(cc)
    #print(N, 'contours, sl_func:', sl_func)
    sps = [sPolygon(c) for c in cc]
    for i in range(N):
        for j in range(i+1,N):
            if max_len is False or sps[i].distance(sps[j]) <= max_len:
                sl = shortest_link(cc[i],cc[j])
                polys.append((i,j))
                cps.append((sl[0]))
                links.append(sl[1])
                LSlinks.append(LineString(sl[1]))
                dists.append(sl[2])
    # for all segment pairs that intersect, remove the longer segment
    M = len(polys)
    #print(M, ' links @ ',time.clock())
    keep = [True]*M
    if not allow_crosses:
        for k in range(M):
            for m in range(k+1,M):
                if keep[k] and keep[m]:
                    if LSlinks[k].crosses(LSlinks[m]):
                        if dists[k] > dists[m]: keep[k] = False
                        else: keep[m] = False
    #print(sum(keep), ' links left @ ',time.clock())
    # now also remove segments that itersect contours
    mp = MultiPolygon(sps)
    shorten_links = 0.01 # in m, to avoid links touching their own polygons
    #print('testing intersections against %d contours'%len(sps))
    for m in range(M):
        if keep[m]:
            s = LSlinks[m]
            ls = LineString([s.interpolate(shorten_links), s.interpolate(-shorten_links)])
            try: # This test occasionally crashes
                if ls.intersects(mp): keep[m] = False # ! NOT crosses()! - include boundaries of MP.
            except: # Test intersection contour by contour instead: 
                for sp in sps:
                    if ls.intersects(sp): keep[m] = False
    #print(sum(keep), ' links left @ ',time.clock())
    # Find locations of contact indices for each polygon
    lcidxs = [[] for i in range(N)]
    for m in range(M):
        if keep[m]:
            cp = cps[m]
            pp = polys[m]
            for i in [0,1]:
                lcidxs[pp[i]].append(float(cp[i]))# convert ints to floats so that JSON can encode the list.
    # Nearest neighbours:
    NNs = [[] for n in range(N)]
    #print(NNs)
    for pair in [polys[m] for m in range(M) if keep[m]]:
        #print(pair)
        NNs[pair[0]].append(pair[1])
        NNs[pair[1]].append(pair[0])
    #print(NNs)
    return {'links': [links[m] for m in range(M) if keep[m]],
            'island_NNs': [sorted(nn) for nn in NNs],
            'LCIs': [sorted(lci) for lci in lcidxs]}  
# links: are segments (for drawning)
# island_NNs: is the connectivity graph of the islands, by island index
# LCIs: contain the the sorted values of where each island is touched by its
# successive links, encoded as integer indices for links touching a corner,
# and floats with fractional part showing the fraction of the distance between
# the corners with the surrounding integers contact points.


# May be useful for later -> move to ai.py if used.
def NN_triangulation(NNs):
    triplets = set()
    for i, island in enumerate(NNs):
        for neigh in island:
            for twohop in NNs[neigh]:
                if twohop != i and i in NNs[twohop]:
                    triplets.add(tuple(sorted([i, neigh, twohop])))
    return sorted(list(triplets))

def sets(k, n):
    return [()] if k == 0 else ((i,)+s for i in range(n) for s in sets(k-1, i))

# this is a hack for managing old files and for the bug in shortest_link...()
# repair LCIs that are > len(island)
def clean_link_contact_indices(LCIs, islands):
    cleanLCIs = []
    for i in range(len(LCIs)):
        N = len(islands[i])
        cleanLCIs.append([i if i<N else 2*N-i for i in LCIs[i]])
    return {'link_contact_indices':cleanLCIs}




















########
# MISC #
########

def make_square(centre, radius):
    return make_circle(centre, radius, facets = 4, phase=0.5)
def make_circle(centre, radius, facets = 100, phase=0):
    a = 2*pi/facets
    return [(centre[0]+radius*cos(a*(n+phase)),centre[1]+radius*sin(a*(n+phase))) for n in range(facets)]
    #return list(Point(centre[0],centre[1]).buffer(radius,int(facets/4)).exterior.coords)[0:-1]


# Used in Meisam's paper
# rasterization is performed at 1 symbol/m^2:
def rasterize_contours(lrbt, contours):
    MP = Polygon.Polygon()
    for c in contours:
        MP.addContour(c)
    # MP = cascaded_union([sPolygon(c) for c in contours])
    NX = 1000#round(lrbt[1]-lrbt[0])
    NY = 1000#round(lrbt[3]-lrbt[2])
    CX = 0#(lrbt[0]+lrbt[1])/2
    CY = 0#(lrbt[3]+lrbt[2])/2
    txt = bytearray((NX+1)*NY)
    SX = -(NX-1)/2
    SY = -(NY-1)/2
    n = 0
    for x in range(NX):
        for y in range(NY):
            if MP.isInside(SX+x,SY+y): #MP.contains(Point(SX+x,SY+y)):
                txt[n] = 49 #'1'
            else:
                txt[n] = 48 #'0'
            n = n + 1
        txt[n] = 10 # '\n'
        n = n + 1
    return txt

