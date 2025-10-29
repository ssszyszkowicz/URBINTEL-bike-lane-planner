#from geometry import make_circle,merge_and_fill,dilate,dist - deprecated, no more Polygon!
from scipy.spatial.distance import pdist, squareform
from shapely.geometry.polygon import orient as shapely_orient
from shapely.ops import unary_union
from utils import *
from qcommands import *
from pyopencl import enqueue_copy as cl_enqueue_copy, CommandQueue as cl_CommandQueue, Buffer as cl_Buffer, mem_flags as cl_mem_flags
from pyopencl import array as cl_array, Program as cl_Program
from tripy import earclip as tripy_earclip
from polygon_triangulate import polygon_triangulate
from math import hypot, sin


##def merge_and_fill(contours,engine='shapely'):
##    if len(contours) < 2: return contours
##    if engine == 'shapely':
##        print('geometry.merge_and_fill() not yet implemented with shapely engine')
##        return False
##        CU = unary_union([sPolygon(c) for c in contours])
##        if CU.geom_type == 'Polygon': PP = [CU]
##        else:
##            assert CU.geom_type == 'MultiPolygon'
##            PP = [P for P in CU]
##        # not finished - see merge()
##    if engine == 'Polygon':    
##        pp = Polygon.Polygon()
##        for c in contours:
##            pp += Polygon.Polygon(c)
##            if not random.randint(0,19): # Fill holes about once in 20 
##                pp = fillHoles(pp)
##        pp = fillHoles(pp)
##        pp.simplify()
##        pp = prunePoints(pp)
##        return [p for p in pp] # needed to iterate over Polygon object


def dilate(contours, radius, interiors = False):
    if interiors:
        print('interiors not implemented in dilate()')
        return False
    if not contours: return []
    MP = shapely_MultiPolygon(contours)
    DI = MP.buffer(distance=radius, **dilation_style)
    return shapely_get_exteriors(DI)  


def dist(p,q):
    return hypot(p[0]-q[0],p[1]-q[1])

def make_circle(centre, radius, facets = 100, phase=0):
    a = 2*pi/facets
    return [(centre[0]+radius*cos(a*(n+phase)),centre[1]+radius*sin(a*(n+phase))) for n in range(facets)]
    #return list(Point(centre[0],centre[1]).buffer(radius,int(facets/4)).exterior.coords)[0:-1]







class Frame:
    def __init__(self,lrbt):
        self.lrbt = lrbt
        l,r,b,t = lrbt
        self.sPolygon = sPolygon([(l,b), (l,t), (r, t), (r, b)])
        
    


def dilate_boundaries(ListOfShapely): #returns G3
    g = GeometryCollection(ListOfShapely)
    b = g.buffer(10)
    t = b.geom_type
    if t == 'MultiPolygon': PP = list(b.geoms)
    elif t == 'Polygon': PP = [b]
    else: Qer('Invalid shapely geom_type',t,'in dilate_boundaries')
    x,y = [LoL([P.exterior.coords.xy[d] for P in PP]) for d in [0,1]]
    x(y)
    return Geometry(x,y,'xy')
    
        

def polygon_centre(G3):
    if len(G3) == 1: return tuple(sPolygon(G3[0]).centroid.coords)[0]
    P = [sPolygon(g) for g in G3]
    A = [p.area for p in P]
    SA = sum(A)
    X,Y = tuple(zip(*[tuple(p.centroid.coords)[0] for p in P]))
    N = len(P)
    return (sum(A[n]*X[n] for n in Qr(N)), sum(A[n]*Y[n] for n in Qr(N)))

# tripy fails for some difficult contours.
# First, try with tripy, if geometry is wrong, try the slow python implementation
def triangulate_contour(contour,fast_bad=True):
    if fast_bad: return tripy_earclip(contour) # faster but occasionally broken
    # contour must be CCW
    contour = clean_contour(contour)
    #print('polygon_triangulate',len(contour))
    if not LinearRing(contour).is_ccw: contour=contour[::-1]
    ixs =  polygon_triangulate(len(contour),
                               [p[0] for p in contour],
                               [p[1] for p in contour])
    return [tuple(contour[p] for p in t) for t in ixs]

def clean_contour(contour):
    out = [contour[0]]
    for xy in contour[1:]:
        if xy != out[-1]: out.append(xy)
    if out[0]==out[-1]: out.pop()
    return out


def lrbt_and_lrbt(lrbt1,lrbt2):
    l1,r1,b1,t1 = lrbt1
    l2,r2,b2,t2 = lrbt2
    L = max(l1,l2)
    R = min(r1,r2)
    B = max(b1,b2)
    T = min(t1,t2)
    if L<R and B<T: return (L,R,B,T)
    else: return False


def boundary_sPolygons(boundary):
    if not boundary: return []
    d = get_depth(boundary)
    if d==2: return [sPolygon(boundary)]
    if d==3:
        B = [sPolygon(b) for b in boundary]
        B.sort(key = lambda p:p.area, reverse=True)
        return B
    return False


# unprotected - offline only
def polygon_to_solids(contour, holes): #G2,G3->G3
    POLYGON = sPolygon(contour,holes=holes)
    AREA = POLYGON.area
    l,b,r,t = POLYGON.bounds
    l-=1;r+=1;b-=1;t+=1
    bounds = [sPolygon(h).bounds for h in holes]
    x = [l]+[(b[0]+b[2])/2 for b in bounds]+[r]
    x.sort()
    solids = []
    for n in Qr(len(x)-1):
        CUTTER = sPolygon(make_box(lrbt=[x[n],x[n+1],b,t]))
        G = POLYGON.intersection(CUTTER)
        if G.geom_type == 'Polygon': solids.append(G)
        elif G.geom_type == 'MultiPolygon': solids += [g for g in G.geoms] ### SZ
    SOLIDS_AREA = sum(s.area for s in solids)
    if SOLIDS_AREA/AREA<0.999999: return "Failed to partition geometry: some contours were lost!"
    return [shapely_orient(s,-1).exterior.coords[0:-1] for s in solids] # -1 is CW, for .shp solid
#if G.geom_type == 'Polygon': solids.append(tuple(G.exterior.coords[0:-1]))
    
# unprotected - offline only
def contours_to_solids(contours): #G3->G3
    if len(contours) < 2: contours
    SP = [sPolygon(C) for C in contours]
    area = [sp.area for sp in SP]
    N = len(SP)
    parents = [[] for _ in Qr(N)]
    for n in Qr(N):
        for m in Qr(n):
            Ia = (SP[n].intersection(SP[m])).area
            if Ia/area[m]>0.999999: parents[m].append(n)
            if Ia/area[n]>0.999999: parents[n].append(m)
    if max([len(p) for p in parents])>1: return "Contours have multiple parents: topology not yet supported!"
    polygons = [[] for _ in Qr(N)]
    for n,p in enumerate(parents):
        if not p: polygons[n].append(contours[n])
    for n,p in enumerate(parents):
        if p: polygons[p[0]].append(contours[n]) # only one parent each
    out = []
    for p in polygons:
        if len(p) == 1: out.append(p[0])
        if len(p) > 1: out += polygon_to_solids(p[0],p[1:])
    return [clean_contour(c) for c in out]



def assign_holes_to_solids(solids, holes):
    S = [sPolygon(s) for s in solids]
    out = [[] for s in S]
    for i,h in enumerate(holes):
        H = sPolygon(h)
        for n,s in enumerate(S):
            if s.contains(H): out[n].append(i)
    return out

def dilate_and_merge(G4,r=0): #G4->G3
    if not G4: return False
    for n,g in enumerate(G4):
        if n == 0:
            G = unary_union([sPolygon(c).buffer(r) for c in g])
        else: G = G.union(unary_union([sPolygon(c).buffer(r) for c in g]))
    if G.geom_type == 'Polygon': return [tuple(G.exterior.coords[0:-1])]
    if G.geom_type == 'MultiPolygon': return [tuple(G.geoms[n].exterior.coords[0:-1]) for n in range(len(G.geoms))] # need .geoms (twice)!
    return False


def scale_points(points, s, ys=None):
    x = s
    if ys == None: y = s
    else: y = ys
    return tuple((p[0]*x, p[1]*y) for p in points)

def translate_points(points, dx, dy):
    return tuple((p[0]+dx, p[1]+dy) for p in points)

def lrbt(points):
    xx = tuple(xy[0] for xy in points)
    yy = tuple(xy[1] for xy in points)
    return [min(xx),max(xx),min(yy),max(yy)]

def bbox(points):
    xx = tuple(xy[0] for xy in points)
    yy = tuple(xy[1] for xy in points)
    return [(min(xx),min(yy)),(max(xx),max(yy))]

def make_box(bbox=None, lrbt=None):
    if bbox:
        l,b = bbox[0]
        r,t = bbox[1]
    elif lrbt:
        l,r,b,t = lrbt
    return [(l,b), (r,b), (r,t), (l,t)] # keep this order for texcoord in shaders

def hex_grid(lrbt, r, vertical=True, forGPU=False):
    if not vertical:
        print('Non-vertical hex_grid not implemented!')
        return False
    r32 = 0.5*3**0.5
    w = lrbt[1]-lrbt[0]
    h = lrbt[3]-lrbt[2]
    xs = r * r32 * 2
    ys = r * 3
    xN = int(w//xs)
    yN = int(h//ys)
    x0 = lrbt[0]
    y0 = lrbt[2]
    out = []
    np_circle = Qf32(make_circle((0,0), r, 6, 0.5)).reshape(6,2)
    for x in Qr(xN+2):
        for y in Qr(yN+2):
            for c in [(x0+xs*x,y0+ys*y), (x0+xs*(x+0.5),y0+ys*(y+0.5))]: 
                if forGPU:
                    c_ = np_array(c,dtype='float32').reshape(1,2)
                    for n in range(6):
                        n_ = (n+1)%6
                        out += [c_, c_+np_circle[n], c_+np_circle[n_]]
                else: out.append(make_circle(c, r, 6, 0.5))
    return np_concatenate(out) if forGPU else out




def buffer_polygon(contour, radius):
    E = merge_and_fill(dilate([contour], radius))
    #sP = sPolygon(contour).buffer(distance=radius, cap_style=1, join_style=1)
    #E = shapely_get_exteriors(sP)
    if len(E)>1: return False
    else: return E[0]
    
def closing(contours, radius, interiors = False):
    if interiors:
        print('interiors not implemented in closing()')
        return False
    if not contours: return []
    if False:# This doesnt work:
        MP = shapely_MultiPolygon(contours)
        DI = MP.buffer(distance=radius, **dilation_style)
        ME = unary_union(DI)
        ER = ME.buffer(distance=-radius, **dilation_style)
        return shapely_get_exteriors(ER)
    else: #Instead, do step by step:
        CC = dilate(contours, radius)# dilate
        CC = merge_and_fill(CC)# merge
        return dilate(CC, -radius) # erode   


def dist(p,q):
    return ((p[0]-q[0])**2+(p[1]-q[1])**2)**0.5
    
def cicrumcircle(points):
    mm, MM = bbox(points)
    mx, my = mm
    Mx, My = MM
    c = ((mx+Mx)/2,(my+My)/2)
    r = dist((mx,my),(Mx,My))/2
    return (c,r)
##    if False: # very flaky (incorrect m,n choice?)            
##        N = len(points)
##        AM = np_argmax(pdist(points))
##        m = int((2*AM+0.25)**0.5+0.5)
##        n = int(AM - m*(m-1)/2)
##        r = geometry.dist(points[n],points[m])*(0.5*3**0.5)
##        x = (points[n][0]+points[m][0])/2
##        y = (points[n][1]+points[m][1])/2
##        return ((x,y),r)












##################### HASH GRID ##################
CL_HASH_LINES_CODE = ('HASH_LINES', """
__kernel void B(
__global const uint *r_offsets, // ui32
const uint NH, // ui32
__global const uint *hr_offsets, // ui32
__global ulong *hw_data, // ui64
__global uint *hrw_temp_len, // ui32
__global const uint *o_offsets, //ui32
__global const uint *o_data //ui32
)
{
    uint r = get_global_id(0);
    uint ro = r_offsets[r];
    uint ns = r_offsets[r+1]-ro-1;
    for(uint s=0;s<ns;s++) {
        uint gs = s+ro;
        uint d_o = o_offsets[gs-r]; // calculate seg offset properly
        uint dl = o_offsets[gs-r+1] - d_o; // calculate seg offset properly
        uint k = 0;
        uint h = o_data[d_o];
        while(k<dl && h<NH) {
            uint len = atomic_inc(hrw_temp_len+h);
            hw_data[hr_offsets[h]+len] = (((ulong)r)<<32)+s;
            h = o_data[d_o+(++k)];
        }
    }
}


__kernel void A(
__global const float *r_x,
__global const float *r_y,
__global const uint *r_offsets, // ui32
const float hg_res,
const float hg_left,
const float hg_bottom,
const uint hg_XN, // ui32
const uint hg_YN, // ui32
__global const uint *o_offsets, //ui32
__global uint *o_data //ui32
)
{
    uint r = get_global_id(0);
    uint ro = r_offsets[r];
    uint ns = r_offsets[r+1]-ro-1;
    for(uint s=0;s<ns;s++){
        uint gs = s+ro;
        float x1 = r_x[gs];
        float x2 = r_x[gs+1];
        float y1 = r_y[gs];
        float y2 = r_y[gs+1];
        int X1,X2,Y1,Y2; // ! can be negative if outside frame!
        uint o_off = o_offsets[gs-r]; // calculate seg. offset properly!
        float slope = (y2-y1)/(x2-x1);
        if(slope<1.0 && slope>-1.0) { // couldn't get abs() to work
            float t;
            if(x2<x1) { // order: x1, x2
                t = x1; x1 = x2; x2 = t;
                t = y1; y1 = y2; y2 = t;
            }
            X1 = (int)floor((x1-hg_left)/hg_res);
            X2 = (int)floor((x2-hg_left)/hg_res);
            uint len = 0;
            float i1;
            float i2 = y1;
            float dy = hg_res*slope;
            Y2 = (int)floor((y1-hg_bottom)/hg_res);
            for(int X=X1; X<=X2; X++) {
                i1 = i2;
                if(X==X1){i2 = y1 + ((X+1)*hg_res+hg_left-x1)*slope;}
                if(X==X2){i2=y2;}
                if(X!=X1 && X!=X2){i2 += dy;}
                Y1 = Y2;
                Y2 = (int)floor((i2-hg_bottom)/hg_res);
                if(X>=0 && X<hg_XN) {
                    char Y1hash=0;
                    char Y2hash=0;
                    if(Y1>=0 && Y1<hg_YN) {Y1hash=1;}
                    if(Y2>=0 && Y2<hg_YN && (Y1!=Y2 || Y1hash==0)) {Y2hash=1;}
                    if(Y1hash) {o_data[o_off+len++] = X + Y1*hg_XN;}
                    if(Y2hash) {o_data[o_off+len++] = X + Y2*hg_XN;}
                }
            }
        } else {// can also handle slope = +/-inf
            float t;
            if(y2<y1) { // order: y1, y2
                t = y1; y1 = y2; y2 = t;
                t = x1; x1 = x2; x2 = t;
            }
            Y1 = (int)floor((y1-hg_bottom)/hg_res);
            Y2 = (int)floor((y2-hg_bottom)/hg_res);
            uint len = 0;
            float i1;
            float i2 = x1;
            float dx = hg_res/slope;
            X2 = (int)floor((x1-hg_left)/hg_res);
            for(int Y=Y1; Y<=Y2; Y++) {
                i1 = i2;
                if(Y==Y1){i2 = x1 + ((Y+1)*hg_res+hg_bottom-y1)/slope;}
                if(Y==Y2){i2=x2;}
                if(Y!=Y1 && Y!=Y2){i2 += dx;}
                X1 = X2;
                X2 = (int)floor((i2-hg_left)/hg_res); 
                if(Y>=0 && Y<hg_YN) {
                    char X1hash=0;
                    char X2hash=0;
                    if(X1>=0 && X1<hg_XN) {X1hash=1;}
                    if(X2>=0 && X2<hg_XN && (X1!=X2 || X1hash==0)) {X2hash=1;}
                    if(X1hash) {o_data[o_off+len++] = X1 + Y*hg_XN;}
                    if(X2hash) {o_data[o_off+len++] = X2 + Y*hg_XN;}
                }
            }
        }
    }
}
""")
# Hash = self.XN*yn+xn
class HashGrid:
    def get_cell_as_lrbt(self, h):
        x = h%self.XN
        y = h//self.XN
        l = self.left + x*self.res
        r = l+self.res
        b = self.bottom + y*self.res
        t = b+self.res
        return (l,r,b,t)
    def find_all_in_lrbt(self, lrbt, name, with_subindex = False):
        HT = self.hashed[name]
        l,r,b,t = lrbt
        out = set()
        if with_subindex or HT.data.dtype != np_uint64:
            for h in self._hash_rectangle((l,b),(r,t)): out |= set(HT[h])
            return np_array(tuple(out),HT.data.dtype)
        else:
            for h in self._hash_rectangle((l,b),(r,t)): out |= set(HT[h]>>32) # remove subindices
            return Qu32(tuple(out))
    def find_all_on_polygon(self, name, polygon):
        H = self._hash_polygon(polygon)
        POLYGON = sPolygon(polygon)
        HASHED = self.hashed[name]
        GEO = self.geometry[name]
        N = list(set(lol_to_tuple([HASHED[h] for h in H])))
        return  [n for n in N if POLYGON.intersects(GEO[n])]
    def __len__(self): return self.XN*self.YN
    def __init__(self, COMPUTE, lrbt, res=100):
        #QL("COMPUTE res {hashed{geometry{data{centre{radius",self,locals())
        self.COMPUTE = COMPUTE
        l,r,b,t=lrbt
        self.left = l
        self.bottom = b
        self.XN = int((r-l)/res)+1
        self.YN = int((t-b)/res)+1
        self.res = res
        self.hashed = {}
        self.geometry = {}
        self.data = {}
        self.centre = {}
        self.radius = {}
    def freeze(self):
        for a in 'hashed geometry centre radius'.split():
            attr = getattr(self,a)
            for k in attr:
                if type(attr[k]) is list: attr[k] = tuple(attr[k])
    def _hash_polygon(self, polygon):
        return self._hash_rectangle(*bbox(polygon))
    def hash_rectangle_to_lrbt(self,p1,p2):
        res = self.res
        nx = int((min(p1[0],p2[0])-self.left)//res)
        nX = int((max(p1[0],p2[0])-self.left)//res)
        ny = int((min(p1[1],p2[1])-self.bottom)//res)
        nY = int((max(p1[1],p2[1])-self.bottom)//res)
        XN = self.XN
        YN = self.YN
        if nx<0 and nX<0: return None
        if nx>=XN and nX>=XN: return None
        if ny<0 and nY<0: return None
        if ny>=YN and nY>=YN: return None
        return (max(0,nx),min(XN-1,nX),max(0,ny),min(YN-1,nY))
    def _hash_rectangle(self,p1,p2):
        lrbt = self.hash_rectangle_to_lrbt(p1,p2)
        if lrbt==None: return []
        l,r,b,t = lrbt 
        XN = self.XN
        ### Numpy acceleration of check = sum(...) in comment:
        W = (r-l+1)
        H = (t-b+1)
        #print(W,H)
        out = Qu32(W*(t-b+1), None)
        for n,Y in enumerate(Qr(b,t+1)):
            XNY = XN*Y
            out[W*n:W*(n+1)] = np_arange(XNY+l,XNY+r+1)
        ###check = np_array(sum([[XN*Y+X for X in Qr(l,r+1)] for Y in Qr(b,t+1)],[]),'uint32')
        ###print(sum(check==out),out.shape) # TESTED - WORKS!
        return out
    def _hash_point(self,p):
        X = int((p[0]-self.left)/self.res)
        if X<0: X = 0
        elif X>=self.XN: X = self.XN-1
        Y = int((p[1]-self.bottom)/self.res)
        if Y<0: Y = 0
        elif Y>=self.YN: Y = self.YN-1
        return X + Y * self.XN

    def find_points(self, polygon, name):
        POLYGON = sPolygon(polygon)
        hsh = self.hashed[name]
        ids = list(set(sum([list(hsh[H]) for H in self._hash_rectangle(*bbox(polygon))],[])))
        shp = self.geometry[name]
        return [i for i in ids if POLYGON.contains(shp[i])]
        
    def find_one_polygon(self,point,name):
        POINT = Point(point)
        H = self._hash_point(point)
        if H is not None:
            geo = self.geometry[name]
            rad = self.radius[name]
            ctr = self.centre[name]
            for P in self.hashed[name][H]:
                if dist(point,ctr[P]) < rad[P]:
                    if geo[P].contains(POINT): return P
        return None
    def find_one_line(self,point,name_s,d_max,values=None,_scale_test=0.146):
        if type(name_s) is str: names = [name_s]
        else: names = name_s
        out_L = None
        x,y = point
        d_test = self.res*_scale_test
        r = min(d_max, d_test)
        H = self._hash_rectangle((x-r,y-r),(x+r,y+r))
        if H is not None:
            for name in names:
                hashed = self.hashed[name]
                geo = self.geometry[name]
                POINT = Point(point)
                for h in H:
                    for L in set((hashed[h]>>32).astype('uint32')): # New implementation: line hashes index line<<32+segment in uint64
                        d = geo[L].distance(POINT)
                        if d<r and (values==None or hashed.data[L] in values):
                            r = d
                            out_L = L
                            out_name = name
        if out_L is None:
            if d_test>=d_max: return None
            return self.find_one_line(point,name_s,d_max,values,1.5*_scale_test+0.281)
        if type(name_s) is str: return out_L
        else: return (out_name, out_L)
    def find_one_point(self,point,name_s,d_max,_scale_test=0.146):
        if type(name_s) is str: names = [name_s]
        else: names = name_s
        out_val = None
        x,y = point
        d_test = self.res*_scale_test
        r = min(d_max, d_test)
        H = self._hash_rectangle((x-r,y-r),(x+r,y+r))
        if H is not None:
            for name in names:
                hashed = self.hashed[name]
                geo = self.geometry[name]
                POINT = Point(point) 
                for h in H:
                    for P in hashed[h]:
                        d = geo[P].distance(POINT)
                        if d<r:
                            r = d
                            out_val = P
                            out_name = name
        if out_val is None:
            if d_test>=d_max: return None
            return self.find_one_point(point,name_s,d_max,1.5*_scale_test+0.281)
        if type(name_s) is str: return out_val
        else: return (out_name, out_val)
    def hash_points(self, name, points):
        hsh = [[] for _ in Qr(self.XN*self.YN)]
        for N,P in enumerate(points):
            hsh[self._hash_point(P)].append(N)
        self.geometry[name] = [Point(P) for P in points]
        self.hashed[name] = LoL(hsh,deepctype(len(points)))
    def hash_polygons(self, name, polygons):
        hsh = [[] for _ in Qr(self.XN*self.YN)]
        for N,P in enumerate(polygons):
            for H in self._hash_rectangle(*bbox(P)): hsh[H].append(N)
        self.geometry[name] = [sPolygon(P) for P in polygons]
        self.hashed[name] = LoL(hsh,deepctype(len(polygons)))
        CC = [cicrumcircle(P) for P in polygons]
        self.centre[name] = [cc[0] for cc in CC]
        self.radius[name] = [cc[1] for cc in CC]
    def hash_lines(self, name, x, y, offsets, seg_len_m_array):
        # This can be automated:
        PROGRAM = self.COMPUTE['BUILT']['HASH_LINES']
        CL_CONTEXT = self.COMPUTE['CL_CONTEXT']
        clQ = cl_CommandQueue(CL_CONTEXT)
        print(self.XN,self.YN)
        print(len(x),len(y),len(offsets),len(seg_len_m_array))
        print('Hashing',offsets.size-1,'lines:',end='') # off-1 lines!
        NH = self.XN*self.YN
        I = (seg_len_m_array/self.res).astype('uint32')
        d_offsets = np_concatenate((np_zeros(1,'uint32'), # sic, just one '0'
            np_cumsum((I+2)*2)))
        d_data = np_full(shape=d_offsets[-1],fill_value=NH,dtype='uint32')
        o_offsets = cl_input(d_offsets,CL_CONTEXT)
        o_data = cl_io(d_data,CL_CONTEXT)
        r_offsets = cl_input(offsets,CL_CONTEXT)
        r_x = cl_input(x,CL_CONTEXT)
        r_y = cl_input(y,CL_CONTEXT)
        hg_res = np_float32(self.res)
        hg_left = np_float32(self.left)
        hg_bottom = np_float32(self.bottom)
        hg_XN = np_uint32(self.XN)
        hg_YN = np_uint32(self.YN)
        
        PROGRAM.A(clQ, (offsets.size-1,), None,r_x,r_y,r_offsets,hg_res,
                  hg_left,hg_bottom,hg_XN,hg_YN,o_offsets,o_data)
        cl_enqueue_copy(clQ, d_data, o_data)
        print('A',end='')
        ## Step 2: Maybe optimize in openCL? Seems fast.
        h_len = np_bincount(np_concatenate((np_full(1,NH,'uint32'), d_data)))[:-1]
        h_offsets = np_concatenate((np_zeros(1,'uint32'),np_cumsum(h_len,dtype='uint32')))
        print('B',end='')
        ## Step 3:        
        hr_offsets = cl_input(h_offsets,CL_CONTEXT)
        h_data = Qu64(h_offsets[-1], None)
        hw_data = cl_output(h_data,CL_CONTEXT)
        hrw_temp_len = cl_io(Qu32(h_len.size,0),CL_CONTEXT)
        PROGRAM.B(clQ, (offsets.size-1,), None,
            r_offsets,np_uint32(NH),hr_offsets,hw_data,hrw_temp_len,
            o_offsets,o_data)
        cl_enqueue_copy(clQ, h_data, hw_data)
        h = LoL(dtype='uint64')
        h.N = NH
        h.offsets = h_offsets
        h.datalen = h_offsets[-1]
        h.data = h_data
        self.hashed[name] = h
        print('C',end='')
        ### shapely object
        geo = []
        for n,o in enumerate(offsets[:-1]):
            o_ = offsets[n+1]
            geo.append(LineString(tuple(zip(x[o:o_],y[o:o_]))))
        self.geometry[name] = geo
        print('D.')
    def hash_lines_slow(self, name, x, y, offsets, seg_len_m_array, store_shapely=False):
        if store_shapely: Qe('store_shapely not implemented in hash_lines')
        N = len(offsets) - 1
        res = self.res
        XN = self.XN
        YN = self.YN
        XNYN = XN*YN
        left = self.left
        bottom = self.bottom
        seg_len_i = 0
        seg_buffer = LoL()
        A = np_cumsum((Qu32(seg_len_m_array//res)+2)**2)
        O = Qu32(len(A)+1,None)
        O[0] = 0
        O[1:] = A
        D = Qu32(O[-1],XNYN)
        seg_buffer.load(D, O)
        code = Qu64(len(O)-1, None)
        for n in range(N):
            if n%10000==0:print(n,'/',N)
            a = offsets[n]
            b = offsets[n+1]
            for m in range(b-a-1):
                j = a+m
                k = j+1
                x0 = (x[j]-left)//res
                if x0<0: x0 = 0
                if x0>=XN: x0 = XN-1
                x1 = (x[k]-left)//res
                if x1<0: x1 = 0
                if x1>=XN: x1 = XN-1
                if x0>x1: x0,x1=x1,x0
                y0 = (y[j]-bottom)//res
                if y0<0: y0 = 0
                if y0>=YN: y0 = YN-1
                y1 = (y[k]-bottom)//res
                if y1<0: y1 = 0
                if y1>=YN: y1 = YN-1
                if y0>y1: y0,y1=y1,y0
                hits = []
                for v in range(int(x0),int(x1)+1):
                    for w in range(int(y0),int(y1)+1):
                        hits.append(v+XN*w)
                s = seg_buffer.offsets[seg_len_i]
                try:
                    seg_buffer.data[s:s+len(hits)] = Qu32(hits)
                except:
                    print(s,len(seg_buffer.data))
                    print(seg_buffer.data[s:s+len(hits)])
                    print(Qu32(hits))
                    input()
                code[seg_len_i] = (2**32)*n+m
                seg_len_i += 1
        A = Qu32(len(seg_buffer.data)+1, None)
        A[0:len(seg_buffer.data)] = seg_buffer.data
        A[-1] = XNYN
        B = np_bincount(A)[:-1]
        C = np_cumsum(B)
        A = Qu32(len(C)+1, None)
        A[0] = 0
        A[1:] = C
        C = A
        D = Qu64(C[-1], None)
        hsh = LoL()
        hsh.load(D, C)
        counters = Qu32(XNYN, 0)
        for n in range(len(seg_buffer)):
            if n%100000==0: print(n,'/',len(seg_buffer))
            H = seg_buffer[n]
            c = code[n]
            for h in H:
                if h==XNYN: break
                hsh.data[hsh.offsets[h]+counters[h]] = c
                counters[h] += 1
        self.hashed[name] = hsh
        return hsh
    def find_one_line_no_shapely(self,point,name_s,d_max,X,Y,lines,test_lines=None,_scale_test=0.146,dist_mem=None):
        if type(name_s) is str: names = [name_s]
        else: names = name_s
        if dist_mem==None: dist_mem = {name:{} for name in names}
        out_L = None
        x,y = point
        POINT = Point(point)
        d_test = self.res*_scale_test
        r = min(d_max, d_test)
        H = self._hash_rectangle((x-r,y-r),(x+r,y+r))
        if H is not None:
            for name in names:
                hashed = self.hashed[name]
                dist = dist_mem[name]
                for h in H:
                    for L in set((hashed[h]>>32).astype('uint32')): # New implementation: line hashes index line<<32+segment in uint64
                        if L in dist: d = dist[L]
                        else:
                            d = LineString(list(zip(X[lines[L]],Y[lines[L]]))).distance(POINT)
                            dist[L] = d
                        if d<r and (test_lines is None or test_lines[L]):
                            r = d
                            out_L = L
                            out_name = name
        if out_L is None:
            if d_test>=d_max: return None
            return self.find_one_line_no_shapely(point,name_s,d_max,X,Y,lines,test_lines,1.5*_scale_test+0.281,dist_mem)
        if type(name_s) is str: return out_L
        else: return (out_name, out_L)
        
        

                
                
            
        

        
######        #print("hsh[H].append(N)")
######        hsh = [[] for _ in Qr(self.XN*self.YN)]
######        for N,L in enumerate(lines):
######            for n in Qr(len(L)-1):
######                for H in self._hash_rectangle(L[n],L[n+1]):
######                    hsh[H].append(N)
######        #print("self.geometry[name]")
######        self.geometry[name] = [LineString(L) for L in lines]
######        print(name,'total line length:', sum(LS.length for LS in self.geometry[name]))
######        #print("self.hashed[name]")
######        self.hashed[name] = LoL(hsh,deepctype(len(lines)))







##def grid_hash(xx, yy, res):
##    return "Deprecated!"
##    N = len(xx)
##    minx = min(xx)
##    maxx = max(xx)
##    miny = min(yy)
##    maxy = max(yy)
##    NX = int((maxx-minx)/res) + 1   
##    NY = int((maxy-miny)/res) + 1
##    HASH = [int((xx[n]-minx)/res)+NX*int((yy[n]-miny)/res) for n in Qr(N)]
##    out = {n:[] for n in Qr(NX*NY)}
##    for i,H in enumerate(HASH): out[H].append(i)
##    return out




