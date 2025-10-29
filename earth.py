from utils import *
from qcommands import *
from pyproj import Proj as pyproj_Proj
from pyproj import datadir # fixes pyproj compile
import warnings, math
# class Geometry goals:
# have types: polygons, polylines, points
# have graphical manifestations:
#   - polygon: triangulation
#   - polyline: rectangles - needs road thickness, or will be drawn as "thin_line"
#   - points - nothing - maybe a thickness as well for drawing?
# optional data attached -already done!
# How to handle indexing on joint geometries from shp?
#   - expand Table columns and have an index counter to keep track of jointness if necessary
# Shapefiles should try to expand to load as G3, not G4

# Earth radii based on Wikipedia "Earth" 
geoVCm = 40007860 # sic, add about 7.86km.
geoHCm = 40075017 # 40kkm +75km +17m
    
def _deg2m(centre, lon_a, lat_a):
    x = Qf32((lon_a-centre[0])*(geoHCm/360)*np_cos(lat_a*(pi/180)))
    y = Qf32((lat_a-centre[1])*(geoVCm/360))
    return x,y

def _m2deg(centre, x_a, y_a):
    lat = Qf64(y_a)/(geoVCm/360) + centre[1]
    lon = Qf64(x_a)/(np_cos(lat*(pi/180))*(geoHCm/360)) + centre[0]
    return lon, lat

def Mercator_projection(lat_a):
    lat_rad = np_radians(lat_a)
    # To avoid infinities at the poles, clip latitude
    lat_rad_clipped = np_clip(lat_rad, -np_pi/2 + 1e-10, np_pi/2 - 1e-10)
    y_a = np_log(np_tan(np_pi/4 + lat_rad_clipped/2))/np_pi*180 #to have as many units as lon
    return y_a

def Mercator_inv_projection(y_a):
    lat_a = np_degrees(2*(np_atan(np_exp(y_a/180*np_pi))-np_pi/4))
    return lat_a

def earth_distance_m(lon1, lat1, lon2, lat2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    radius_m = 6371000
    distance = radius_m * c
    return distance

def buffer_lrbt_deg(lrbt_deg, buffer_m):
    l,r,b,t = lrbt_deg
    dy = buffer_m/111111
    top = t + dy
    bottom = b - dy
    dx = buffer_m/(111111*min([math.cos(a*0.01745329251) for a in [top,bottom]]))
    return [l-dx, r+dx, bottom, top]
        
    


# Encodes lat/lon in millionth of a degree (approx 1dm)
def encode_64_4(c):
    offset = 8.0
    if abs(c)>offset: Qe("Value too large in encode_64_4")
    i = int((c+offset)*10**6)
    out = ""
    for n in Qr(4):
        out += chr(48+i%64)
        i = i//64
    return out
def encode_geolist_64_4(G23, centre):
    def G2644(L, centre):
        out = ""
        x,y = centre
        for P in L:
            out += encode_64_4(P[0]-x)
            out += encode_64_4(P[1]-y)
        return out
    D = get_depth(G23)
    out = "G"+str(D)
    if D==2: return out+G2644(G23, centre)
    if D==3: return out+' '.join(G2644(g, centre) for g in G23)
    Qe("bad depth in encode_geolist_64_4!")


# check my code: what does "GPS_PRJ_TEXT" do and why do I need to add it in EPSG_CODES ? 
#a.k.a. EPSG4326:
GPS_PRJ_TEXT = """GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]"""
## always_xy=True - fixes warning flood
EPSG_CODES = {'NAD_1983_UTM_Zone_10N':26910,
              'NAD83_CSRS_UTM_zone_10N':26910,
              'NAD_1983_CSRS_UTM_Zone_10N':26910,
              'NAD_1983_UTM_Zone_17N':26917,
              'PCS_Lambert_Conformal_Conic':3347,
              'GCS_North_American_1983':4269,
              'GCS_WGS_1984': 4326,
              'WGS_1984_Web_Mercator_Auxiliary_Sphere':3857}
def Canada2deg(points):  
    warnings.filterwarnings("ignore")
    P = pyproj_Proj(init='epsg:'+str(EPSG_CODES['PCS_Lambert_Conformal_Conic']))
    #print('Transformed via coordinate system: NAD83(epsg:3347)')
    R = tuple(P(*p,inverse=True) for p in points)
    warnings.filterwarnings("default")
    return R
def EPSG2deg(geometry, epsg):
    warnings.filterwarnings("ignore")
    P = pyproj_Proj(init='epsg:'+str(epsg))
    D = get_depth(geometry)
    if D == 1: R = tuple(P(*geometry,inverse=True))
    if D == 2: R = tuple(P(*p,inverse=True) for p in geometry)
    if D == 3: R = tuple(tuple(P(*p,inverse=True) for p in q) for q in geometry)
    if D == 4: R = tuple(tuple(tuple(P(*p,inverse=True) for p in q) for q in r) for r in geometry)
    warnings.filterwarnings("default")
    return R


##Worth investigating for Geometry:
##    - type: P, L, C
##    - shapely manifestation - Multi*
##    - metrics: L-(len_m,len_seg_m) - numpy-vec or opencl
##    -          C-areas; centroids (various)
##    - C: triangulation: try the Pandas3D function for speed
##    -       what is triangulation I/O format? Standardize (here)
##    - Universal buffer: taken from shapely, returned as G3:C(x-y)
##    - remove shape=4 (comment out) as shp(4) will be linearized to shp(3)

class Geometry:
    def __Qstr__(self=None):
        if self==None: return '*Geometry'
        s = '*Geo['+str(self.shape)
        if 'x' in self.data: s+='M'
        if 'lon' in self.data: s+='D'
        return s+'](%d)'%len(self)
    def lrbt_frames(self, unit):
        H,V = {'m':'x y'.split(), 'deg':'lon lat'.split()}[unit]
        if self.shape == 4:
            X = self.data[H]
            Y = self.data[V]
            return [(X[n].min(), X[n].max(), Y[n].min(), Y[n].max()) for n in Qr(self)] 
        Qer('Geometry.lrbt_frames() not implemented for shape',self.shape,'.')
    def get_lrbt(self,unit):
        if unit not in self.lrbt:
            H,V = {'m':'x y'.split(), 'deg':'lon lat'.split()}[unit]
            if self.shape == 1: lrbt = [[self.data['lon']]*2+[self.data['lat']]*2]
            elif self.shape == 2 or self.shape == 3:
                lrbt = [[self.data[H].min(),self.data[H].max(), self.data[V].min(), self.data[V].max()]]
            elif self.shape == 4:
                lrbt = [[self.data[H][n].min(),self.data[H][n].max(),
                         self.data[V][n].min(), self.data[V][n].max()] for n in Qr(self)]
            else: self.lrbt[unit] = False
            l,r,b,t = [[f[N] for f in lrbt if f[N] is not None] for N in Qr(4)]
            if not l: self.lrbt_deg = None
            else: self.lrbt[unit] = (min(l), max(r), min(b), max(t))
        return self.lrbt[unit]
    def get_trapez_m(self):
        if not hasattr(self,'centre'): return False
        lrbt = self.get_lrbt('deg')
        if not lrbt: return lrbt
        l,r,b,t = lrbt
        centre = self.centre
        X,Y = _deg2m(centre, Qf64([l,r,r,l]), Qf64([b,b,t,t]))
        return tuple(zip(list(X),list(Y)))
    def _coords(self,H,V):
        if self.shape == 1: return (H,V)
        if self.shape == 2: return tuple(zip(H.tolist(),V.tolist()))
        if self.shape == 3:
            H = H.tolol()
            V = V.tolol()
            return tuple(tuple(zip(H[n],V[n])) for n in Qr(H))
        if self.shape == 4:
            H = [h.tolol() for h in H]
            V = [v.tolol() for v in V]
            return tuple(tuple(tuple(zip(H[n][m],V[n][m])) for m in Qr(H[n])) for n in Qr(H))
        Qer('Bad Geometry._coords')
    def _coord(self,C):
        if self.shape == 1: return C
        if self.shape == 2: return C.tolist()
        if self.shape == 3: return C.tolol()
        if self.shape == 4: return [c.tolol() for c in C]
    def lon(self): return self._coord(self.data['lon'])
    def lat(self): return self._coord(self.data['lat'])
    def x(self): return self._coord(self.data['x'])
    def y(self): return self._coord(self.data['y'])
    def lonlat(self): return self._coords(self.data['lon'],self.data['lat'])
    def xy(self): return self._coords(self.data['x'],self.data['y'])
    def raw(self, dim):
        D = self.data[dim]
        if self.shape == 2: return D.array() # 'List' megre all 1-d arrays.
        if self.shape == 3: return D.LoL() # 'ListOfLists' merge to 'LoL'
        Qer('Bad Geometry.raw()')
    def __call__(self, centre, delete = False):
        self.centre = centre
        S = Qmx('lon lat x y float32 float64')
        if 'x' not in self.data:
            Hi, Vi, Ho, Vo, DTo, DTi = S
            func = _deg2m
        elif 'lon' not in self.data:
            Ho, Vo, Hi, Vi, DTi, DTo = S
            func = _m2deg
        else: return self # already mounted.
        if self.shape == 1:
            h,v = func(centre, np_array([self.data[Hi]],DTi),np_array([self.data[Vi]],DTi))
            H,V = h.item(0),v.item(0)
        elif self.shape == 2:
            H = List(dtype=DTo); V = List(dtype=DTo)
            for n in Qr(self.data[Hi].data):
                h,v = func(centre, self.data[Hi].data[n], self.data[Vi].data[n])
                H.extend(h); V.extend(v)
        elif self.shape == 3:
            H = ListofLists(dtype=DTo); V = ListofLists(dtype=DTo)
            for n,L in enumerate(self.data[Hi].data):
                h = LoL(dtype=DTo);v = LoL(dtype=DTo)
                h(L);v(L)
                h.data,v.data = func(centre, self.data[Hi].data[n].data, self.data[Vi].data[n].data)
                H.extend(h); V.extend(v)
        elif self.shape == 4:
            H = [];V = [] # type: L(ListofLists)
            for m in Qr(self.data[Hi]):
                HH = ListofLists(dtype=DTo); VV = ListofLists(dtype=DTo)
                for n,L in enumerate(self.data[Hi][m].data):
                    h = LoL(dtype=DTo);v = LoL(dtype=DTo)
                    h(L);v(L)
                    h.data,v.data = func(centre, self.data[Hi][m].data[n].data, self.data[Vi][m].data[n].data)
                    HH.extend(h); VV.extend(v)
                H.append(HH); V.append(VV)        
        else: Qer('Bad Geometry.shape =',self.shape)
        self.data[Ho] = H; self.data[Vo] = V
        if delete:
            del self.data[Hi]
            del self.data[Vi]
        return self
    def __len__(self): return self.len
    def __getitem__(self,key): return self.optional[key]
    def get(self, key): return self.optional.get(key) 
    def __setitem__(self,key, item): self.optional[key] = item
    def __contains__(self, item): return item in self.optional
    def __init__(self, G=None, G1=None, form='lonlat'):
        self.optional = {}
        self.centre = None
        self.lrbt = {}
##        if len(G) == 0:
##            self.data = {'lon':[],'lat':[],'x':[],'y':[]}
##            self.shape = 1
##            return None
        self.data = {}
        if form == 'lonlat': H='lon';V='lat';DT='float64'
        if form == 'xy': H='x';V='y';DT='float32'
        if G1 is not None:    
            pt = get_depth(G)
            pt1 = get_depth(G1)
            if pt != pt: return False
            self.shape = pt+1
            if  pt == 0:
                self.data[H] = float(G)
                self.data[V] = float(G1)
            elif pt == 1:
                self.data[H] = List(G,dtype=DT)
                self.data[V] = List(G1,dtype=DT)
            elif pt == 2:
                self.data[H] = ListofLists(G,dtype=DT)
                self.data[V] = ListofLists(G1,dtype=DT)
            elif pt == 3:
                self.data[H] = [ListofLists(MP,dtype=DT) for MP in G]
                self.data[V] = [ListofLists(MP,dtype=DT) for MP in G1]           
            else: Qer('Bad Geometry.shape =',self.shape)
        else:
            pt = get_depth(G)
            self.shape = pt
            if pt == 1:
                self.data[H] = float(G[0])
                self.data[V] = float(G[1])
            elif pt == 2:
                self.data[H] = List(tuple(p[0] for p in G),dtype=DT)
                self.data[V] = List(tuple(p[1] for p in G),dtype=DT)
            elif pt == 3:
                self.data[H] = ListofLists(tuple(tuple(p[0] for p in P) for P in G),dtype=DT)
                self.data[V] = ListofLists(tuple(tuple(p[1] for p in P) for P in G),dtype=DT)
            elif pt == 4:
                self.data[H] = [ListofLists(tuple(tuple(p[0] for p in P) for P in MP),dtype=DT) for MP in G]
                self.data[V] = [ListofLists(tuple(tuple(p[1] for p in P) for P in MP),dtype=DT) for MP in G]      
            else: Qer('Bad Geometry.shape =',self.shape)
        if self.shape != 1: self.len = len(self.data[H])
        if self.shape == 3: self.data[V].data[0](self.data[H].data[0])
        if self.shape == 4:
            for n in Qr(self): self.data[V][n].data[0](self.data[H][n].data[0])     
# G4 is not a very efficient compression method - data size estimates per x/y/lon/lat point:
# python list: 240 bytes
# G4: 80 bytes
# C-numpy marginal: 24 bytes




## LAMBERT PROJECTION & CANADA FED DATA
## From 'dissemination_area' documentation
## p.25 (Canada open data, rsrc: lda_000a16a_e)
##
##Projection: Lambert conformal conic
## ? False easting: 6200000.000000
## ? False northing: 3000000.000000
## lambda0 Central meridian: -91.866667
## phi1:     Standard parallel 1: 49.000000
## phi2:     Standard parallel 2: 77.000000
## phi0? (aka ref. lat.) ==?== Latitude of origin: 63.390675
## Linear unit: metre (1.000000)
## Datum: North American 1983 (NAD83)
## Prime meridian: Greenwich
## Angular unit: degree
## Spheroid: GRS 1980
############### This is identical to 'NAD83' #####################
########## Code is epsg:3347 for "Statistics Canada Lambert" #####
##Canada_Fed_Lambert = {'phi0':63.390675,
##                      'phi1':49.0,
##                      'phi2':77.0,
##                      'lambda0':-91.866667,
##                      'fE':6200000.0,
##                      'fN':3000000.0}

### Not needed for Winter2019 - use Canada2deg
##def TODEBUG___Lambert2deg(points, phi0, phi1, phi2, lambda0, fE, fN):
##    TO_DO_!!! Clean up imports into utils.py if this is ever re-used.
##    #import numpy
##    from math import log as ln
##    from math import sin, cos, tan, asin, acos, atan, pi
##    pideg = pi/180
##    if phi1==phi2: n = sin(phi1*pideg)
##    else: n = ln(cos(phi1*pideg)/cos(phi2*pideg))/ln(tan(pi/4+phi2*pideg/2)/tan(pi/4+phi1*pideg/2))
##    F = cos(phi1*pideg)*(tan(pi/4+phi1*pideg/2)**n)/n
##    rho0 = F*(tan(pi/4+phi0*pideg/2)**-n)
##    print(n,F,rho0)
##    x = tuple(p[0] for p in points)
##    y = tuple(p[1] for p in points)
##    L = len(x)
##    assert L == len(y)
##    lambda_ = tuple(atan(x[i]/fE/(rho0-y[i]/fN))/n/pideg + lambda0 for i in range(L))#divide by fE,fN? - my guess
##    print(lambda_)
##    phi = tuple(2*(atan((sin(n*(lambda_[i]-lambda0)*pideg)*F/n)**(1/n)))/pideg - 90 for i in range(L))
##    return tuple((lambda_[i],phi[i]) for i in range(L)) #lon, lat
##
##def deg2Lambert(points, phi0, phi1, phi2, lambda0, fE, fN):
##    print("earth.deg2Lambert not implemented! Hopefully I'll never need this monster)")
##    print("Conversion formula on Wikipedia 'Lambert projection'.")
##    pass



STREET_SUFFIXES_DICT = {
    'ave':'avenue',
    'av':'avenue',
    'blvd':'boulevard',
    'cres':'crescent',
    'crt':'court',
    'dr':'drive',
    'hwy':'highway',
    'ln':'lane',
    'pkwy':'parkway',
    'pl':'place',
    'priv':'private',
    'rd':'road',
    'st':'street',
    'terr':'terrace',
    'way':'way'}
STREET_SUFFIXES = set(v for k,v in STREET_SUFFIXES_DICT.items())

def merge_street_names(names):
    valid = set(name for name in names if not name.endswith('_?'))
    if len(valid) == 1: return list(valid)[0]
    else: return '?'
    
STREET_CONJUNCTIONS = [' & ',' - ',' to ']
def parse_street_name(name):
    s = name.lower()
    for sc in STREET_CONJUNCTIONS:
        if sc in s:
            return parse_street_name(s.split(sc)[0])
    s = s.split()
    if not s: return ''
    if s[0].isdigit(): return parse_street_name(' '.join(s[1:]))
    if s[-1] in 'bridge tunnel underpass overpass'.split(): s.pop()
    if not s: return ''
    card = 'south north west east'.split()
    dirs = {c[0]:c for c in card}
    for n in [0,1]:
        dirs.update({card[n][0]+c[0]:card[n]+c for c in card[2:]})
    dirs_long = set(v for k,v in dirs.items())
    DIR = None
    e = s[-1]
    if e in dirs_long: DIR = e
    elif e in dirs: DIR = dirs[e]
    if DIR: s.pop()
    if not s: return '' if DIR is None else DIR
    pre = {'st':'saint'}
    if len(s)>1 and s[0] in pre: s[0] = pre[s[0]]
    X = True
    if s[-1] in STREET_SUFFIXES_DICT:
        X = False
        s[-1] = STREET_SUFFIXES_DICT[s[-1]]
    if s[-1] in STREET_SUFFIXES: X = False
    s = ' '.join(s)
    if DIR: s += '_'+DIR
    if X: s += '_?'
    return s


def pretty_street_name(name, allow_higher_chars=True):
    if '!' in name: return ''
    f = name.find('@')
    if f>-1: name = name[:f]
    subs = [('_',' '),('?',' '),(chr(8217),"'"),(chr(8211),'_')]
    for a,b in subs:
        name = name.replace(a,b)
    w = name.split()
    words = []
    for v in w:
        if '=' not in v: words.append(v)
    lower = ' '.join(words)
    out = ''
    
    #if allow_higher_chars: return out
    for n,c in Qe(lower):
        if n == 0: c = c.upper()
        else:
            if not lower[n-1].isalpha(): c = c.upper()
        if not allow_higher_chars:
            if ord(c)>255: out+='?'
            else: out += c
        else: out += c
    return out
    
    
    

    
