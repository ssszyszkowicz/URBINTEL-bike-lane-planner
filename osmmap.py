from utils import *
from earth import Geometry, parse_street_name
import osmfilter as OF
from geometry2 import make_box, HashGrid, assign_holes_to_solids, boundary_sPolygons
from qcommands import *



#### Jul 11 2018 Memory Budget:
##OSM_Map memory size for Ottawa-Gat:
##MAP.buildings                     : 203.1MB  (658B /building)
##MAP.roads                         : 20.4 MB   (305B /road) - sometimes up to 69MB (1030B/road)
##MAP.nodes:
##    - x, y [fl32]                 : 11.5MB x2 (8B /node)
##    - lat, lon [fl64]             : 23MB x2 (16B /node)
##    - formerly:IXs [{int:int}]    : (346.8MB    (120B /node))
##    - now:IXs [IndexDict]         : 48.9MB (17B /node)
##    - IDs [ui64]	            : 22.9MB  (8B/node)
##                                  : total = 140.8MB i.e. 49B/node (down from 152)
##Python internal	(?)	    : 100~400MB


def roads_shp2osm(shp, name, way_ix, node_ix, subs, boundary=[]): # TO_DO Redo to reflect new format
    BOUNDARY = boundary_sPolygons(boundary)
    ways = {}
    nodes = {}
    records = shp['records']
    head = records.head
    node_count = node_ix
    if 'polylines' not in shp: return False
    for n,L in enumerate(shp['polylines']):
        if len(L) < 2: continue
        if BOUNDARY:
            LINE = LineString(L)
            IN = False
            for B in BOUNDARY:
                if B.intersects(LINE):
                    IN = True
                    break
            if not IN: continue
        way = {}
        way['.shp'] = name
        for i,v in enumerate(records[n]):
            if type(v) is not str or v.strip():
                if head[i] in subs: way[subs[head[i]]] = v
                else: way[head[i]] = v
        for i,ll in enumerate(L):
            nodes[-node_count-i-1] = ll
        l = len(L)
        way['_nodes'] = list(Qr(-node_count-1,-node_count-1-l,-1))
        node_count += l
        ways[-way_ix-n-1] = way
    return {'nodes':nodes, 'ways':ways, 'ixs':(len(ways),len(nodes))}

##
##
##class OSM_Nodes:
##    def __init__(self):
##        self.IDs = List(dtype='int64')
##        self.x = List(dtype='float32')
##        self.y = List(dtype='float32')
##        self.N = 0
##    def extend(self, ids, xs, ys):
##        self.IDs.extend(ids)
##        self.x.extend(xs)
##        self.y.extend(ys)
##        self.N = len(self.IDs)
##    def freeze(self):
##        self.IDs = self.IDs.array()
##        self.x = self.x.array()
##        self.y = self.y.array()
##        self.IXs = IndexDict(self.IDs)      
##    def __len__(self): return self.N
##    def get_xy(self,ID):
##        IX = self.IXs[ID]
##        return (self.x[IX], self.y[IX])

## TO_DO: Some of these properties are necessary, their absence breaks code: how to validate this?
## Maybe have checked all instances of 'TRs['***']' in .py code and validate those strings in a registry?
# Undirected properties:
OSM_ROADS_FILTERS = {'osm_id': (OF.osm_road_id, 'uint64'),
                     'streetName': (lambda w: parse_street_name(OF.street_name(w)), ''),
                     'has_name': (OF.has_name, 'uint8'),
                     'passageType': (OF.passage, 'uint8'),
                     'clampSpeed_mps': (OF.clamp_speed, 'float32'),
                     'clampSpeed_kph': (lambda w: 3.6*OF.clamp_speed(w), 'float32'),
                     'vis_clamp': (lambda w: min(5,int((3.6*OF.clamp_speed(w)+1)/3)), 'uint8'),
                     'hasHouses': (OF.hasHouses, 'int8'),
                     'is_residential': (OF.is_residential, 'int8'),
                     'is_service': (OF.is_service, 'int8'),
                     'has_duration': (lambda w: int(bool(OF.duration_s(w))), 'uint8'),
                     'oneway':(OF.oneway, 'int8'),
                     '__bicycle_road':(OF.__bicycle_road, 'int8'),
                     '__sidewalk':(OF.__sidewalk, 'int8'),
                     'bike_access':(OF.bike_access, 'int8'),
                     'motor_allowed':(OF.motor_allowed, 'int8'),
                     'cycle_type':(OF.cycle_type, 'int8'),
                     'is_area':(OF.is_area, 'int8'),
                     'is_upgradeable':(OF.is_upgradeable, 'int8'),
                     'BLANK':(lambda w:0, 'int8'),
# directed properties: fwd_, bwd_, equ_, vis_
                     'fwd_dur_s': (OF.duration_s, 'float32'),
                     'bwd_dur_s': (OF.duration_s, 'float32'),
                     'speedLimit_kph': (OF.get_maxspeed, 'float32', -128.0), 
                     'numLanes': (OF.get_lanes, 'float32', -128.0),
                     'force_dismount': (OF.force_dismount, 'int8', -128),
                     'osm_infra': (OF.bike_infra, 'int8', -128),
                     'streetParking': (OF.has_onstreet_parking, 'int8', -128)}




class OSM_Roads:
    def get_prop(self, n): return self.__getitem__(self, n)
    def __getitem__(self, n): # get an osm_way as dict
        return self.props_decoder(self.props[n])
    def __init__(self, node_x, node_y, lines, len_m, seg_len_m, props, props_decoder = lambda w:w):
        self.node_x = node_x
        self.node_y = node_y
        self.lines = lines
        self.len_m = len_m
        self.seg_len_m = seg_len_m
        self.props = props
        self.props_decoder = props_decoder
        self.property = {}
        for P in OSM_ROADS_FILTERS:
            for k in [P] if len(OSM_ROADS_FILTERS[P])==2 else [k+'_'+P for k in 'fwd bwd equ vis'.split()]:
                self.property[k] = []
    def extend_props(self, start=0, end=-1):
        print('OSM_Roads.extend_props()')
        for way in self.props[start:end]:
            way = self.props_decoder(way)
            for P,V in OSM_ROADS_FILTERS.items():
                if len(V) == 2: self.property[P].append(V[0](way))
                else:
                    F = V[0](way,1)
                    B = V[0](way,0)
                    E = int(F==B)
                    Vis = F if E else V[2]
                    for T in [(F,'fwd'), (B,'bwd'), (E,'equ'), (Vis,'vis')]:
                        self.property[T[1]+'_'+P].append(T[0])
    def freeze(self):
        print('OSM_Roads.freeze() for', len(self), 'roads.')
        for PP,V in OSM_ROADS_FILTERS.items():
            for P in [PP] if len(V)==2 else [k+'_'+PP for k in 'fwd bwd equ vis'.split()]:
                a = self.property[P]
                if V[1]:
                    numtype = 'bool' if P.startswith('equ_') else V[1]
                    a = np_array(a, numtype)
                self.property[P] = a
                if V[1] and (len(P)<4 or P[1:4] != 'wd_') and '_id' not in P:
                    print('Prop ',P+'['+numtype+']: ',end='')
                    print('{'+str(sorted_unique(self.property[P],', '))+'}')
    def __len__(self): return len(self.lines)
    def json(self):
        out = {'node_x':self.node_x.tolist(),
               'node_y':self.node_y.tolist(),
               'lines':lol_to_tuple(self.lines.tolol()),
               'offsets':self.lines.offsets.tolist(),
               'value':self.property['LTS'].tolist()}
        return out
    def search_props(self, s): # faulty!
        decoder = self.props_decoder
        for eway in self.props:
            for k,v in decoder(eway).items():
                if s in k or s in v:
                    print(eway)
                    input()
                    break
    def give_props(self):
        decoder = self.props_decoder
        return [decoder(eway) for eway in self.props]
 

## needs import re, Polygon (deprecate)
##class OSM_Buildings:
####    def get_centroids(self):
####        if not hasattr(self, 'centroids'):
####            self.centroids = tuple(s.centroid for s in self.shapely)
####        return self.centroids
####    def get_areas(self):# in m^2
####        if not hasattr(self, 'areas'):
####            self.areas = np_array(tuple(s.area for s in self.shapely),dtype='float32')
####        return self.shapely        
##    def get_shapely(self):
##        if not hasattr(self, 'shapely'):
##            self.shapely = tuple(sPolygon(tuple(self.nodes.get_xy(n) for n in p)) for p in self.polygons)
##        return self.shapely
##    def __init__(self, OSM, nodes):
##        self.nodes = nodes
##        # TO_DO:
##        # - ? implement landuse fully as separate feature
##        # - Proper code for relations: multi-part buildings, building:part, overlapping buildings
##        # - how to calculate volume, floors, overlap for complex buildings?
##        # - revise building type selection -> refactor into osmfilter.py
##        BTY = set(osmfilter.BUILDING_TYPES.keys())
##        if 0:#if 'landuse' in OSM['ways']:
##            LANDUSE = OSM['ways']['landuse']
##            LAND_BUILD_TYPES = set(OSM['way_types']['landuse'].keys()).intersection(BTY)
##            LANDUSE_AREAS = {ty:Polygon.Polygon() for ty in LAND_BUILD_TYPES}####import Polygon
##            self.landuse_nodes = OSM_Nodes(OSM['nodes']['landuse'], self.nodes.centre)
##        #else: LANDUSE = []; LANDUSE_AREAS = []; self.landuse_nodes = [];
##            for k in LANDUSE:
##                lu = LANDUSE[k]
##                ty = lu['landuse']
##                if ty in LAND_BUILD_TYPES:
##                    LANDUSE_AREAS[ty].addContour(tuple(self.landuse_nodes.get_xy(n) for n in lu["nodes"]))
##        BUILDINGS = OSM['ways']['building']
##        BUILDING_PROPS = [];self.areas = [];self.centroids = []
##        BUILDING_IDs = [];self.props = [];NODES_LIST=[]
##        for ID,B in BUILDINGS.items():         
##            BP = Polygon.Polygon(tuple(self.nodes.get_xy(n) for n in B["nodes"]))####import Polygon
##            self.areas.append(BP.area())
##            self.centroids.append(tuple(BP.center()))
##            if B['building'] == 'yes': # attempt to fill in unknown building type:
##                match = set(B.keys()).intersection(BTY)
##                if len(match) == 1: B['building'] = tuple(match)[0]
##                if len(match) > 1: pass#print('Ambiguous building: ',match,B)
##                if len(match) == 0:
##                    for ty in LANDUSE_AREAS:
##                        if BP.overlaps(LANDUSE_AREAS[ty]): match.add(ty)
##                    if len(match) == 1: B['building'] = tuple(match)[0]
##                    if len(match) > 1: pass#print('Polygon: Ambiguous building: ',match,B)
##                    if len(match) == 0:
##                        pass # some intelligence to guess building?
##            BUILDING_IDs.append(ID)
##            NODES_LIST.append(tuple(self.nodes.IXs[int(n)] for n in B['nodes']))
##            B['nodes'] = len(B['nodes']) # must come after reading nodes.IXs !
##            self.props.append(B) # must follow B['nodes'] = len(B['nodes'])
##        self.props = tuple(self.props)
##        self.N = len(self.props)
##        self.areas = Qf32(self.areas)
##        self.centroids = tuple(self.centroids)
##        self.polygons = LoLu32(NODES_LIST)
##        self.types = tuple(osmfilter.building_type(B) for B in self.props)
##        self.get_levels()
##    def __len__(self): return self.N
##    def get_levels(self):
##        if not hasattr(self,'levels'):
##            self.levels = np_array(tuple(osmfilter.get_building_levels(B) for B in self.props),'float32')
##        return self.levels
##    def to_refactor_later(self):
##        # 'building_data':
##        if 0: 
##            ##B_HEAT = {'R':-100,'W':0,'C':0.25,'P':0.5,'S':0.75,'X':0.85,'0':1,'?':100} # categories for JS
##            B_HEAT = {'R':-1,'W':0,'C':0,'P':0,'S':0,'X':0,'0':0,'?':1,None:1} # categories for GL drawing
##            data['building_data'] = [B_HEAT[k] for k in self.types]
##        if 0: # Building colour based on # of floors:
##            data['building_data'] = []
##            for _,B in BUILDINGS.items():
##                if 'building:levels' in B:
##                    try: L = float(B['building:levels'])
##                    except:
##                        print(B['building:levels'])
##                        L=10000
##                else: L = -111
##                data['building_data'].append(L/30)




            

class OSM_Map:
    def shp_roads(self):
        lon = self.road_nodes.lon
        lat = self.road_nodes.lat
        out = {'polylines':[[(lon[p],lat[p]) for p in road] for road in self.roads.lines],
               'field_names':['LTS'],
               'records':{'LTS':self.roads.lts}}
        return out
    def __contains__(self,key): return key in self.optional
    def __getitem__(self,key): return self.optional[key]
    def get(self,key): return self.optional.get(key)
    def __setitem__(self, key, geometry): self.mount(key, geometry, delete = False)#True - delete is breaking code right now
    def mount(self, key, geometry, delete = False):
        if isinstance(geometry, Geometry): # this is 'mount'
            geometry(self.centre, delete)
        elif hasattr(geometry, '__call__'): geometry(self.centre)
##            if 'lon' in geometry.data:
##                if del_lonlat:
##                    print('OSMMAP_MOUNT: Deleting lon/lat from:',key)
##                    for c in 'lon lat'.split():
##                        if c in geometry.data:
##                            del geometry.data[c] # save on memory
##                else:
##                    print('OSMMAP_MOUNT: Non-del mount:',key)
        self.optional[key] = geometry # do in two steps to make sure to avoid NoneType object.
    def __init__(self, centre):
        self.centre = centre
        self.optional = {}
        self.empty = True
    def load_data(self, OSM_DATA):
        if 'highway' not in OSM_DATA.tynames: return None
        print('Making OSM_Roads...')
        self.roads = OSM_Roads(OSM_DATA.node_x['highway'],
                               OSM_DATA.node_y['highway'],
                               OSM_DATA.way_lines['highway'],
                               OSM_DATA.len_m['highway'],
                               OSM_DATA.seg_len_m['highway'],
                               OSM_DATA.way_props['highway'],
                               OSM_DATA.props_decoder)
        BLOCK_SIZE = 10000
        N = len(OSM_DATA.way_props['highway'])
        print('Loaded %d OSM roads.'%N)
        for n in Qr(1+(N-1)//BLOCK_SIZE):
            self.roads.extend_props(BLOCK_SIZE*n, min(N, BLOCK_SIZE*(n+1)))
        self.roads.freeze()
        if len(self.roads) == 0: return None
        print('OSM_Map ready!')
        self.empty = False
    def json(self, pad = [100,100]):
###        def coord(C): return np_around(C,decimals=2).tolist()
        data = {'centre':tuple(self.centre)}
####        xpad,ypad = pad # meters border around map
####        data['lrbt_m'] = (self.lrbt_m[0]-xpad, self.lrbt_m[1]+xpad,
####                          self.lrbt_m[2]-ypad, self.lrbt_m[3]+ypad)
##        if 1:#hasattr(self, 'roads'):
##            data['road_offsets'] = self.roads.lines.offsets
##            data['road_len'] = self.roads.lines.lens
##            data['road_x'] = self.roads.x.data
##            data['road_y'] = self.roads.y.data
##            if self.roads.props_decoder: data['roads_decoder'] = self.roads.props_decoder
##            data['road_seg_len_m'] = self.roads.seg_len_m.data
##            data['road_data'] = self.roads.lts
##            data['road_width'] = 5
            
##        if hasattr(self, 'lrbt_view'): data['lrbt_view'] = self.lrbt_view
##        if hasattr(self, 'buildings'):
##            data['buildings'] = self.buildings.polygons#.tolol()
##            data['building_x'] = coord(self.building_nodes.x)
##            data['building_y'] = coord(self.building_nodes.y)
##        if hasattr(self, 'water'):
##            data['water'] = self.water.polygons
##            data['water_x'] = coord(self.water.nodes.x)
##            data['water_y'] = coord(self.water.nodes.y)
        return data



##class OSM_Water:
##    def __init__(self, OSM, centre):
##        self.centre = centre
##        USED_NODES = set()
##        USED_RNODES = set()
##        POLYGONS = []
##        NAT = OSM['ways'].get('natural')
##        NODES = OSM['nodes'].get('natural')
##        REL = OSM['relations'].get('natural')
##        RNODES = OSM['nodes'].get('$RELATION$')
##        RWAYS = OSM['ways'].get('$RELATION$')
##        if NAT and NODES:
##            for _,nat in NAT.items():
##                if nat['natural'] == 'water':
##                    nodes = nat['nodes']
##                    USED_NODES |= set(nodes)
##                    POLYGONS.append([nodes])
##        if REL and RWAYS and RNODES:
##            for _,nat in REL.items():
##                if nat['natural'] == 'water':
##                    if len(nat['outer']) == 1:
##                        P = []
##                        for W in nat['outer']+nat['inner']:
##                            nodes = RWAYS[W]['nodes']
##                            USED_RNODES |= set(nodes)
##                            P.append(nodes)
##                        POLYGONS.append(P)
##                    else:
##                        H2S = assign_holes_to_solids([[RNODES[n] for n in RWAYS[W]['nodes']] for W in nat['outer']],
##                                                     [[RNODES[n] for n in RWAYS[W]['nodes']] for W in nat['inner']])
##                        
##                        for n,h2s in enumerate(H2S):
##                            for W in [nat['outer'][n]]+[nat['inner'][i] for i in h2s]:
##                                nodes = RWAYS[W]['nodes']
##                                USED_RNODES |= set(nodes)
##                                P.append(nodes)
##                            POLYGONS.append(P)   
##        N = {n:NODES[n] for n in USED_NODES}
##        N.update({n:RNODES[n] for n in USED_RNODES})
##        self.nodes = OSM_Nodes(N, self.centre)
##        IXs = self.nodes.IXs
##        self.polygons = tuple(tuple(tuple(IXs[n] for n in C) for C in P) for P in POLYGONS)
##    

