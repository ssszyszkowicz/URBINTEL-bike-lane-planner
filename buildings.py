from utils import *
from qcommands import *
from geometry2 import *
import osmfilter


OSM_BUILDING_TYPE = {'R':'R','W':'W','C':'W','P':'W','S':'?','0':'W','?':'?','X':'?'}
def parse_OSM_buildings(OSM_DAT, const_h_m):
    node = {'x':OSM_DAT.node_x['building'],
            'y':OSM_DAT.node_y['building']}
    pde = OSM_DAT.props_decoder
    wl = OSM_DAT.way_lines['building']
    wp = OSM_DAT.way_props['building']
    XY = {}
    for c in 'xy':
        C = LoL()
        C.data = node[c][wl.data]
        C.offsets = wl.offsets
        C.dtype = node[c].dtype
        C.datalen = C.offsets[-1]
        C.N = len(C.offsets)-1
        XY[c] = C
    TYPE = []
    HEIGHT = []
    first_floor_m = const_h_m['first_floor_m']
    next_floor_m = const_h_m['next_floor_m']
    for b in wp:
        bd = pde(b)
        TYPE.append(OSM_BUILDING_TYPE[osmfilter.building_type(bd)])
        try:
            height_unit = osmfilter.building_height_unit(bd)
            if height_unit[1] == None: h = first_floor_m
            elif height_unit[1] == 'm': h = height_unit[0]
            elif height_unit[1] == 'floors': h = \
                first_floor_m+(height_unit[0]-1)*next_floor_m
            if h<first_floor_m: h = first_floor_m
        except: h = first_floor_m
        HEIGHT.append(h)
    return ({'TYPE':TYPE, 'HEIGHT':HEIGHT},XY)
    








### This is all temporary and should be a global config - this is an auto-detect hack.
SHP_BUILDING_SEMANTICS = {'TYPE_COLUMN':['type','buildingty'],
'AREA_COLUMN':['area','shape_area','shape__are'],
'HEIGHT_COLUMN':['height', 'buildinghe'],                      
'TYPES':str_to_inv_dict("""
R multifamily residential apartment senior mobile residential-duplex residential-house 1 
W school commercial medical hotel office industrial institutional church community arena 3 4 5 6 7 8 9 10 11 12 13
? unknown general miscellaneous 2""")}
SHP_BUILDING_SEMANTICS['TYPES'].update({'fire station':'W',
                                    'residential-low rise':'R',
                                    'residential-high rise':'R'})



def get_building_column(fields, prop, BUILDING_SEMANTICS=SHP_BUILDING_SEMANTICS):
    TCs = BUILDING_SEMANTICS[prop+'_COLUMN']
    for f in fields:
        if f.lower() in TCs: return f
    return None

def get_building_type(type_list, BUILDING_SEMANTICS=SHP_BUILDING_SEMANTICS):
    Ts = BUILDING_SEMANTICS['TYPES']
    return [Ts.get(str(ty).lower(),'?') for ty in type_list]

def parse_buildings(table, const_h_m, BUILDING_SEMANTICS=SHP_BUILDING_SEMANTICS):
    COLUMN = {prop:get_building_column(table.head,prop) for prop in 'TYPE AREA HEIGHT'.split()}
    for prop in COLUMN:
        if COLUMN[prop] is None:
            print('Could not find property',prop,'in building data.')
            return False
    DATA = {'TYPE': get_building_type(table[COLUMN['TYPE']])}
    area = table[COLUMN['AREA']]
    DATA['AREA'] = area
    height = table[COLUMN['HEIGHT']]
    DATA['HEIGHT'] = height
    default_h_m = const_h_m['first_floor_m']
    DATA['VOLUME'] = [area[n]*(height[n] if (height[n] and height[n]>0) \
                               else default_h_m) for n in Qr(area)]
    return DATA

def assign_to_polygons(buildings, polygons): #Geometry4, Geometry4
    b_frames = buildings.lrbt_frames('m')
    p_frames = polygons.lrbt_frames('m')
    b_shapely = [[sPolygon(contour) for contour in polygon] for polygon in buildings.xy()]
    p_shapely = [[sPolygon(contour) for contour in polygon] for polygon in polygons.xy()]
    b_intersects = [[] for b in Qr(b_frames)]
    p_intersects = [[] for p in Qr(p_frames)]
    for P in Qr(p_frames):
        print('Parcel',P,'/',len(p_frames))
        for B in Qr(b_frames):
            if lrbt_and_lrbt(p_frames[P], b_frames[B]):
                intersects = False
                for bC in b_shapely[B]:
                    for pC in p_shapely[P]:
                        if bC.intersects(pC):
                            intersects = True
                            break
                    if intersects: break
                if intersects:
                    b_intersects[B].append(P)
                    p_intersects[P].append(B)
    histo([len(x) for x in b_intersects])
    return (b_intersects,p_intersects)
                    
        
    
                    
            


