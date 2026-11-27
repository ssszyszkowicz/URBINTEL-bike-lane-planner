from MAIN import safe_load_file, main_make_LTS, main_best_infra, main_planner_fixes__
from conflation import main_shp_bikeways
from osmxml import OSMData
from UrbNet import Network
from gGlobals import gGLOBAL as GLOBAL
from osmmap import OSM_Map
from utils import *
from qcommands import*
from earth import _m2deg
import pickle

CITY = 'Charlottetown'
PROV = 'PE'

GLOBAL['COMPUTE']['BUILD_ALL'](GLOBAL['COMPUTE'])
UI_DATA = safe_load_file("C:/data/"+PROV+"/CA_"+PROV+"_"+CITY+".uic",'uic')
OSM_DAT = OSMData()
OSM_DAT.import_ui(UI_DATA)
#OSM_DAT_city_mods(OSM_DAT, GLOBAL['GEOGRAPHY'])  ### REFACTOR!
CENTRE = UI_DATA['centre']
GLOBAL['MAP'] = OSM_Map(CENTRE)
GLOBAL['MAP'].load_data(OSM_DAT)
GLOBAL['NWK'] = Network(GLOBAL['MAP'],GLOBAL,build=False)
GLOBAL['NWK'].import_ui(UI_DATA)
if 0: # if bikeways .shp used
    #GLOBAL['PATHS'] = {'BIKEWAYS_SHP':"C:/data/City of Prince George/Bikeways/Cycle_Network_Existing_OCP_8383.shp"}
    #GLOBAL['BIKEWAYS_SHP'] = {'column':'NetworkTyp','1':[3,4],'3':[1]}
    #main_shp_bikeways(GLOBAL,'')
    pass
main_best_infra(GLOBAL,'')
main_make_LTS(GLOBAL,'')

L = GLOBAL['NWK'].TRs.lines
Xd, Yd = _m2deg(CENTRE,
    GLOBAL['NWK'].TRs.roads.node_x[L.data],
    GLOBAL['NWK'].TRs.roads.node_y[L.data])
X = LoL(); X.data = Xd; X(L)
Y = LoL(); Y.data = Yd; Y(L)

OUT = "const polylines_text = '"


CHAR_FOR_INT = '0123456789abcdefghijklmnopqrstuvwxyz'
def to_string(n, base):
    if n < 0: return '-' + to_string(-n,base)
    if n < base:
        return CHAR_FOR_INT[n]
    return to_string(n // base, base) + CHAR_FOR_INT[n % base]


lines = L
for n, p in Qe(lines):
    PL = ""
    for m in Qr(p):
        PL += to_string(int((X[n][m]-CENTRE[0])*1e6+0.5), 36) + "," +\
              to_string(int((Y[n][m]-CENTRE[1])*1e6+0.5), 36)
        if m<len(p)-1: PL +=" "
    if n<len(lines)-1: PL += ";"
    OUT += PL
OUT += "';\n"
LTS = GLOBAL['NWK'].TRs['vis_LTS']
OUT += "const lts_values = ["+','.join(str(min(x,5)) for x in LTS) + "];\n"
OUT += "const lts_legend = ["
LTS = set(LTS)
LAYERS = []
for layer in GLOBAL['LAYERS']['LTS']:
    if layer.value in LTS:
        L = "["+str(layer.value)+",'"+layer.legend_name+"',"
        rgb = [to_string(int(v*255),16) for v in layer.rgb]
        L += "'#"+''.join(v if len(v)==2 else '0'+v for v in rgb)+"']"
        LAYERS.append(L)
OUT += ','.join(LAYERS) + "];\n"
OUT += "const osm_id_text = '"
GOSMA = GLOBAL['NWK'].TRs.get_osm_attribute #function
OUT += ','.join(to_string(int(i),36) for i in GOSMA('osm_id'))+"';\n"

LANES = [int(n) for n in np_maximum(GOSMA('fwd_numLanes'), GOSMA('bwd_numLanes'))]
KPH = [int(n) for n in np_maximum(GOSMA('fwd_speedLimit_kph'), GOSMA('bwd_speedLimit_kph'))]
ONEWAY = GOSMA('oneway')
BRIDGE = Qbool(GOSMA('passageType')==ord('b'))
TUNNEL = Qbool(GOSMA('passageType')==ord('t'))
STREETPARKING = Qbool(GOSMA('fwd_streetParking')) | Qbool(GOSMA('bwd_streetParking'))
for var, x in [('lanes',LANES),
               ('kph',KPH),
               ('oneway',ONEWAY),
               ('bridge',BRIDGE),
               ('tunnel',TUNNEL),
               ('streetparking',STREETPARKING)]:
    OUT += "const " + var + " = ["
    OUT += ','.join(to_string(i,10) for i in x)+"];\n"



### !!! Have to map road index vs TR index
NAME = []
HIGHWAY = []
PD = GLOBAL['NWK'].TRs.roads.props_decoder
for P in GLOBAL['NWK'].TRs.roads.props:
    d = PD(P)
    NAME.append(d.get('name',''))
    HIGHWAY.append(d.get('highway',''))

def words2strix(LIST):
    WORD_DICT = {}
    WORD_IX = []
    for n in LIST:
        if not n:
            WORD_IX.append(-1)
        else:
            if n not in WORD_DICT:
                i = len(WORD_DICT)
                WORD_DICT[n] = i
                WORD_IX.append(i)
            else:
                WORD_IX.append(WORD_DICT[n])
    WORD_DICT = {v:k for k,v in WORD_DICT.items()}
    WORD_LIST = ' '.join(WORD_DICT[n].replace(' ','_') for n in Qr(WORD_DICT))
    return WORD_LIST, WORD_IX

NAME_LIST, NAME_IX = words2strix(NAME)
HIGHWAY_LIST, HIGHWAY_IX = words2strix(HIGHWAY)
# convert OSM indices to TR indices:
O = GLOBAL['NWK'].TRs.osm_road
NAME_IX = Qi32(NAME_IX)[O]
HIGHWAY_IX = Qi32(HIGHWAY_IX)[O]

OUT += "const names = \"" + NAME_LIST + "\";\n"
OUT += "const name_ix = ["
OUT += ','.join(to_string(i,10) for i in NAME_IX)+"];\n"
OUT += "const highways = \"" + HIGHWAY_LIST + "\";\n"
OUT += "const highway_ix = ["
OUT += ','.join(to_string(i,10) for i in HIGHWAY_IX)+"];\n"

C = GLOBAL['MAP'].centre
OUT += "const centre = [%f, %f];\n"%(float(C[0]),float(C[1]))


if 1: # priority info:
    with open("C:/data/"+PROV+"/CA_"+PROV+"_"+CITY+".pri", 'rb') as f:
        GLOBAL['NWK'].TRs['vis_PRIORITY'] = pickle.load(f)
    GLOBAL['GEOGRAPHY']['LOC'] = CITY
    main_planner_fixes__(GLOBAL, '')
    OUT += "const pri_values = ["+','.join(str(x) for x in GLOBAL['NWK'].TRs['vis_PRIORITY'])+"];\n"
    OUT += "const pri_legend = ["
    LAYERS = []
    for layer in GLOBAL['LAYERS']['vis_PRIORITY']:
        L = "["+str(layer.value)+",'"+layer.legend_name+"',"
        rgb = [to_string(int(v*255),16) for v in layer.rgb]
        L += "'#"+''.join(v if len(v)==2 else '0'+v for v in rgb)+"']"
        LAYERS.append(L)
    OUT += ','.join(LAYERS) + "];\n"
    #with open("C:/data/"+PROV+"/CA_"+PROV+"_"+CITY+".p03", 'rb') as f:
    #    Curves, Scores = pickle.load(f)
    

with open('ChPE_data.js', 'w') as file:
    file.write(OUT)
