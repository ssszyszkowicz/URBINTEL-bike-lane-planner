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
GOSMA = GLOBAL['NWK'].TRs.get_osm_attribute
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
    with open("C:/data/"+PROV+"/CA_"+PROV+"_"+CITY+".p03", 'rb') as f:
        Curves, Scores = pickle.load(f)
    






with open('ChPE_data.js', 'w') as file:
    file.write(OUT)



JS = """
<!DOCTYPE html>
<html lang="en">
<head>
    <base target="_top">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://www.urbintel.bike/onewebmedia/PG_data_v7.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <style>
        html, body {
            height: 100%;
            margin: 0;
        }
        .leaflet-container {
            height: 600px;
            width: 700px;
            max-width: 100%;
            max-height: 100%;
        }
    </style>
</head>
<body>
<div id="map" style="width: 700px; height: 600px;"></div>
<script>
    const values = pri_values;
    const legend = pri_legend;
    
    var index = {};
    var layers = [];
    var layer_dict = {};
    for (let i = 0; i < legend.length; i++) {
        index[legend[i][0]] = i;
        layers.push(L.layerGroup([]));
        layer_dict['<span style="color: '+legend[i][2]+'"><b>' + legend[i][1] + '</b></span>'] = layers[i];  
    }
    
    //const osm_id = osm_id_text.split(',');
    const centre = [53.91401782, -121.8170443];
    const polylines = polylines_text.split(';');
    const osm_id = osm_id_text.split(',');
    const names_list = names.split(' ');
    const highways_list = highways.split(' ');
    for (let i = 0; i < values.length; i++) {
        PL = [];
        for(const pt_text of polylines[i].split(' ')) {
            xy = pt_text.split(',');
            PL.push([centre[0]+parseInt(xy[1], 36)/1000000.0, 
                     centre[1]+parseInt(xy[0], 36)/1000000.0]);
        }
        var j = index[values[i]];
        var n = name_ix[i];
        var t = '<p>';
        if(n>-1) t = t + '<b>'+names_list[n].replace('_',' ')+'</b><br>';
        var h = highway_ix[i];
        if(h>-1) t = t + highways_list[h].replace('_',' ')+'<br>';
        var o = parseInt(osm_id[i],36).toString();
        t = t + 'OSM Id <b><a href="https://www.openstreetmap.org/way/'+o+'">'+o+'</a></b><br>';
        t = t + 'Lanes <b>'+lanes[i].toString()+'</b><br>';
        t = t + 'Speed Limit <b>'+kph[i].toString()+'</b><br>';
        if(oneway[i]>0) t = t + 'one way<br>';
        if(bridge[i]>0) t = t + 'bridge<br>';
        if(tunnel[i]>0) t = t + 'tunnel<br>';
        if(streetparking[i]>0) t = t + 'street parking<br>';
        t = t + '</p>';
        layers[j].addLayer(L.polyline(PL, {color: legend[j][2], weight:2}).bindPopup(t));
    }
    const map = L.map("map",
                {
                    center: [53.91,-122.77],
                    maxBounds: [[52.91,-123.77],[54.91,-121.77]],
                    zoom: 13,
                    layers: layers
                });
    const osm = L.tileLayer(
                "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                {"attribution": "\u0026copy; \u003ca href=\"https://www.openstreetmap.org/copyright\"\u003eOpenStreetMap\u003c/a\u003e contributors \u0026copy; \u003ca href=\"https://carto.com/attributions\"\u003eCARTO\u003c/a\u003e", "detectRetina": false, "maxNativeZoom": 20, "maxZoom": 20, "minZoom": 10, "noWrap": false, "opacity": 1, "subdomains": "abcd", "tms": false}
            ).addTo(map);
    const baseMaps = {};
    const overlayMaps = layer_dict;
    var layerControl = L.control.layers(baseMaps, overlayMaps).addTo(map);
    layerControl.expand();
</script>
</body>
</html>
"""



