from utils import *
from qcommands import *
import json
from fileformats import *
from data_loaders import *
from Canada import * # Change this line for different Country.
from shapely import Polygon as sPolygon
import numpy
CENSUS_POP = "C:/data/Canada/DAs/DApops2016.csv"
CENSUS_GEOMETRY = "C:/data/Canada/DAs/lda_000a16a_e.shp"

PROJECT = 'GVictoriaA'
VERSION = 20

if PROJECT == 'GVictoriaA':
    RES = {'AAACycling':'All_Ages_and_Abilities_Cycling_(OCP)',
           'Racks':'Bike_Racks',
           'Ferry':'Ferry_Routes',
           'Foot':'Footpaths',
           'TransitRoutes':'GTFS_routes'}
    COLOURS = {'AAACycling':'00FF00', 'Ferry':'007F7F',
               'Foot':'007F00', 'TransitRoutes':'00007F',
               'TransitStops':'00007F','Hospital':'BFBF00',
               'FireDepartment':'FF0000','Police':'FF7F00'}
    ZINDEX = {'AAACycling':1, 'Ferry':2, 'Foot':3, 'TransitRoutes':4,
              'TransitStops':5,'Hospital':6,'FireDepartment':7,'Police':8}
    ROOT = "C:/data/Greater Victoria Area"
    CITY = ['BC','Victoria']
    CENTRE = [-123.36, 48.43]
    POP_DENS_CEILING = 30000
    # Transit stops:
    with open(ROOT+'/Victoria_stops.csv', 'r') as f:
        F = [r for r in csv_reader(f)]
        F[0][0] = '#' # Table doesn't like empty fields
        T = Table(F)
        H = ['stop_code','stop_name']
        SC = T['stop_code']
        SN = T['stop_name']
        TS = {'field_names':H,
              'records':Table([H]+[(SC[n],SN[n]) for n in range(len(SC))])}
        LON = T['stop_lon']
        LAT = T['stop_lat']
        TS['points'] = [(LON[n],LAT[n]) for n in range(len(T))]
        TS['geometry'] = TS['points']
        TRANSIT_STOPS = TS
    # Markers:
    POLICE = [['Victoria Police Department',-123.359217,48.4310329],
              ['Oak Bay Police Department',-123.312818,48.4291914],
              ['Saanich Police Department',-123.3773469,48.4598321],
              ['Royal Canadian Mounted Police (RCMP)',-123.4949852,48.4479641]]
    FIRE = [['Victoria Fire Department Headquarters & Hall #1',-123.3548532,48.4258692],
            ['Victoria Fire Station #2',-123.367291,48.4174761],
            ['Victoria Fire Station #3',-123.3637079,48.4359524],
            ['Oak Bay Fire Department',-123.3130617,48.4292044]]
    HOSPITAL = [['Royal Jubilee Hospital',-123.327585,48.4327613],
                ['Victoria General Hospital',-123.4325793,48.4668184],
                ['Saanich Peninsula Hospital',-123.4098938,48.5957031]]
    MARKERS = {'Police':POLICE,'FireDepartment':FIRE,'Hospital':HOSPITAL}
    POINT_DATA = {'TransitStops':TRANSIT_STOPS}

JSD = "const markers = "+str(MARKERS)+';\n'





# Canada.py-specific code
POP_GEO = get_city_geometry(CITY[1], CITY[0].split(), [], CENSUS_GEOMETRY, 0)
PC = load_data(CENSUS_POP)

DA = PC['Geographic code']
POP = [(p if p else 0) for p in PC['Population, 2016']]
POP_COUNTS = {DA[n]:POP[n] for n in range(len(DA))}

pp = POP_GEO['parcels']
da = POP_GEO['parcel_codes']
areas = [[sPolygon(C).area/10e6 for C in parcel] for parcel in pp]
pop = []
dens = []
for n in range(len(da)):
    p = POP_COUNTS[da[n]]
    d = p/sum(areas[n])
    dens.append([d]*len(areas[n]))
    if len(areas[n])==1: pop.append(p)
    else: pop.append([a*d for a in areas[n]])

# unroll multiple part parcels:
dens = sum(dens,[])
pp = sum(pp,[])
areas = sum(areas,[])
pop = sum(pp,[])

ppll = []
def m2deg(centre, x_a, y_a):
    lat = numpy.array(y_a,'float64')/(40007860/360) + centre[1]
    lon = numpy.array(x_a,'float64')/(numpy.cos(lat*(numpy.pi/180))*(40075017/360)) + centre[0]
    return lon, lat
########def deg2m(centre, lon_a, lat_a):
########    x = (lon_a-centre[0])*(40075017/360)*numpy.cos(lat_a*(pi/180))
########    y = (lat_a-centre[1])*(40007860/360)
########    return x,y
for contour in pp:
    x = numpy.array([p[0] for p in contour],'float64')
    y = numpy.array([p[1] for p in contour],'float64')
    lon,lat = m2deg(POP_GEO['centre'],x,y)
    ppll.append([[(lon[n],lat[n]) for n in range(len(lon))]])

head = "Area_km2 Population Density".split()
body = [(areas[n],pop[n],dens[n]) for n in range(len(areas))]
POP_DENSITY = {'field_names':head,
               'records':Table([head]+body),
               'geometry':ppll,
               'polygons':ppll}


DATA = {k:load_data(ROOT+'/'+RES[k]+'/'+RES[k]+'.shp') for k,v in RES.items()}





DATA['PopulationDensity'] = POP_DENSITY
CHAR_FOR_INT = '0123456789abcdefghijklmnopqrstuvwxyz'
def to_string(n, base):
    if n < 0: return '-' + to_string(-n,base)
    if n < base:
        return CHAR_FOR_INT[n]
    return to_string(n // base, base) + CHAR_FOR_INT[n % base]
JSD += 'const pop_dens_v = \''
for v in dens:
    s = to_string(int((min(float(v if v else 0)/POP_DENS_CEILING,1)**0.5)*255),16)
    if len(s)==1: s = '0'+s
    JSD += s
JSD += '\';\n'

for k,v in list(DATA.items())+list(POINT_DATA.items()):
    print(k,RES.get(k))
    if type(v) is str: print(v)
    else:
        T = set(v.keys()) - set('field_names field_idxs geometry records'.split())
        T = str(list(T)[0])
        print(len(v['geometry']), T)
        for f in v['field_names']:
            F = v['records'][f]
            S = set(F)
            s = str(S)
            out = ' > '+f+'('+str(len(S))+'): '
            if len(s) < 40: out += s
            else: out += s[:40]+'...'
            print(out)
    print()



for k,v in POINT_DATA.items():
    geo = v['geometry']
    LINE = 'const '+k+'_p = \''
    for x,y in geo:
        LINE += to_string(int((x-CENTRE[0])*1e6+0.5), 36) + "," +\
              to_string(int((y-CENTRE[1])*1e6+0.5), 36)
        LINE +=";"
    if LINE[-1]==";": LINE = LINE[:-1]
    JSD += LINE+'\';\n'
    

for k,v in DATA.items():
    if 'polylines' in v or 'polygons' in v:
        geo = v['geometry']
        LINE = 'const '+k+'_p = \''
        for a, b in enumerate(geo):
            for c, d in enumerate(b):
                PL = ""
                for e in range(len(d)):
                    PL += to_string(int((geo[a][c][e][0]-CENTRE[0])*1e6+0.5), 36) + "," +\
                          to_string(int((geo[a][c][e][1]-CENTRE[1])*1e6+0.5), 36)
                    PL +=" "
                if PL[-1]==" ": PL = PL[:-1]
                PL += ";"
                LINE += PL
        if LINE[-1]==";": LINE = LINE[:-1]
        JSD += LINE+'\';\n'



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
    WORD_LIST = ' '.join(str(WORD_DICT[n]).replace(' ','_') for n in Qr(WORD_DICT))
    return WORD_LIST, WORD_IX

def unroll(geo):
    out = []
    n = 0
    for g in geo:
        out += [n]*len(g)
        n += 1
    return out


POPUPS = {}
for k,v in list(DATA.items())+list(POINT_DATA.items()):
    try:
        if 'polylines' in v or 'polygons': u = unroll(v['geometry'])
        else: u = list(range(len(v['geometry'])))
        POPUPS[k] = {}
        for f in v['field_names']:
            sv = k+'_'+f+'_v'
            si = k+'_'+f+'_i'
            POPUPS[k][f] = [sv, si]
            L, I = words2strix(v['records'][f])
            LINE = 'const '+sv+' = "'+L
            JSD += LINE+'";\n'
            LINE = 'const '+si+' = ['+','.join(str(I[n]) for n in u)
            JSD += LINE+'];\n'
    except: pass
JSD_NAME = PROJECT+'_data_v'+str(VERSION)+'.js'
with open('AUTO_JS/'+JSD_NAME, 'w') as file: file.write(JSD)


JSC = "    var geometries = {"+(",\n"+" "*22).join("'"+k+"':"+k+"_p" for k in RES if type(DATA[k]) is not str)+"};\n"
JSC += "    var geo_colours = {"+(",\n"+" "*23).join("'"+k+"':'#"+v+"'" for k,v in COLOURS.items())+"};\n"
JSC += "    var point_clouds = {"+(",\n"+" "*24).join("'"+k+"':"+k+"_p" for k in POINT_DATA)+"};\n"
JSC += "    var zIndex = {"+(",\n"+" "*18).join("'"+k+"':"+str(v) for k,v in ZINDEX.items())+"};\n"
JSC += "    var popups = {"
LINES = []
for k,v in POPUPS.items():
    LINE = "'"+k+"': {"
    PARTS = []
    for f,x in v.items():
        PARTS.append("'"+f+"':["+x[0]+','+x[1]+']')
    LINE += ','.join(PARTS)+"}"
    LINES.append(LINE)
JSC += (",\n"+" "*18).join(LINES)+"};\n"










JSC = """
<!DOCTYPE html>
<html lang="en">
<head>
    <base target="_top">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://www.urbintel.bike/onewebmedia/"""+JSD_NAME+"""\"></script>
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
    const centre = ["""+str(CENTRE[1])+", "+str(CENTRE[0])+"""];
    var layers = [];
    var layer_dict = {};
"""+JSC+"""
    const map = L.map("map",
                {
                    center: centre,
                    maxBounds: [[centre[0]-1,centre[1]-1],[centre[0]+1,centre[1]+1]],
                    zoom: 13
                });
    const osm = L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        {"attribution": "", "detectRetina": false, "maxNativeZoom": 20, "maxZoom": 20, "minZoom": 10, "noWrap": false, "opacity": 1, "subdomains": "abcd", "tms": false}
    ).addTo(map);
    for(let geo in zIndex) {
        map.createPane(geo + 'Pane');
        map.getPane(geo + 'Pane').style.zIndex = 400 + zIndex[geo];
    }
    map.createPane('PopulationDensityPane');
    map.getPane('PopulationDensityPane').style.zIndex = 350;
   
    for(let geo in geometries) {
   	var popup = popups[geo];
        var data = {};

        for (let key in popup) {
            var v = popup[key][0].split(' ');
            var i = popup[key][1];
            var datum = [];
            for (let n=0; n<i.length; n++) {
                if(i[n]>-1) datum.push(v[i[n]]);
                else datum.push('');
            }
            data[key] = datum;
        }
        var colour = "#0000FF";
        if(geo in geo_colours) colour = geo_colours[geo];
        var layer = L.layerGroup([]);
        layers.push(layer);
        layer_dict['<span style="color:'+colour+'"><b>'+geo+'</b></span>'] = layer;  
        var polylines = geometries[geo].split(';');
        for(let n=0;n<polylines.length;n++) {
            var pl_text = polylines[n];
            var PL = [];
            for(const pt_text of pl_text.split(' ')) {
                xy = pt_text.split(',');
                PL.push([centre[0]+parseInt(xy[1], 36)/1000000.0,
                         centre[1]+parseInt(xy[0], 36)/1000000.0]);
            }
            var text = '<span style="color:#0000FF"><b>'+geo+'</b></span>';
            if(geo in popups) {
                for (let k in popups[geo]) {
                    val = data[k][n];
                    text += '<br><b>'+k+'</b> '+val;
                }
            }
            layer.addLayer(L.polyline(PL, {
                color: colour,
                weight: 2,
                pane: geo + 'Pane'
            }).bindPopup(text));
        }
    }
    for(let cloud in point_clouds) {
        var colour = "#0000FF";
        if(cloud in geo_colours) colour = geo_colours[cloud];
        var layer = L.layerGroup([]);
        layers.push(layer);
        layer_dict['<span style="color:'+colour+'"><b>'+cloud+'</b></span>'] = layer;  
        for(const pt of point_clouds[cloud].split(';')) {
            var xy = pt.split(',');
            var point = ([centre[0]+parseInt(xy[1], 36)/1000000.0,
                          centre[1]+parseInt(xy[0], 36)/1000000.0]);
            var text = '<span style="color:#0000FF"><b>'+cloud+'</b></span>';
            if(geo in popups) {
                for (let k in popups[geo]) {
                    val = data[k][n];
                    text += '<br><b>'+k+'</b> '+val;
                }
            }
            layer.addLayer(L.circleMarker(point, {
                color: colour,
                radius: 3,
                pane: cloud + 'Pane'
            }).bindPopup(text));
        }
    }
    var layer = L.layerGroup([])
    layers.push(layer);
    var n = 0;
    for(const pl_text of PopulationDensity_p.split(';')) {
        var PL = [];
        for(const pt_text of pl_text.split(' ')) {
            xy = pt_text.split(',');
            PL.push([centre[0]+parseInt(xy[1], 36)/1000000.0,
                     centre[1]+parseInt(xy[0], 36)/1000000.0]);
        }
        var colour = pop_dens_v.substr(n*2, 2);
        colour += colour;
        colour = "#"+colour+"FF";
        layer.addLayer(L.polygon(PL, {
            fillColor: colour,
            weight: 0,
            pane: 'PopulationDensityPane'
        }));
        n++;
    }
    for(let m in markers) {
	var colour = "#FF0000";
        if(m in geo_colours) colour = geo_colours[m];
        var layer = L.layerGroup([]);
        layers.push(layer);
        layer_dict['<span style="color:'+colour+'"><b>'+m+'</b></span>'] = layer;
        for(const pt of markers[m]) {
            var point = [pt[2],pt[1]];
            layer.addLayer(L.circleMarker(point, {
                color: colour,
                radius: 5,
                pane: m + 'Pane'
            }).bindPopup(pt[0]));
        }
    }
    
    for(let layer of layers) layer.addTo(map);
    var layerControl = L.control.layers({}, layer_dict).addTo(map);
    layerControl.expand();
</script>
</body>
</html>
"""

with open('AUTO_JS/'+PROJECT+'_code.js', 'w') as file: file.write(JSC)
