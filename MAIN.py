ITERATIONS = 3 # Number of times planning solution is refined. Should be at least 3.

PLACES = [["ON","Pickering"], ["BC","Nanaimo"], ["BC","Kamloops"], ["BC","Prince George"],
          ["BC","Vancouver"], ["PE","Charlottetown"], ["QC","Sherbrooke"],
          ["QC","Québec"], ["ON","Ottawa"], ["ON","Toronto"],
          ["ON","Peel"], ["ON","Mississauga"], ["ON","Brampton"], ["ON","Caledon"]]
CHOICE = 5



from conflation import main_shp_bikeways
from study import main_rw_study, main_seg_study
from utils import *
import pickle, json
import interactive, osmmap, osmxml, osmfilter, UrbNet, buildings, formulae
from drawpdf import draw_pdf as drawpdf_draw_pdf
from fileformats import *
from geometry2 import *
from study import *
from data_loaders import *
from earth import *
from popdens import PopDens, pop_geo_remake_geometry
from topography import Topography
from Canada import * # Change this line for different Country.
from International import *
from colorsys import hsv_to_rgb
from gGlobals import gGLOBAL


def safe_load_file(FILE, FORMATS):
    formats = FORMATS.split()
    ext = os_path_splitext(FILE)[-1].strip('.').lower()
    if not os_path_isfile(FILE):
        print('File not found:',FILE)
        END()
    if ext not in formats:
        print('.'+ext+' is not a valid data format here.')
        END()
    D = load_data(FILE)
    if D is False:
        print('Could not read data from:',FILE)
        END()
    return D

# Kamloops LRBT_m_DRAW doesn't work properly!
CITY_CONFIGS = {
'QC_Sherbrooke':"""
###VIEW                  683 715  0.06538 -15793 6159
BIKEWAYS_SHP          Ville de Sherbrooke/Pistes_cyclables/Pistes_cyclables.shp
BIKEWAYS_SHP_COLUMN   TYPEVOIE
BIKEWAYS_SHP_1        ChaussÃ©e dÃ©signÃ©e
BIKEWAYS_SHP_3        Bande cyclable unidirectionnelle
BIKEWAYS_SHP_4        Piste cyclable bidirectionnelle;Piste cyclable unidirectionnelle

###BUILDINGS             Ville de Sherbrooke/Batiments/Batiments.shp""",
'ON_Ottawa':"""
TRANSIT_DOWNTOWN    -75.6968 45.4125 2000
TRANSIT_SUBURB      -75.7264 45.2717 3000
TRANSIT_SUBURB      -75.9138 45.3015 5000
""",
'BC_Nanaimo':"""
VIEW                  683 715  0.06538 -15793 6159
BIKEWAYS_SHP          City of Nanaimo/Bike Routes/BIKE_ROUTES.shp
BIKEWAYS_SHP_COLUMN   BIKEROUTE
BIKEWAYS_SHP_0        Proposed
BIKEWAYS_SHP_3        Current
###BUILDINGS             City of Nanaimo/Cadastre/BUILDINGS.shp
###CITY_BOUNDARY         City of Nanaimo/City Boundary/cityboundary.shp
###WORKPLACES            City of Nanaimo/WorkplaceDensity.csv
###WORKPLACES_LON        Longitude
###WORKPLACES_LAT        Latitude
###WORKPLACES_COUNT      Employees
LRBT_m_DRAW           5000 26000 -16000 5000""",
'BC_Kamloops':"""
BIKEWAYS_SHP          City of Kamloops/Trails_and_Bikeways/Trails_and_Bikeways.shp
BIKEWAYS_SHP_COLUMN   TYPE
BIKEWAYS_SHP_0        PEDESTRIAN SHOULDER
BIKEWAYS_SHP_1        SIGNED ROUTE;MARKED WIDE CURB LANE
BIKEWAYS_SHP_3        BIKE LANE
BIKEWAYS_SHP_4        TRAIL;MULTI USE
###CITY_BOUNDARY         City of Kamloops/City_Area/City_Area.shp
###BUILDINGS             City of Kamloops/Building/Building.shp
LRBT_m_DRAW           -4600 33400 -23700 7300""",
'BC_Prince_George':"""
###CITY_BOUNDARY         City of Prince George/Boundary/Boundary.shp  
BIKEWAYS_SHP          City of Prince George/Bikeways/Cycle_Network_Existing_OCP_8383.shp
BIKEWAYS_SHP_COLUMN   NetworkTyp
BIKEWAYS_SHP_1        3;4
BIKEWAYS_SHP_3        1
###BUILDINGS             City of Prince George/Building/flat.shp
LRBT_m_DRAW           -70000 -55000 -14000 13000"""}

def load_config(FILE_CITY_NAME): # - note: some DAs incomplete in 2021 census! Using 2016 census.
    CONFIG = "ROOT_PATH C:/data/"+\
    """
    CENSUS            Canada/DAs/DisseminationAreas2016_proj_-3017378975245062340.csv
    CENSUS_GEOMETRY   Canada/DAs/lda_000b16a_e.shp
    """
    EXTRA = CITY_CONFIGS.get(FILE_CITY_NAME)
    if EXTRA: CONFIG += EXTRA
    return CONFIG
    
def END(string=''):
    if string: print(string)
    print("Press ENTER to end application...")
    input()
    sys_exit()







def MAIN(COUN_PROV_CITY, RUN='11111111', ITERATIONS=ITERATIONS, GLOBAL = gGLOBAL):
    GLOBAL['ALGORITHM']['iterations'] = ITERATIONS
    FILE_CITY_NAME = file_city_name(*COUN_PROV_CITY)
    CONFIG = parse_config(load_config(FILE_CITY_NAME))
    RUN_TOPOG, RUN_SLOPE, RUN_POP_DENS, RUN_SHP_BIKEWAYS, RUN_RW_SIM, RUN_SEG_SIM, RUN_RW_CURVE, RUN_INTERACTIVE = [int(d) for d in RUN]
    RUN_SIM = RUN_RW_SIM or RUN_SEG_SIM or RUN_RW_CURVE
    print(NOW())
    ALLV = """PATHS POP_GEO GEOGRAPHY COMPUTE ALGORITHM.MAP_HASH_GRID_m 
    MAP>roads.give_props<property<props<props_decoder NWK.TRs>osm_road"""
    Q.default(GLOBAL, ALLV)
    Q@locals()
##    for k,v in locals().items():
##        print(k, Qtype(v), v)
##    print(MAP_HASH_GRID_m)
    COUN, PROV, CITY = COUN_PROV_CITY
    GLOBAL['GEOGRAPHY']['COUN'] = COUN
    GLOBAL['GEOGRAPHY']['PROV'] = PROV
    GLOBAL['GEOGRAPHY']['CITY'] = CITY
    GLOBAL['GEOGRAPHY']['LOC'] = '_'.join(COUN_PROV_CITY)
    COMPUTE = GLOBAL['COMPUTE']
    COMPUTE['BUILD_ALL'](COMPUTE)
    if 0: # CL sandbox for testing CL concepts
        M = Qu32((2,2),1)
        s = np_uint32(4)
        run_cl_program(COMPUTE['BUILT']['SANDBOX'].A,
                       COMPUTE['CL_CONTEXT'], (4,),
                       [s],[],[M],[])
        print(M)
        print(M.tobytes())
        print('M[0,1] =',M[0,1])
    print(CONFIG)
    print(FILE_CITY_NAME)
    GLOBAL['BIKEWAYS_SHP'] = {}
    for k,v in CONFIG:
        if k=='BIKEWAYS_SHP_COLUMN':
            GLOBAL['BIKEWAYS_SHP']['column'] = v
        if k.startswith('BIKEWAYS_SHP_') and k[-1].isdigit():
            GLOBAL['BIKEWAYS_SHP'][int(k[-1])] = v.split(';')
        if k=='LRBT_m_DRAW':
            GLOBAL['GEOGRAPHY']['LRBT_m_DRAW'] = [float(x) for x in v.split()]
    PATHS = GLOBAL['PATHS'] = get_paths(CONFIG)
    PROVINCE_DATA_PATH = PATHS['ROOT_PATH']+PROV+'/'
    if 'OSM_MAP' not in PATHS: 
        PATHS['OSM_MAP'] = PROVINCE_DATA_PATH+PROVINCE_NAMES[PROV].lower().replace(' ','-')+'-latest.osm.pbf' #unprotected
    # Hard-coded census geometry:
    if CITY == '__ALL__' or PROV not in PROVINCE_NAMES: POP_GEO = None 
    else:
        POP_GEO_FILE = PROVINCE_DATA_PATH+FILE_CITY_NAME+'.json'
        POP_GEO = load_data(POP_GEO_FILE) 
        if not POP_GEO:
            if 'CITY_BOUNDARY' in PATHS:
                CITY_BOUNDARY_DATA = safe_load_file(PATHS['CITY_BOUNDARY'],'shp')
                CITY_BOUNDARY_G3 = dilate_and_merge(CITY_BOUNDARY_DATA['polygons']) #G4->G3
            else: CITY_BOUNDARY_G3 = None
            POP_GEO = get_city_geometry(CITY,[PROV],[PROV],PATHS['CENSUS_GEOMETRY'],
                                        GLOBAL['PHYSICAL']['city_dilate_m'],CITY_BOUNDARY_G3)
            POP_GEO.update(pop_geo_remake_geometry(POP_GEO))
            save_data(POP_GEO, POP_GEO_FILE)
    GLOBAL['POP_GEO'] = POP_GEO
    datafiles = PATHS['ROOT_PATH']+PROV+'/'+FILE_CITY_NAME+'.'
    PATHS['DAT_FILE'] = datafiles + GLOBAL['FILES']['UI_EXT'] 
    PATHS['SEG_FILE'] = datafiles + 'seg'
    PATHS['RW_FILE'] = datafiles + 'rw'
    PATHS['PLA_FILES'] = datafiles + 'p'
    PATHS['PRI_FILE'] = datafiles + 'pri'
    PATHS['SLOPE_FILE'] = datafiles + 'slo'
    
##    OUTPUTS = get_outputs(CONFIG)    
##    if type(OUTPUTS) is not str: OUTPUTS = 'outputs/'+CITY[0]+'/'
##    if not os_path_isdir(OUTPUTS):
##    os_mkdir(OUTPUTS)
##    if not os_path_isdir(OUTPUTS):
##        print('Failed to create output directory:',OUTPUTS)
##        END()
    if 1:
        CONFIG_DICT = {l[0]:l[1] for l in CONFIG}
        if 'CENTRE' in CONFIG_DICT:
            CENTRE = tuple([float(c) for c in CONFIG_DICT['CENTRE'].split()])
            print(CENTRE)
        else:
            if POP_GEO: CENTRE = POP_GEO['centre']
            else: Qe('No centre in MAIN()!')
        GLOBAL['MAP'] = osmmap.OSM_Map(CENTRE)
        MAP = GLOBAL['MAP']
        TRANSIT_DOWNTOWN = []
        TRANSIT_SUBURB = []
        for C in CONFIG:
            if C[0]=='VIEW':
                MAP.optional['view'] = [float(s) for s in C[1].strip().split()]
                #print(Q(MAP.optional['view']))
            if C[0]=='TRANSIT_DOWNTOWN':
                TRANSIT_DOWNTOWN.append([float(s) for s in C[1].strip().split()])
            if C[0]=='TRANSIT_SUBURB':
                TRANSIT_SUBURB.append([float(s) for s in C[1].strip().split()])
        if TRANSIT_DOWNTOWN: MAP.optional['downtown'] = TRANSIT_DOWNTOWN
        if TRANSIT_SUBURB: MAP.optional['suburb'] = TRANSIT_SUBURB
        if POP_GEO:
            for k in POP_GEO:
                if k.endswith('_boundary'):
                    MAP.mount(k, Geometry(POP_GEO[k],form='xy')) # non-del mount
###########################################################################################################################################
        NM = 'NWK MAP ' #space! # Q does nothing, yet.
        # IDEA: copy qexpr as comment at beginning of each main_*()
        TASKS = [(main_GTFS, "", 0),
                 (main_transit, "", 0),
                 (main_map_network, "MAP PATHS.DAT_FILE<OSM_MAP FILES.UI_EXT GEOGRAPHY>LRBT_m_VIEW", 1),
                 (main_bridge_fixes, "", 1),
                 #(main_connectivity, "NWK.TRs<TNs", 1),
                 (main_shp_bikeways, """MAP>roads.property.streetName?:osmSN
                    NWK.TRs PATHS.BIKEWAYS_SHP? COMPUTE ALGORITHM.MAP_HASH_GRID_m """,
                    RUN_SHP_BIKEWAYS), # in conflation.py
                 ###    # search: MAP$streetName?:osmSN
                 (main_best_infra, NM, 1),
                 (main_make_LTS, NM, 1),
                 (main_diagnose_OSM_bikeways, ALLV, 0),
                 (main_statistics, ALLV, 0),
                 (main_import_speeds, NM+'GEOGRAPHY', 0),
                 (main_topography, 'MAP>buffered_boundary', RUN_TOPOG),
                 (main_slope, "NWK MAP>topography? PATHS", RUN_SLOPE and RUN_TOPOG),
                 ###lambda: disp(MAP['topography']),
                 (main_workplaces, "MAP PATHS.WORKPLACES CONFIG", RUN_SIM and 'WORKPLACES' in PATHS),
                 (main_buildings, "MAP PATHS.BUILDINGS", 1),
                 (main_popdens, NM+"PATHS.CENSUS POP_GEO>parcel_codes", RUN_POP_DENS and POP_GEO),
                 (main_popsamp, "NWK MAP>pop_dens GEOGRAPHY.PROV<CITY PHYSICAL.pop_res_sample_m",
                      RUN_SIM and POP_GEO), # Population parcels -> point sampling on roads with type that have houses
                 (main_anchors, NM, POP_GEO),
                 (main_rw_study, "MAP NWK>TRs PATHS.RW_FILE COMPUTE", RUN_RW_SIM),
                 (main_seg_study, "MAP NWK>TRs PATHS.SEG_FILE COMPUTE", RUN_SEG_SIM),
                 (main_planner_ai, "", RUN_RW_CURVE),
                 (main_overlays, 'MAP', 1),
                 (main_planner_fixes__, '', 1), 
                 lambda: print("MAP.optional.keys:",MAP.optional.keys())]
        start_time =  GLOBAL['COMPUTE']['START_TIME']
        for TASK in TASKS:
            if Qtype(TASK) == 'function': TASK()
            else:
                call, qexpr, cond = TASK
                if not call and cond: break
                if cond:
                    seconds = int(time_time() - start_time)
                    print('\n++++++', call.__name__, 'at', seconds, 'seconds. ++++++')
                    call(GLOBAL, 'Q-TODO')
        #
        TRs = GLOBAL['NWK'].TRs
        print('TRs extra:',TRs.extra.keys())
        print('TRs roads property:',TRs.roads.property.keys())
        
        if RUN_INTERACTIVE:
            GLOBAL['TO_DRAW'] = [['pop_dens','vis_LTS'],
                                 ['workplaces','vis_LTS'],
                                 ['topography','inst_slope'],
                                 ['S*B_3D|B*S_3D S*B_2D|B*S_2D'],
                                 ['!S*B_3D|B*S_3D !S*B_2D|B*S_2D']]
            interactive.run(GLOBAL,"""vis_LTS vis_best_infra passageType is_residential
BLANK inst_slope vis_PRIORITY fwd_PRIORITY bwd_PRIORITY vis_thread """.split())
            # S*B_3D|B*S_3D !S*B_3D|B*S_3D S*B_2D|B*S_2D !S*B_2D|B*S_2D
        return GLOBAL
    else:
        print("Problems encountered in configuration - cannot continue.")
        END()



def main_planner_fixes__(GLOBAL, VARS):
    # hypothesis: some segments are not colored because they are
    # at the start/end of a betweenness path - TO_DO: fix this
    P = GLOBAL['NWK'].TRs.get('vis_PRIORITY')
    if GLOBAL['GEOGRAPHY']['LOC'] == 'BC_Prince George':
        for n in [22947]:
            P[n] = 5
        for n in [22942,22945]:
            P[n] = 6

            

def main_bridge_fixes(GLOBAL, VARS):
    PRINCE_GEORGE_BRIDGE_FIXES = """641121849
477487011
641121846
641121853
477487002
641121839
641121844
641121848
921843062
42958213
24657467
921843067
641121851
641121841
641121843
641110783
615446516
5212569
615446510
615446511
615446515
641110781
42894450
5212805
293215006
5212810
293215006
641104298
641104297
912952979
293215014
912952978
477655016
428892
255706701
42889255
477663698
5008792
477663704
292624110
42889265
641128825
641128839"""
    bf = set(int(x) for x in PRINCE_GEORGE_BRIDGE_FIXES.split())
    TRs = GLOBAL['NWK'].TRs
    oi = TRs['osm_id']
    pt = TRs['passageType']
    for n in Qr(pt):
        if oi[n] in bf: pt[n] = ord('b')


    
def main_transit(GLOBAL, VARS):
    MAP = GLOBAL['MAP']
    for k in 'downtown suburb'.split():
        if k in MAP:
            v = MAP[k]
            radii = [x[-1] for x in v]
            centres = [(x[0],x[1]) for x in v]
            MAP[k] = Geometry(centres) #mount G2
            centres = MAP[k].xy()
            circles = []
            for n,c in enumerate(centres):
                xy = Point(c).buffer(radii[n]).exterior.coords.xy
                circles.append(tuple(zip(*xy)))
            MAP[k] = Geometry(circles,form='xy') #mount G3
    GTFS = GLOBAL.get('GTFS')
    if GTFS:
        shapes = GTFS.get('shapes')
        if shapes:
            lines = []
            for _,v in shapes.items():
                lines.append([(r[1],r[2]) for r in v])
            MAP['GTFS_shapes'] = Geometry(lines) #mount G3
    if GTFS and 'downtown' in MAP and 'suburb' in MAP:
        # test intersection
        GS = [LineString(ls) for ls in MAP['GTFS_shapes'].xy()]
        DT = MultiPolygon([sPolygon(p) for p in MAP['downtown'].xy()])
        SU = MultiPolygon([sPolygon(p) for p in MAP['suburb'].xy()])
        Keep = []
        for gs in GS:
            if gs.intersects(DT) and gs.intersects(SU): Keep.append(True)
            else: Keep.append(False)
        MAP['GTFS_downsub'] = Geometry([gs for n,gs in \
            enumerate(MAP['GTFS_shapes'].xy()) if Keep[n]],form='xy') #mount G3
        
def main_GTFS(GLOBAL, VARS):
    PATHS = GLOBAL['PATHS']
    PATHS['GTFS'] = GLOBAL['PATHS']['ROOT_PATH']\
        + GLOBAL['GEOGRAPHY']['PROV']+'/'\
        + GLOBAL['GEOGRAPHY']['CITY']+' GTFS/'
    if os_path_exists(PATHS['GTFS']):
        GTFS_DATA = {}
        for f in 'stops shapes'.split(): #stop_times
            with open(PATHS['GTFS']+f+'.txt', mode='r') as file:
                GTFS_DATA[f] = read_csv_to_table(file)
        GTFS = {}
        if 'stop_times' in GTFS_DATA:
            print('stop_times')
            def time_to_s(tstr):
                hms = tstr.split(':')
                return 3600*int(hms[0])+60*int(hms[1])+int(hms[2])
            s = GTFS_DATA['stop_times']
            i = s['trip_id']
            at = [time_to_s(t) for t in s['arrival_time']]
            dt = [time_to_s(t) for t in s['departure_time']]
            si = s['stop_id']
            ss = s['stop_sequence']
            t = {n:list() for n in set(i)}
            for n in Qr(i):
                t[i[n]].append((ss[n],si[n],at[n],dt[n]))
            for k,v in t.items():
                v.sort(key=lambda x:x[0])
            GTFS['stop_times'] = t
        if 'stops' in GTFS_DATA:
            print('stops')
            s = GTFS_DATA['stops']
            i = s['stop_id']
            lon = s['stop_lon']
            lat = s['stop_lat']
            GTFS['stops'] = {i[n]:(lon[n],lat[n]) for n in Qr(i)}
        if 'shapes' in GTFS_DATA: # parse 'shapes.txt':
            print('shapes')
            s = GTFS_DATA['shapes']
            i = s['shape_id']
            p = s['shape_pt_sequence']
            lon = s['shape_pt_lon']
            lat = s['shape_pt_lat']
            r = {n:list() for n in set(i)}
            for n in Qr(p):
                r[i[n]].append((p[n],lon[n],lat[n]))
            for k,v in r.items():
                v.sort(key=lambda x:x[0])
            GTFS['shapes'] = r
        GLOBAL['GTFS'] = GTFS

#https://stackoverflow.com/questions/64276513/draw-dotted-or-dashed-rectangle-from-pil
def main_draw_results(GLOBAL, VARS):
    from PIL import ImageDraw, Image, ImageFont
    from gGlobals import Layer
    TRs = GLOBAL['NWK'].TRs
    if 1: #make values
        LTS = GLOBAL['NWK'].TRs['fwd_LTS']
        Priority = [None, None]
        if 'STUDY' in GLOBAL:
            y = (LTS==3)|(LTS==4)|(LTS==0)|(LTS==-1)
            for n in range(2):
                k = 'S*B_%dD|B*S_%dD'%(n+2,n+2)
                x = GLOBAL['STUDY']['*Seg'].get(k, None)
                if x is not None:
                    Priority[n] = x*y*(x>4) # formula.
    if 1: #make .shp
        l,r,b,t = GLOBAL['GEOGRAPHY']['LRBT_m_VIEW']
        shFrame = sPolygon([(l,b),(r,b),(r,t),(l,t)])
        shTRs = TRs.get_LineStrings()
        inFrame = np_array([shFrame.intersects(sr) for sr in shTRs],dtype='bool')
        print('For .shp:',sum(inFrame),len(inFrame),'inFrame.')
        pl = []
        for n,v in enumerate(inFrame):
            if v:
                co = shTRs[n].coords
                pl.append([[co[i] for i in Qr(len(co))]])
        # change from meter to latlon coordinates:
        G = Geometry(pl,form='xy')
        G(GLOBAL['MAP'].centre)
        shp_data = {'polylines':G.lonlat(),
                    'field_names':['LTS'],
                    'records':{'LTS':LTS[inFrame]}}
        if 'STUDY' in GLOBAL:
            if Priority[0] is not None:
                shp_data['field_names'].append('Priority_2D')
                shp_data['records']['Priority_2D'] = Priority[0][inFrame]
            if Priority[1] is not None:
                shp_data['field_names'].append('Priority_3D')
                shp_data['records']['Priority_3D'] = Priority[1][inFrame]
        save_data(shp_data,'outputs/'+GLOBAL['GEOGRAPHY']['CITY']+'.shp')
    for TITLE in ['LTS','Priority_2D','Priority_3D','Priority_Diff']:# Draw .png:
        toDraw = None
        if TITLE == 'LTS':
            toDraw = LTS
            reverseLayers = True
            Palette = GLOBAL['LAYERS']['LTS']
        elif 'STUDY' in GLOBAL:
            if TITLE == 'Priority_2D' and Priority[0] is not None: toDraw = Priority[0]
            if TITLE == 'Priority_3D' and Priority[1] is not None: toDraw = Priority[1]
            if TITLE == 'Priority_Diff' and Priority[0] is not None and Priority[1] is not None:
                toDraw = (((Priority[0]==0) & (Priority[1]>0)) \
                            | ((Priority[1]==0) & (Priority[0]>0))) * 2 \
                            + (Priority[0]!=Priority[1])*5
            reverseLayers = False
            Palette = GLOBAL['LAYERS']['PRIORITY']
        if toDraw is not None:
            print('Drawing',TITLE+'.png')
            data = {'lrbt_view': GLOBAL['GEOGRAPHY']['LRBT_m_VIEW'],
                    'road_width': GLOBAL['PHYSICAL']['road_width_m'],
                    'road_x':TRs.roads.node_x,
                    'road_y':TRs.roads.node_y,
                    'roads':TRs.lines,
                    'road_width_m': 5}#GLOBAL['PHYSICAL']['road_width_m']}
            for d in 'xy': # bounds - or use lrbt_view?
                r = data['road_'+d][data['roads'].data]
                data['m'+d] = min(r)
                data['M'+d] = max(r)
            print(data['mx'],data['Mx'],data['my'],data['My'])
            print(data['lrbt_view'])
            data['road_data'] = toDraw
        ##    layers = [Layer('road', 0, (1,0,0))]
        ##    layers.append(Layer('road', 1, (1,0.5,0)))
        ##    layers.append(Layer('road', 2, (1,1,0)))
        ##    layers.append(Layer('road', 3, (0.5,1,0)))
        ##    layers.append(Layer('road', 4, (0,1,0)))
        ##    layers.append(Layer('road', 5, (0,1,0.5)))
            data['layers'] = Palette
            ### Make PDF map:
            ### drawpdf_draw_pdf(data, 'C:/Users/A/Desktop/map.pdf')
            if 1:
                UPSAMPLE = 3
                if 'LRBT_m_DRAW' in GLOBAL['GEOGRAPHY']:
                    l,r,b,t = GLOBAL['GEOGRAPHY']['LRBT_m_DRAW']
                else: l,r,b,t = data['lrbt_view']
                aspect = (t-b)/(r-l)
                max_pixels = 9400
                if aspect < 1: W = max_pixels; H = int(max_pixels*aspect)
                else: H = max_pixels; W = int(max_pixels/aspect)
                print(aspect, H, W, 'aspect H W')
                img = Image.new("RGB", (W*UPSAMPLE, H*UPSAMPLE))
                img1 = ImageDraw.Draw(img)
                img1.rectangle([0,0,W*UPSAMPLE,H*UPSAMPLE], fill = tuple([127]*3))
                scale = H/(t-b)
                A = l
                B = b
                X = (data['road_x']-A)*scale*UPSAMPLE
                Y = (data['road_y']-B)*scale*UPSAMPLE
                colour_map = {x.value:tuple(int(c*255) for c in x.rgb) for x in data['layers']}
                road_ix = {v:list() for v in set(data['road_data'])}
                for rn, v in enumerate(data['road_data']):
                    road_ix[v].append(rn)
                print([(x,len(y)) for x,y in road_ix.items()], Qstr(data['road_data']))
                YMAX = H*UPSAMPLE
                for v in sorted(road_ix.keys(), reverse=reverseLayers):
                    colour = colour_map[v]
                    for rn in road_ix[v]:
                        R = data['roads'][rn]
                        for n in range(len(R)-1):
                            shape = (X[R[n]], YMAX-Y[R[n]], X[R[n+1]], YMAX-Y[R[n+1]])
                            img1.line(shape, fill=colour, width=2*UPSAMPLE)#max(1,int(0.5+data['road_width']*scale)))
                        for rn in R:
                            r = 1*UPSAMPLE - 0.5
                            ym = YMAX-Y[rn]
                            img1.ellipse([(X[rn]-r,ym-r),(X[rn]+r,ym+r)], fill=colour)
                ##flipped by hand, no need for: img.transpose(Image.FLIP_TOP_BOTTOM)
                # LEGEND
                font = ImageFont.truetype("arialbd.ttf", 100*UPSAMPLE)
                LegendText = [TITLE]+[x.legend_name for x in data['layers']]
                LegendRGB = [(0,0,0)]+[tuple(int(c*255) for c in x.rgb)
                                       for x in data['layers']]
                LegendTextW = max(font.getsize(x)[0] for x in LegendText)
                TextH = font.getsize('I')[1]
                LegendTextH = TextH*1.4*len(data['layers'])+TextH
                LEGEND_POS = 1
                if LEGEND_POS == 1: #NE
                    LegendX = (W-200)*UPSAMPLE - LegendTextW
                    LegendY = 200*UPSAMPLE
                LegendBox = [LegendX, LegendY,
                    LegendX+LegendTextW,
                    LegendY+LegendTextH]
                img1.rectangle(LegendBox, fill = tuple([255-32]*3))
                for n,t in enumerate(LegendText):
                    img1.text((LegendX, LegendY + n*1.4*TextH),
                              t, font=font, fill=LegendRGB[n])
                #img.show()
                img = img.resize((W,H), resample=Image.ANTIALIAS) 
                img.save('outputs/'+GLOBAL['GEOGRAPHY']['CITY']+' '+TITLE+'.png')

    
    ################# SVG code: not used. Make PDF instead.
##########    mx, Mx, my, My = data['lrbt_view'] #
##########    SCALE = 10^-7 # 10cm ^-1.
##########    VIEW = '%f %f %f %f'%(mx*SCALE, Mx*SCALE, my*SCALE, My*SCALE)
##########    path_test = 'C:/Users/A/Desktop/test_svg.svg'
##########    XML_test = '<?xml version="1.0" encoding="UTF-8"?>'
##########    XML_test += "<!DOCTYPE svg  PUBLIC '-//W3C//DTD SVG 1.1//EN' "
##########    XML_test += " 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'>"
##########    XML_test += '<svg enable-background="new '+VIEW+'" version="1.1" '
##########    XML_test += ' viewBox="'+VIEW+'" '
##########    XML_test += ' xml:space="preserve" xmlns="http://www.w3.org/2000/svg">'
##########    XML_test += '<g fill="none" stroke="rgb(0,0,0)" stroke-linecap="round" stroke-width="%f">'%(SCALE*data['road_width_m'])
##########    X = (data['road_x']-mx)*SCALE
##########    Y = (data['road_y']-my)*SCALE
##########    for r,R in enumerate(data['roads']):
##########        if data['road_data'][r]==0: #test
##########            XML_test += '<polyline points="'
##########            for p in R: XML_test += str(X[p])+','+str(-Y[p])+' ' #-Y sic.
##########            XML_test += '"/>'
##########    XML_test += "</g>"
##########    XML_test += "</svg>"
##########    with open(path_test, 'w') as f:
##########        f.write(XML_test)
##########    drawing = svg2rlg(path_test)
##########    renderPDF.drawToFile(drawing, "C:/Users/A/Desktop/pdf_test.pdf")


def main_connectivity(GLOBAL, VARS):
    TNs = GLOBAL['NWK'].TNs
    TRs = GLOBAL['NWK'].TRs
    los = [[n]+list(TNs.neighbTNs[n]) for n in Qr(TNs.N)]
    merged = merge_sets(los)
    #print([len(v) for v in merged])
    tn2tr = {}
    for tr in Qr(TRs):
        for tn in [TRs.startTN[tr], TRs.endTN[tr]]:
            if tn in tn2tr: tn2tr[tn].append(tr)
            else: tn2tr[tn] = [tr]
    nets = [set(sum([tn2tr[tn] for tn in s],[])) for s in merged]
    nets.sort(key=lambda x:len(x),reverse=True)
    NN = Qu32(len(TRs))
    for n, net in enumerate(nets):
        for v in net: NN[v] = n
    TRs.extra['net'] = NN
    TRs.extra['main_net'] = NN==0


def main_best_infra(GLOBAL, VARS):
    B = {}
    T = GLOBAL['NWK'].TRs
    for D in Qmx('fwd_ bwd_'):
        i = T[D+'osm_infra']
        e = T.extra.get(D+'shp_infra')
        if e is not None:
            #i = np_maximum(i, e)
            i = e ###
        B[D+'best_infra'] = i
    B['equ_best_infra'] = B['fwd_best_infra'] == B['bwd_best_infra']
    B['vis_best_infra'] = np_maximum(B['fwd_best_infra'], B['bwd_best_infra'])
    T.extra.update(B)



def main_diagnose_OSM_bikeways(GLOBAL, VARS):
    max_count = 50
    Q>GLOBAL
    osr = Q('NWK TRs osm_road')
    rp = (Q('MAP roads give_props'))()
    props = [rp[n] for n in osr]
    out = {}
    for way in props:
        for k,v in way.items():
            if k not in out: out[k] = set()
            out[k].add(v)
    keys = list(out.keys())
    keys.sort()
    for k in keys:
        print(k,'==> ',end='')
        v = list(out[k])
        if len(v) > max_count:
            print('MANY')
        else:
            v.sort()
            print(', '.join(v))
    input()


    ##########
##    for n,p in enumerate(props):
##        L = []
##        if str(p.get('oneway')) in ['-1','yes']:      
##            for k,v in p.items():
##                if 'cycl' in k:
##                    print(p)
##                    input()
##                    break
##            if 'bicycle' in k or 'cycleway' in k:
##                s = k+'='+v
####                for r in 'right left both'.split():
####                    s = s.replace(r,'DIR')
##                L.append(s)
##        if L:
##            L.sort()
##            SET.add('  '.join(L))
##    L = list(SET)
##    L.sort()
    #print('\n'.join(L))
    #input()
    
    
# Can optimize this by checking for duplicate dir/TR before passing to formula.
def main_make_LTS(GLOBAL, VARS): #!
    OP = GLOBAL['MAP'].roads.property
    T = GLOBAL['NWK'].TRs
    W2 = {w:T[w] for w in Qmx('passageType is_residential is_service cycle_type bike_access')}
    one = Qmx('best_infra speedLimit_kph numLanes streetParking force_dismount')
        #streetParking, bikeaccess - check for 1/2way in OSM
    WF = {}; WB = {}
    for d,W in [('f',WF),('b',WB)]:
        W.update({w:T[d+'wd_'+w] for w in one})
    I = (len(T), None)
    fLTS = Qi8(*I); bLTS = Qi8(*I); hLTS = Qbool(*I)
    fLTSlane = Qi8(*I); bLTSlane = Qi8(*I); hLTSlane = Qbool(*I)
    call = formulae.LTS_formula
    for t in Qr(T):
        if t and t%10000 == 0: print(t,len(T))
        f = {k:v[t] for k,v in WF.items()}
        b = {k:v[t] for k,v in WB.items()}
        two = {k:v[t] for k,v in W2.items()}
        if f==b:
            f.update(two)
            L = call(f) # costly
            fLTS[t] = L; bLTS[t] = L; hLTS[t] = 0
            if L == 0: L = 1
            elif L in [2,3,4]:
                f['best_infra'] = max(3,f['best_infra']) # upgrade to at least 3 - painted lane (LTS)
                L = call(f)
            fLTSlane[t] = L; bLTSlane[t] = L; hLTSlane[t] = 0
        else:
            f.update(two); b.update(two)
            Lf = call(f); Lb = call(b) # costly
            fLTS[t] = Lf; bLTS[t] = Lb; hLTS[t] = 1
            for n in [0,1]:
                L = [Lf,Lb][n]
                if L == 0: L = 1
                elif L in [2,3,4]:
                    x = [f,b][n]
                    x['best_infra'] = max(3,x['best_infra']) # this modifies b/f - by ref.
                    L = call(x)
                [fLTSlane,bLTSlane][n][t] = L
            hLTSlane[t] = int(f!=b)
    T['fwd_LTS'] = fLTS; T['bwd_LTS'] = bLTS
    T['equ_LTS'] = fLTS == bLTS
    T['vis_LTS'] = np_maximum(fLTS, (127-(254*T['equ_LTS'])))
    T['het_LTS'] = hLTS
    T['fwd_LTS_lane'] = fLTSlane; T['bwd_LTS_lane'] = bLTSlane
    T['equ_LTS_lane'] = fLTSlane==bLTSlane
    T['vis_LTS_lane'] = np_maximum(fLTSlane, (127-(254*T['equ_LTS_lane'])))
    T['het_LTS_lane'] = hLTSlane
##    T['fwd_add_lane'] = fLTS != fLTSlane
##    T['bwd_add_lane'] = bLTS != bLTSlane
##    T['equ_add_lane'] = T['fwd_add_lane'] == T['bwd_add_lane']
##    T['vis_add_lane'] = Qu8(T['fwd_add_lane'] | T['bwd_add_lane']) + Qu8(~T['equ_add_lane'])
    T['lane_makes_worse'] = Qu8(((T['fwd_LTS_lane']>T['fwd_LTS']) & (T['fwd_LTS']>0)) |
                                ((T['bwd_LTS_lane']>T['bwd_LTS']) & (T['bwd_LTS']>0)))
    for DIR in 'fb':
        P = DIR+'wd_'
        T[P+'candidate_lane'] = ((T[P+'LTS']>2) & (T[P+'LTS_lane']<3)) | ((T[P+'LTS']==0) & (T[P+'LTS_lane']==1))
        T[P+'candidate_track'] = (T[P+'LTS']==3) | (T[P+'LTS']==4) & (T[P+'LTS_lane']>2)
        T[P+'candidate_infra'] = (T[P+'candidate_track']*2) + (T[P+'candidate_lane'] & ~T[P+'candidate_track'])
    T['equ_candidate_infra'] = T['fwd_candidate_infra'] == T['bwd_candidate_infra']
    T['vis_candidate_infra'] = np_maximum(T['fwd_candidate_infra'],T['bwd_candidate_infra'])
    #GLOBAL['make_LTS'] = locals()
    
def main_statistics(GLOBAL, VARS):
    # Exercise: rewrite this all in Q!?
    # QL('NWK TRs>extra LTS<<osm_road roads>props_decoder props')
    TRs = GLOBAL['NWK'].TRs
    ###obsolete 2 way now  - #LTS = TRs.extra['LTS']
    osmr = TRs.osm_road
    roads = TRs.roads
    decoder = roads.props_decoder
    props = roads.props
    cats = {v:{} for v in set(LTS)}
    if 0:
        for n, lts in enumerate(LTS):
            for k,v in decoder(props[osmr[n]]).items():
                if k in cats[lts]: cats[lts][k].add(v)
                else: cats[lts][k] = set([v])
        for lts, stat in cats.items():
            print('####### LTS =',lts,'\n')
            for k,v in stat.items():
                if len(v)<15: print(k,v)
    if 0:
        for n, lts in enumerate(LTS):
            d = decoder(props[osmr[n]])
            if sum(['cycleway:' in k for k in d.keys()]):
                print(lts,'|',d)
    if 1:
        S = set()
        for n, lts in enumerate(LTS):
            d = decoder(props[osmr[n]])
            if d.get('highway') == 'track':
                for k in 'osm_id name source'.split():
                    if k in d: d.pop(k)
                S.add(str(d))
    disp(list(S))
    input()


    


class Slope:
    def __init__(self, node_x, node_y, lines, data, draw, TR_offsets, step_m):
##        self.node_x = node_x
##        self.node_y = node_y
##        self.lines = lines
##        self.data = data
##        self.draw = draw
##        self.TR_offsets = TR_offsets
##        self.step_m = step_m
        Qsave(self, locals())
def main_slope(GLOBAL, VARS):
    TRs = GLOBAL['NWK'].TRs
    MAP = GLOBAL['MAP']
    passageType = TRs['passageType']
    topography = MAP.get('topography')
    tiles = topography
    nominal_step_m = 5
    SLOPE_FILE = GLOBAL['PATHS']['SLOPE_FILE']
    if os_path_isfile(SLOPE_FILE):
        with open(SLOPE_FILE, 'rb') as f:
            out = pickle.load(f)
    else:
        out = {}
        if tiles: tiles = [t[1] for t in tiles.tiles]
        ###if... #os_path_isfile(SLOPE_FILE):
        #Quantize TRs:
        # not much gain from : nx,ny = Q(TRs,'roads.node_x<node_y')
        nx = TRs.roads.node_x
        ny = TRs.roads.node_y
        lines = TRs.lines
        lx = LoL(); ly = LoL()
        for lol, nodes in [(lx,nx), (ly,ny)]:
            lol.load(nodes[lines.data], lines.offsets)
        LS = [LineString(zip(lx[i],ly[i])) for i in Qr(lines.N)]
        POINTS = []
        offsets = Qu32(len(LS)+1,0)
        step_m = Qf32(len(LS),0)
        print('Sampling TR polylines every',nominal_step_m,'meters (nominal) for slope...')
        for count, ls in enumerate(LS):
            if count%10000==0: print(count)
            len_m = ls.length
            N = ceil(len_m/nominal_step_m)
            step = len_m/N
            step_m[count] = step
            POINTS += tuple(ls.interpolate(step*n,normalized=False) for n in Qr(N+1))
            offsets[count+1] = offsets[count] + N+1
        # Map points on tiles
        G = QcF(Geometry,[[p.x for p in POINTS], [p.y for p in POINTS], 'xy'])
        del POINTS
        G(MAP.centre)
        lon = Qf64(G.lon()); lat = Qf64(G.lat())
        N = len(lon)
        T = len(tiles)
        pw = Qf64([t.pixel_w for t in tiles])
        ph = Qf64([t.pixel_h for t in tiles])
        l,r,b,t = (Qf64([t.data_lrbt_deg[n] for t in tiles]) for n in Qr(4))
        W = r-l
        H = t-b
        print('Assigning topog tiles...')
        print(N)
        for i in Qr(T):
            ini = (lon>=l[i]) & (lon<=r[i]) & (lat>=b[i]) & (lat<=t[i])
            ta = ini*i + ~ini*np_int16(T)
            if i == 0: tile_assign = ta
            else: tile_assign = np_minimum(ta, tile_assign)
        #in_tile = tile_assign<T
        #on_tile = tile_assign[in_tile]
        #tile_assign_valid = tile_assign[in_tile]
        NaN =  float('nan')
        height = Qf32(N)
        if 1:
            for n in Qr(N):
                if n%100000 == 0: print(n)
                ta = tile_assign[n]
                if ta==T: height[n] = NaN
                else: height[n] = tiles[ta].sample_one(lon[n],lat[n])
        height_lol = LoL()
        height_lol.load(height, offsets)
        # slope:
        slope_lol = QcF(LoLf32, [[np_diff(h)/step_m[n] for n,h in enumerate(height_lol)]])
        for n, v in enumerate(passageType): # set slope of all passages to zero:
            if v: 
                slope_lol.data[slope_lol.offsets[n]:slope_lol.offsets[n+1]] = 0
        abs_slope_lol = LoL()
        abs_slope_lol.load(abs(slope_lol.data).astype('float32'), slope_lol.offsets)
        keys = {'min':min,
                'max':max,
                'avg':lambda x:sum(x)/len(x)}
        def slope_to_perc_int(lst):
            cap = 6.5
            v = np_clip(Qf32(lst)*100.0, a_max=cap, a_min=-cap)
            return np_nan_to_num(v, copy = False, nan = -127.0).astype('int8')
        def slope_penalty(lst, table):
            l = int(len(table)//2)
            v = l + np_clip(Qf32(lst)*100.0, a_max=l, a_min=-l).astype('uint8')
            return sum(table[v])/len(lst)
        out['TRs.extra'] = {}
        for k,lamb in keys.items():
            out['TRs.extra']['abs_slope_'+k] = slope_to_perc_int([lamb(s) for s in abs_slope_lol])
        # slope to penalty:
        table = formulae.slope_table
        for n in [0,1]:
            sign = (-1)**n
            out['TRs.extra']['fb'[n]+'wd_climb'] = Qf32(
                [slope_penalty(s*sign,table) for s in slope_lol])
        x,y = [G.data[d].array() for d in 'xy']
        lin = LoL()
        #lines.load(np_arange(x.size, dtype='uint32'), offsets)
        lens = np_diff(offsets)
        data = np_concatenate([offsets[n] + np_arange(1, (lens[n]-1)*2+1, dtype='uint32')//2
                               for n in Qr(lens)],axis=None)
        offs = np_arange(data.size//2+1,dtype='uint32')*2
        lin.load(data, offs)
        slope = Slope(x, y, lin, slope_lol.data, slope_to_perc_int(abs_slope_lol.data), offsets, step_m)
        out['SLOPE'] = {'inst_slope':slope}
        with open(SLOPE_FILE, 'wb') as f:
            pickle.dump(out, f)
    # modified state:
    GLOBAL['SLOPE'] = out['SLOPE']
    TRs.extra.update(out['TRs.extra'])
    
    


    
def dilate_and_quantize(contour, dilate, quant):
    LS = LineString(contour+[contour[0]]) # Must be LineString to support interpolation
    if dilate: dilated = LineString(LS.buffer(dilate,1).exterior) # use default(rounded) dilation
    else: dilated = LS
    N_cand = ceil(dilated.length/quant)
    return [tuple(dilated.interpolate(n*quant).coords)[0] for n in Qr(N_cand)]



def main_import_speeds(GLOBAL, VARS):
    if GLOBAL['GEOGRAPHY']['LOC'] != 'BC_Prince George': return None
    if 1:
        HERE_PATH = GLOBAL['PATHS']['ROOT_PATH']+"City of Prince George/HERE Measured Hourly Motor Speeds/"
        STREETS = load_data(HERE_PATH+"Prince_George_streets.shp") # is G4 with second layer always len=1
        polylines = Geometry([v[0] for v in STREETS['polylines']]) # remove second layer
        GLOBAL['MAP']['shp_roads'] = polylines # mount
        LS = [LineString(line) for line in polylines.xy()]
        SAMPLES = []
        print('sampling...')
        samples_per_road = 100
        for ls in LS:
            SAMPLES.append([ls.interpolate(n/(samples_per_road-1),True) for n in Qr(samples_per_road)])
        fol = GLOBAL['NWK'].hashgrid.find_one_line
        print('finding lines...')
        TR_guesses = [[fol((P.x,P.y), 'TRs', d_max = 1.0) for P in L] for L in SAMPLES]
        print('done!')
        TR_guesses = [[v for v in L if v is not None] for L in TR_guesses]
        most_frequent = lambda x: max(set(x), key = x.count) if len(x) else None
        TR_guess = [most_frequent(L) for L in TR_guesses]
        TR_freq = [[L.count(v) for v in set(L)] for L in TR_guesses]
        for L in TR_freq: L.sort(reverse=True)
        N = 0
        for n, L in enumerate(TR_freq):
            if len(L)>1 and L[0]==L[1]:
                N+=1; print(TR_guesses[n])
        print(N)
        GLOBAL['TEMP'] = locals()
    if 0:
        SPEEDS = load_data(HERE_PATH+"Prince_George_data_avg_speed_hourly_weekday_weekend.csv")
        x = Q(SPEEDS)
        speed_data = {}
        for h in SPEEDS.head:
            if h == 'LINK_DIR': speed_data['code'] = SPEEDS[h]
            else:
                i = h.split()
                if i[-1] == 'mean':
                    k = str(ord(i[0][4])-100)+' ' # week... d->0 e->1
                    k += str((int(i[1])+12*(i[2]=='PM'))%24)
                    speed_data[k] = Qf32([NaN if type(v) is str else v for v in SPEEDS[h]])
        for hour in "AMrush PMrush daylight 24hour".split():
            for day in "weekday weekend".split():
                    for extreme in "min max".split():
                            hours = {'AMrush': [7, 8, 9],
                                     'PMrush': [3, 4, 5],
                                     'daylight':list(range(5,22)),
                                     '24hour':list(range(0,24))}[hour]
                            func = {'min':lambda x: min(x) if x else NaN, 'max':lambda x: max(x) if x else NaN}[extreme]
                            print(hour, day, extreme, hours, func.__name__)
                            columns = [speed_data[str(ord(day[4])-100)+' '+str(h)] for h in hours]
                            vals = Qf32([func([col[n] for col in columns if not np_isnan(col[n])]) \
                                         for n in Qr(SPEEDS)])
                            speed_data[' '.join([hour, day, extreme])] = vals





def main_topography(GLOBAL, VARS):
    MAP = GLOBAL['MAP']
    TOPOG_BOUNDARY = MAP['buffered_boundary'].lonlat()
    TOPOG_FILES = get_topog_files(TOPOG_BOUNDARY)
    if type(TOPOG_FILES) is str: Qer(TOPOG_FILES) # if error
    else:
        TOPOG_CONFIG_TEXT = download_all_topog(GLOBAL['PATHS']['ROOT_PATH']+'Canada/topography/',TOPOG_FILES)
        TOPOG_LIST = get_topography(parse_config(TOPOG_CONFIG_TEXT))
        TOPOG = Topography(TOPOG_BOUNDARY)
        for T_FILE,T_LRBT in TOPOG_LIST:
            TOPOG.add(T_FILE, safe_load_file(T_FILE,'tif'), T_LRBT[0:2],T_LRBT[2:4])
        #TOPOG.crop_all() # not implemented
        MAP['topography'] = TOPOG # mount non-Geo
        TOPOG.place_on_map()


def OSM_DAT_city_mods(OSM_DAT, GEOGRAPHY):
    LOC = GEOGRAPHY['PROV']+'_'+GEOGRAPHY['CITY']
    if 0:#LOC =='BC_Nanaimo':
        print('OSM_DAT_city_mods:',LOC)
        # add 3 copies of bridge property:
        P = OSM_DAT.way_props['highway']
        find = None
        for n,p in enumerate(P):
                if 'osm_id\x00331014949' in p:
                        find = n
                        break
        L = list(OSM_DAT.way_props['highway'])
        L += [L[find]]*3
        OSM_DAT.way_props['highway'] = tuple(L)
        ################### These numbers have changed - should be by absolute OSM ID.
        NODES = [164431, 189500, 164432, 164345, 189020, 55352]
        # add two lines:
        L = OSM_DAT.way_lines['highway']
        L = list(L.tolol())
        L += [NODES[0:2]]
        L += [NODES[2:4]]
        L += [NODES[4:6]]
        OSM_DAT.way_lines['highway'] = LoLu32(L)
        #################
        XX = [OSM_DAT.node_x['highway'][n] for n in NODES]
        YY = [OSM_DAT.node_y['highway'][n] for n in NODES]
        DX = [XX[1]-XX[0], XX[3]-XX[2], XX[5]-XX[4]]
        DY = [YY[1]-YY[0], YY[3]-YY[2], YY[5]-YY[4]]
        LEN_M = [(dx**2+dy**2)**0.5 for dx,dy in zip(DX,DY)]
        L = OSM_DAT.seg_len_m['highway']
        L = list(L.tolol())
        for len_m in LEN_M: L += [tuple([len_m])]
        OSM_DAT.seg_len_m['highway'] = LoLf32(tuple(L))


def main_map_network(GLOBAL, VARS):
    PATHS = GLOBAL['PATHS']; MAP = GLOBAL['MAP']
    print(PATHS['DAT_FILE'])
    if os_path_isfile(PATHS['DAT_FILE']):
        print('DAT file available.')
        UI_DATA = safe_load_file(PATHS['DAT_FILE'], GLOBAL['FILES']['UI_EXT'])
        OSM_DAT = osmxml.OSMData()
        OSM_DAT.import_ui(UI_DATA)
        OSM_DAT_city_mods(OSM_DAT, GLOBAL['GEOGRAPHY'])  ### REFACTOR!   
        QcF(MAP.load_data,[OSM_DAT])
        NWK = UrbNet.Network(MAP,GLOBAL,build=False)
        NWK.import_ui(UI_DATA)
        QcF(NWK.spatial_hash,[GLOBAL])
    else:
        print(PATHS['OSM_MAP'])
        if os_path_isfile(PATHS['OSM_MAP']):
            wt = [osmfilter.WAY_TYPES, ['building','building:part']]
            wtp = set()
            if 'analysis_boundary' in MAP:
                OSM_BOUNDARY = MAP['analysis_boundary'].lonlat()
                OSM_NODE_FRAME = [([-0.5,0.5]*2)[n] + MAP['analysis_boundary'].get_lrbt('deg')[n] for n in Qr(4)]
            else: OSM_BOUNDARY = None; OSM_NODE_FRAME = None
            OSM_DAT = osmxml.OSMData(MAP.centre, PATHS['OSM_MAP'], wt, wtp, OSM_BOUNDARY, OSM_NODE_FRAME)
            OSM_DAT_city_mods(OSM_DAT, GLOBAL['GEOGRAPHY'])  ### REFACTOR!
            QcF(MAP.load_data,[OSM_DAT])
            UI_DATA = OSM_DAT.export_ui()
            print('Building Network...',end='')
            NWK = UrbNet.Network(MAP,GLOBAL,build=True)
            NWK.spatial_hash(GLOBAL)
            print('Network done.')
            UI_DATA.update(NWK.export_ui())
            save_data(UI_DATA, PATHS['DAT_FILE'], GLOBAL['FILES']['UI_EXT'])
        else: END('OSM_MAP file missing.')
    # unprotected:
    CENTRE_MATCH = dist(MAP.centre, OSM_DAT.centre)
    if CENTRE_MATCH:
        print('Distance between centres (should be zero):',CENTRE_MATCH)
        END('Centre match integrity check failed.')
    if 'buffered_boundary' in MAP:
        GLOBAL['GEOGRAPHY']['LRBT_m_VIEW'] = MAP['buffered_boundary'].get_lrbt('m')
    else:
        RX,RY = MAP.roads.node_x, MAP.roads.node_y
        GLOBAL['GEOGRAPHY']['LRBT_m_VIEW'] = [RX.min(),RX.max(),RY.min(),RY.max()]
    Qsave(GLOBAL, locals(),'NWK OSM_DAT')

def main_workplaces(GLOBAL, VARS):
    WORKPLACES = build_workplaces(safe_load_file(GLOBAL['PATHS']['WORKPLACES'],'shp csv xls xlsx'),
                                  get_workplaces(GLOBAL['CONFIG']))
    if WORKPLACES: GLOBAL['MAP']['workplacesT'] = WORKPLACES # mount
    else: END('Failed to read workplace database!')

def main_buildings(GLOBAL, VARS): ### unprotected
    # Consider triangulation:
    # In Nanaimo, residential building #21255 has a missing node (not closed)
    # fast triangulation works (1.35ms avg) but slow (~8ms avg) fails for this polygon only 
    MAP = GLOBAL['MAP']
    PATHS = GLOBAL['PATHS']
    CONST_H_M = GLOBAL['PHYSICAL']['building_height']
    USE_OSM = 'BUILDINGS' not in PATHS
    if USE_OSM:
        BDATA, BXY = buildings.parse_OSM_buildings(GLOBAL['OSM_DAT'], const_h_m=CONST_H_M)
        BHEIGHT = BDATA['HEIGHT']
    else:
        BUILDINGS = load_data(PATHS['BUILDINGS'])
        BDATA = buildings.parse_buildings(BUILDINGS['records'], const_h_m=CONST_H_M)
        BVOLUME = BDATA['VOLUME']
        BPOLYGONS = BUILDINGS['polygons']
    BTYPE = BDATA['TYPE']
    ### Need to change this, and return only height, then derive areas from mounted polygons
    print('Found %d buildings'%len(BTYPE),'in',['SHP','OSM'][USE_OSM],'.')
    for ty in 'W':#only parse work buildings for speed# set(BTYPE): # only make building entries for non-empty sets:
        print('Mounting buildings type '+ty+'...')
        k = ty+'buildings'
        if USE_OSM:
            X = BXY['x'];Y = BXY['y']
            polygons = [[tuple(zip(X[n].tolist(),Y[n].tolist()))] \
                        for n,T in enumerate(BTYPE) if T==ty] # sic:G4
            MAP[k] = Geometry(polygons,form='xy')
            area = Qf32([sPolygon(p[0]).area for p in polygons])
            MAP[k]['volume'] = Qf32([BHEIGHT[n] for n,T in enumerate(BTYPE) if T==ty])*area
        else:
            MAP[k] = Geometry([BPOLYGONS[n] for n,T in enumerate(BTYPE) if T==ty]) # mount
            MAP[k]['volume'] = Qf32([BVOLUME[n] for n,T in enumerate(BTYPE) if T==ty])
            polygons = MAP[k].xy()
        MAP[k]['centroid'] = Geometry([polygon_centre(P) for P in polygons],form='xy')   
    if 'Wbuildings' in MAP:
        k = 'workplacesB'
        MAP[k] = MAP['Wbuildings']['centroid']
        MAP[k]['count'] = MAP['Wbuildings']['volume']
    #Not used: return buildings.assign_to_polygons(MAP['Rbuildings'], MAP['pop_dens'].parcels)
            
def main_popdens(GLOBAL, VARS):
    MAP = GLOBAL['MAP']; NWK = GLOBAL['NWK']
    CENSUS = load_data(GLOBAL['PATHS']['CENSUS']) # unprotected!
    POP_GEO = GLOBAL['POP_GEO']
    POP_COUNTS = get_pop_counts(POP_GEO['parcel_codes'],CENSUS)
    if type(POP_COUNTS) is str:
        print('Errors encountered when parsing CENSUS file:')
        print(POP_COUNTS)
        END()
    print('Making PopDens...',end='')
    MAP['pop_dens'] = PopDens(POP_GEO,POP_COUNTS) #mount does nothing here
    print('done.')

def main_popsamp(GLOBAL, VARS):
    GEOG = GLOBAL['GEOGRAPHY']
    POP_SAMP_FILE = GLOBAL['PATHS']['ROOT_PATH']+GEOG['PROV']+'/'+GEOG['PROV']+'_'+\
        GEOG['CITY'].replace(' ','_')+'___pop_samp.json'
    MAP = GLOBAL['MAP']
    if not os_path_isfile(POP_SAMP_FILE):
        print("Placing population samples on roads...")
        P,W = GLOBAL['NWK']._place_residences(MAP['pop_dens'].give_G3(),
                res_m = GLOBAL['PHYSICAL']['pop_res_sample_m'])
        BOUND = MultiPolygon([sPolygon(p) for p in GLOBAL['MAP']['buffered_boundary'].xy()])
        F = [BOUND.contains(Point(p)) for p in P]
        P = [(float(p[0]),float(p[1])) for n,p in Qe(P) if F[n]]
        W = [float(w) for n,w in Qe(W) if F[n]]
        with open(POP_SAMP_FILE, 'w') as file:
            json.dump({'points':P, 'count':W}, file)
    else:
        print("Loading:",POP_SAMP_FILE)
        with open(POP_SAMP_FILE, 'r') as file:
            J = json.load(file)
            P,W = J['points'],J['count']
    MAP['residencesS'] = Geometry(P,form='xy')###
    MAP['residencesS']['count'] = W

# not used?
##def match_TR_street_names(GLOBAL, name_vec):
##    MAP = GLOBAL['MAP']; NWK = GLOBAL['NWK']; e = NWK.extra
##    if 'streetName' not in e: e['streetName'] = MAP.roads.property['streetName']
    




def main_anchors(GLOBAL, VARS):
    ###? Q(locals(),GLOBAL,VARS)
    MAP = GLOBAL['MAP']; NWK = GLOBAL['NWK']
    print("Making Anchors..")
    MK = list(MAP.optional.keys()) # dict changes during iteration!
    for k in MK:
        if k.startswith('workplaces'):
            NWK.Terminals(k,MAP[k])
            NWK.groupTerminals(k, GLOBAL['PHYSICAL']['workplace_group_m'])
            MAP['W'+k[10:]+'_anchor_links'] = NWK.get_anchor_links_G3(NWK.terminals[k])
        elif k.startswith('residences'):
            NWK.Terminals(k,MAP[k])



def main_overlays(GLOBAL, VARS):
    MAP = GLOBAL['MAP'],
    OVERLAYS = [('GTFS_shapes','thin_line',(0,0.75,0.75),'T'),
                ('GTFS_downsub','thin_line',(1,0.25,0.25),'T'),# has to go after GTFS_shapes
                ('analysis_boundary','thin_contour',(1,0,0),'V'),
                ('buffered_boundary','thin_contour',(0,1,0),'V'),
                ('municipal_boundary','thin_contour',(0,0,1),'V'),
                ('Rbuildings','thin_contour',(0,0,0),'B'),
                ('Wbuildings','thin_contour',(1,0,0),'B'),
                ('?buildings','thin_contour',(1,0.3,1),'B'),
                ('residencesS','points',(255,215,0),'[',False),#visible=False
                ('workplacesB','points',(255,215,0),'K',False),
                ('WB_anchor_links','thin_line',(255,215,0),'K',False),
                ('workplacesT','points',(255,215,0),';',False),
                ('WT_anchor_links','thin_line',(255,215,0),';',False),
                ('halo','thin_line',(255,0,0),'O')]
                #('existing_bikeways','thin_line',(0,0,0),'F'),
                #('proposed_bikeways','thin_line',(0,128,128),'F'),
                #('bikeways','thin_line',(64, 0, 192),'F')]
    #bikeway_types = [k for k in MAP.optional if k.startswith('bikeways_')]
    #if not bikeway_types: OVERLAYS.append(('bikeways','thin_line',(64, 0, 192),'F'))
    #else:
    #    for n,t in enumerate(bikeway_types): OVERLAYS.append((t,'thin_line',hsv_to_rgb(360*n/len(t),80,80),'F'))
    if 'bikeways_named' in MAP: 
        OVERLAYS += [('bikeways_named','thin_line',(0,1,0),'F'),
                     ('bikeways_unnamed','thin_line',(1,0,0),'F')]
    else:
        OVERLAYS += [('bikeways','thin_line',(0,0,1),'F')]
    GLOBAL['OVERLAYS'] = [interactive.GeometryOverlay(*O) for O in OVERLAYS if O[0] in GLOBAL['MAP']]  


#K = load_data(GLOBAL['PATHS']['ROOT_PATH']+'City of Kamloops/Trails_and_Bikeways/Trails_and_Bikeways.shp')

if __name__ == '__main__':
    X = MAIN(['CA']+PLACES[CHOICE])

    

    

