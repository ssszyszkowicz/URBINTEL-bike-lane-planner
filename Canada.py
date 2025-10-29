from utils import *
from fileformats import *
from geometry2 import *
from earth import *
from osmmap import OSM_Map
from fuzzywuzzy import process as fuzzywuzzy_process
from data_loaders import *



def get_topog_files(boundary):
    URL_ROOT = 'http://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdsm_mnsc/'
    ###URL_ROOT = 'http://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/'
    BOUNDARY = MultiPolygon(sPolygon(P) for P in boundary)
    l,b,r,t = BOUNDARY.bounds
    # All Canada CDSM Array:
    W,E,S,N = (-144,-48,+40,+60)
    w,h = (0.25,0.25)
    if l<W or r>E or b<S or t>N: return 'Topography request out of bounds for Canada-south-of-60.'
    L = int((l-W)/w)
    R = int((r-W)/w)
    B = int((b-S)/h)
    T = int((t-S)/h)
    tiles = []
    for X in Qr(L,R+1):
        for Y in Qr(B,T+1):
            lrbt = (W+X*w,W+X*w+w,S+Y*h,S+Y*h+h)
            if sPolygon(make_box(lrbt=lrbt)).intersects(BOUNDARY):
                tiles.append({'TILE':(X,Y),'lrbt_deg':lrbt})
    # Format is XXYLNND
    # XX is 00 to 11 - E to W
    # Y is 0 to 4 - S to N
    # L is 'A' to 'P' on a 4x4 grid - "snake pattern"
    # NN is 01 to 16 on a 4x4 grid - "snake pattern"
    # D is 'W' or 'E'
    SNAKE = "DCBAEFGHLKJIMNOP"
    for tile in tiles:
        X,Y = tile['TILE']
        vXX = 11-X//32
        CODE = '0'*(vXX<10)+str(vXX)
        vY = Y//16
        CODE += str(vY)
        vL = SNAKE[(X%32)//8 + 4*((Y%16)//4)]
        CODE += vL
        vNN = ord(SNAKE[(X%8)//2 + 4*(Y%4)]) - ord('A') + 1
        CODE += '0'*(vNN<10)+str(vNN)
        vD = 'WE'[X%2]
        CODE += vD
        print(X,Y,CODE,'   ',end='')
        BASE_FILE = CODE[:-1]+'_cdsm_final' ###'_cdem_final'
        URL_PATH = URL_ROOT+CODE[0:3]+'/'
        tile['zip_file'] = BASE_FILE+'.zip'
        tile['url'] = URL_PATH+tile['zip_file']
        tile['topog_file'] = BASE_FILE+'_'+CODE[-1].lower()+'.tif'
        tile['directory'] = URL_PATH
    print('\n')
    return tiles


# Problems with CDEM: Tiles empty (Kamloops), or halved (Nanaimo)
# Also, sampling gives strange results.
def ___CDEM_get_topog_files(boundary):
    URL_ROOT = 'http://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/'
    BOUNDARY = MultiPolygon(sPolygon(P) for P in boundary)
    l,b,r,t = BOUNDARY.bounds
    # All Canada CDSM Array:
    W,E,S,N = (-144,-48,+40,+68)
    w,h = (2.0,1.0)
    if l<W or r>E or b<S or t>N: return 'Topography request out of bounds for Canada-south-of-68.'
    L = int((l-W)/w)
    R = int((r-W)/w)
    B = int((b-S)/h)
    T = int((t-S)/h)
    tiles = []
    for X in Qr(L,R+1):
        for Y in Qr(B,T+1):
            lrbt = (W+X*w,W+X*w+w,S+Y*h,S+Y*h+h)
            if sPolygon(make_box(lrbt=lrbt)).intersects(BOUNDARY):
                tiles.append({'TILE':(X,Y),'lrbt_deg':lrbt})
    # Format is XXYLNND
    # XX is 00 to 11 - E to W
    # Y is 0 to 6 - S to N
    # L is 'A' to 'P' on a 4x4 grid - "snake pattern"
    SNAKE = "DCBAEFGHLKJIMNOP"
    for tile in tiles:
        X,Y = tile['TILE']
        vXX = 11-X//4
        CODE = '0'*(vXX<10)+str(vXX)
        vY = Y//4
        CODE += str(vY)
        vL = SNAKE[(X%4) + 4*(Y%4)]
        CODE += vL
        print(X,Y,CODE,'   ',end='')
        BASE_FILE = 'cdem_dem_'+CODE
        URL_PATH = URL_ROOT+CODE[:-1]+'/'
        tile['zip_file'] = BASE_FILE+'_tif.zip'
        tile['url'] = URL_PATH+tile['zip_file']
        tile['topog_file'] = BASE_FILE+'.tif'
        tile['directory'] = URL_PATH
    print('\n')
    return tiles

if 0: # Get all topography for a city
    from data_loaders import download_all_topog
    X = load_data('data/PE/PE_Charlottetown.json')
    B = Geometry(X['buffered_boundary'],form='xy')
    TT = get_topog_files(B(X['centre']).lonlat())
    disp(TT)
    C = download_all_topog('data/Canada/topography/',TT)
    print(C)
    T = get_topography(parse_config(C))
    disp(T)



PROVINCE_NAMES = {'ON':'Ontario', 'BC':'British Columbia', 'PE':'Prince Edward Island', 'QC':'Quebec',
                  'MB':'Manitoba','SK':'Saskatchewan','AB':'Alberta','NB':'New Brunswick',
                  'NS':'Nova Scotia','NL':'Newfoundland and Labrador','YT':'Yukon',
                  'NT':'Northwest Territories','NU':'Nunavut',
                  'NY':'New York'} # Move to US when ready.

PROVINCE_CENTERS = {#lat, lon -farily arbitrary should be "good-enough"
    'ON':(-80,45),
    'BC':(-122,51),
    'PE':(-63,46),
    'QC':(-73,47),
    'MB':(-98,51),
    'SK':(-106,52),
    'AB':(-113,53),
    'NB':(-66,47),
    'NS':(-63,45),
    'NL':(-57,51),
    'YT':(-135,61),
    'NT':(-114,62),
    'NU':(-69,64),
    'NY':(-74,41)} # Move to US when ready.
    


PROVINCE_CODES = {v:k for k,v in PROVINCE_NAMES.items()}





PROVINCE_CENSUS_CODE = {'NL':10,'PE':11,'NS':12,'NB':13,'QC':24,'ON':35,'MB':46,'SK':47,'AB':48,'BC':59,'YT':60,'NT':61,'NU':62}

REMOVE_ACCENTS = {} # Remove French accents from city name

CANADA_GEOMETRY_FIELDS = {'province':'PRUID','census_parcels':'DAUID'}


def get_city_geometry(city, native_provinces, buffered_provinces, geometry_file,
                      buffer_m = 9000, boundary_contours = None,
                      prov_codes = PROVINCE_CENSUS_CODE, field_names = CANADA_GEOMETRY_FIELDS):
    print('Canada.get_city_boundary() is an unprotected offline function.')
    LDA = shp_reader(geometry_file)
    print("Found population geometry file with %d shapes."%len(LDA))
    EPSG = shp_get_EPSG(geometry_file)
    fields = {f[0]:n-1 for n,f in enumerate(LDA.fields)}
    fPRUID = fields[field_names['province']]
    fDAUID = fields[field_names['census_parcels']]
    # Get all parcels belonging to city
    PRUIDs = [int(prov_codes[pr]) for pr in native_provinces]
    Parcels = {}
    if boundary_contours:
        municipal_boundary = Geometry(boundary_contours)
    else:
        for n in Qr(LDA):
            R = LDA.record(n)
            match = False
            if int(R[fPRUID]) in PRUIDs:
                for F in R:
                    if F == city:
                        match = True
                        break
            if match:
                sh = LDA.shape(n)
                Parcels[int(R[fDAUID])] = Geometry([clean_contour(c) for c in shp_polygon_parts(EPSG2deg(sh.points,EPSG),sh.parts)]) #G3
                #if len(Parcels)%100==0: print('Found',len(Parcels),'parcels.')
        municipal_boundary = dilate_and_merge([par.lonlat()  for _,par in Parcels.items()]) #G4->G3
        # For Canada, throw out all but the biggest contour:
        municipal_boundary = Geometry([municipal_boundary[np_argmax([sPolygon(C).area for C in municipal_boundary])]]) #G3
    # get centre from municipal boundary:
    l,r,b,t = municipal_boundary.get_lrbt('deg')
    centre = ((l+r)/2,(b+t)/2)
    municipal_boundary(centre)
    # Buffer boundary
    buffered_boundary = Geometry(dilate_and_merge([municipal_boundary.xy()],buffer_m),form='xy') #G4->G3
    buffered_boundary(centre)
    lrbt_deg = buffered_boundary.get_lrbt('deg')
    # Add all parcels that intersect with buffer
    BUFFER = boundary_sPolygons(buffered_boundary.lonlat())
    PRUIDs = [int(prov_codes[pr]) for pr in buffered_provinces]
    for n in Qr(LDA):
        R = LDA.record(n)
        if int(R[fPRUID]) in PRUIDs:
            dauid = int(R[fDAUID]) 
            if dauid not in Parcels:
                sh = LDA.shape(n)
                l,b,r,t = sh.bbox
                X,Y = list(zip(*EPSG2deg([(l,b),(l,t),(r,b),(r,t)],EPSG)))
                if lrbt_and_lrbt(lrbt_deg, [min(X),max(X),min(Y),max(Y)]):
                    match = False
                    G3 = shp_polygon_parts(EPSG2deg(sh.points,EPSG),sh.parts)
                    shape = boundary_sPolygons(G3)
                    for A in BUFFER:
                        for B in shape:
                            if A.intersects(B):
                                match = True
                                print(dauid)###
                                break
                        if match == True: break
                    if match == True:
                        Parcels[int(R[fDAUID])] = Geometry(G3) #G3
                        #if len(Parcels)%100==0: print('Found',len(Parcels),'parcels.')
    print('Found',len(Parcels),'parcels.')
    # Reformat parcels:
    parcel_codes = tuple(Parcels.keys())
    parcels = [contours_to_solids(Parcels[pc](centre).xy()) for pc in parcel_codes] #G4
    analysis_boundary = dilate_and_merge(parcels) #G4->G3
    return {'city':city,'provinces':buffered_provinces,
            'centre':centre,
            'lrbt_deg':lrbt_deg,
            'municipal_boundary':municipal_boundary.xy(),
            'buffer_m':buffer_m,
            'buffered_boundary':buffered_boundary.xy(),
            'analysis_boundary':analysis_boundary,#sic, no .xy()
            'parcel_codes':parcel_codes,
            'parcels':parcels}


# should match the column names in census data file:
CANADA_CENSUS_FIELDS = {'census_parcels':'DAUID','population_count':'Pop2016'} 

# production code - error checking - yes.
def get_pop_counts(parcel_codes, pop_table,field_names=CANADA_CENSUS_FIELDS):
    # Get data columns
    def error(field, fields):
        errors = 'Could not find "'+field+'" column in CENSUS.'
        match, rating = fuzzywuzzy_process.extractOne(field,fields)
        if rating>49: errors += '\nDid you mean "'+match+'"?'
        return errors
    POP = {c:None for c in parcel_codes}
    PDA = pop_table
    errors = ""
    CPfield = field_names['census_parcels']
    if CPfield in PDA: GC = PDA[CPfield]
    else: errors += error(CPfield, PDA.head)
    PCfield = field_names['population_count']
    if PCfield in PDA: PC = PDA[PCfield]
    else: errors += '\n'*bool(errors)+error(PCfield, PDA.head)
    if errors: return errors
    # Get data:
    DIGITS = set('1234567890')
    for n,code in enumerate(GC):
        if type(code) is str:
            if not code: continue
            if set(code)-set(DIGITS): continue
            code = int(code)
        if code in POP: 
            if POP[code] is not None: return 'Duplicate census code: "'+str(code)+'" in CENSUS.'
            pop = PC[n]
            # In theory, pop may be float if Table parsed it as such.
            if type(pop) is str:
                if not pop.strip(): pop = 0 # This happens in Canada-2016 Census
                else:
                    if set(pop)-set(DIGITS): return 'Invalid population count: "'+pop+'" in CENSUS.'
                pop = int(pop)
            POP[code] = pop
    POPS = [POP[c] for c in parcel_codes]
    missing = POPS.count(None)
    if missing: return 'Missing population counts for census code "'+\
                       str(parcel_codes[POPS.index(None)])+'"'+\
                       (missing>1)*(' and '+str(missing-1)+' others')+' in CENSUS.'
    return POPS

            
    
            
###### MAKE TOP 100 CITIES IN CANADA

if 0: # Test get_pop_counts()
    PATHS = {}
    PATHS['CENSUS_DATA'] = 'data/Canada/DAs/DApops.csv'
    POP_TABLE = load_data(PATHS['CENSUS_DATA'])
    PARCEL_CODES = load_data('data/PE/PE_Charlottetown.json')['parcel_codes']
    X = get_pop_counts(PARCEL_CODES, POP_TABLE)
    print(X)

BUFFERED_PROVINCES = {'Ottawa':['QC'],'Gatineau':['ON']}











                  
if 0: # Get 101 Cities.
    print(NOW())
    PATHS = {}
    PATHS['CENSUS_GEOMETRY'] = 'data/Canada/DAs/lda_000a16a_e.shp'
    CITIES_FILE = 'data/Canada/100 cities.csv'
    Cities = load_data(CITIES_FILE)
    Mun = Cities['Municipality']+['Charlottetown']
    Prov = Cities['Province']+['Prince Edward Island']
    for n in Qr(12,60):#(71-1,-1,-1):#len(Mun)
        C = Mun[n].strip()
        P = PROVINCE_CODES[Prov[n].strip()]
        OUT_FILE = 'data/'+P+'/'+file_city_name(P,C)+'.json'
        #if os_path_isfile(OUT_FILE): continue
        print(n,P,C,file_city_name(P,C))
        native_provinces = [P]
        buffered_provinces = [P]
        if C in BUFFERED_PROVINCES:
            buffered_provinces += BUFFERED_PROVINCES[C]
        CG = get_city_geometry(C, native_provinces, buffered_provinces,
                               PATHS['CENSUS_GEOMETRY'], buffer_m = 9000)
        save_data(CG,OUT_FILE)
        print(NOW())
    












    
        
def load_population(cities, geometry_file, population_file, prov_codes = PROVINCE_CENSUS_CODE, buffer_r_m = 9000):
    Cities = []
    for City in cities:
        if City[0] in prov_codes:
            Cities.append([prov_codes[City[0]],City[1]])
        else:
            return "Unknown geographical code: "+City[0]
     # Note: all shapes in lda....shp are POLYGONs
    if os_path_isfile(geometry_file):
        LDA = shp_reader(geometry_file)
    if not LDA.numRecords: return ""
    EPSG = shp_get_EPSG(geometry_file)
    fields = {f[0]:n-1 for n,f in enumerate(LDA.fields)}
    fPRUID = fields['PRUID']
    fDAUID = fields['DAUID']
    DAUIDs = []
    shapes = {}
    for n in Qr(LDA):
        R = LDA.record(n)
        match = False
        for City in Cities:
            if City[0] == int(R[fPRUID]):
                for F in R:
                    if F == City[1]:
                        match = True
                        break
            if match == True: break
        if match:
            dauid = int(R[fDAUID])
            sh = LDA.shape(n)
            shapes[dauid] = shp_polygon_parts(EPSG2deg(sh.points,EPSG),sh.parts) 
            DAUIDs.append(dauid)
    if not DAUIDs: return False
    print('%d shapes successfully matched'%len(DAUIDs))
    ############
    POP = {d:None for d in DAUIDs}
    PDA = load_data(population_file) 
    GC = PDA['Geographic code']
    PO = PDA['Population, 2016']
    for n,code in enumerate(GC):
        if code == None: dauid = -1
        else: dauid = int(code)
        if dauid in POP:
            s = str(PO[n])
            if not s: POP[dauid] = 0
            elif not set(s)-set('1234567890'): POP[dauid] = int(PO[n])
            else:
                print("Invalid population count:",s)
                return False
    print('Population counts successfully loaded.')
    polygons = [shapes[d] for d in DAUIDs]
    PP = Geometry(polygons)()
    PPc = PP.get_centre()
    PPxy = PP.xy()
    boundary = [m2deg(PPc, c) for c in dilate_and_merge(PPxy, buffer_r_m)]
    return {'pop':[POP[d] for d in DAUIDs],
            'polygons':PP, 'boundary':boundary}







######## TESTS ############

if 0:
    S1 = []
    GG = []
    LDA = shapefile_Reader(PATHS['FILE_ROOT']+PATHS['LDA'],encoding = "ISO-8859-1")
    for n in Qr(LDA):
        sh = LDA.shape(n)
        if len(sh.parts) > 1:
            points = EPSG2deg(sh.points, 3347)
            PP = shp_polygon_parts(points,sh.parts)
            O = [int(not LinearRing(C).is_ccw) for C in PP]
            print(n, LDA.record(n)['DAUID'],O)
            S1.append(n)
            G = Geometry(PP)
            G(points[0])
            GG.append(G)
    POLYGONS = []
    for n in Qr(30):
        for m in Qr(32):
            i = m+n*32
            if i < len(GG):
                G = GG[i]
                B = G.get_lrbt_m()
                s = 500/max([abs(x) for x in B])
                POLYGONS.append([translate_points(scale_points(xy,s),m*1000,n*1000) for xy in G.xy()])
    Map = OSM_Map({'lrbt_deg':[0]*4})
    Map.lrbt_m = [-500, 31500, -500, 29500]
    draw(Map, Polygons = POLYGONS, Polygon_Data = [0.1]*len(POLYGONS))

if 0:
    X = load_population_Canada(Cities, PATHS)
    print('Found %d DAs'%len(X['polygons']))
    M = draw_pop_density(**X)

    




######## OLD CODE ########################


##
##if 0: #Reduce DAs to necessary (9km, POPCOUNT, DAUID):
##    City = 'Sherbrooke'
##    BUFFER = sPolygon(load_city_buffer(City)[0])
##    DAs = load_data('data/Canada/DAs/'+City+'.shp')
##    SEL = Selection(len(DAs['polygons']))
##    for item in DAs['records'].head:
##        SEL[item] = DAs['records'][item]
##    SEL['*polygons*'] = DAs['polygons']
##    SEL([BUFFER.intersects(sPolygon(P)) for P in DAs['polygons']])
##    out={}
##    out['polygons'] = SEL['*polygons*']
##    out['records'] = {'POPCOUNT':SEL['POPCOUNT'],'DAUID':SEL['DAUID']}
##    out['field_names'] = ['POPCOUNT','DAUID']
##    save_data(out, 'data/Canada/DAs/'+City+'.shp')
##
##
##def get_geography(City, CENSUS_FILE):
##    _DAGEO = load_data(CENSUS_FILE)
##    _POLYGONS = _DAGEO['polygons']
##    _City = City.copy()
##    del _City['City']
##
##def load_population_density(City, PATHS):
##    Name = City['City']
##    POP_IN = PATHS['FILE_ROOT']+PATHS['PDA']
##    C = PATHS['FILE_ROOT']+Name+'.shp'
##    if not os_path_isfile(C):
##        _DAGEO = get_geography(City,PATHS['FILE_ROOT']+PATHS['LDA'])
##    else:
##        _DAGEO = load_data(C)
##    print(len(_DAGEO))
##    XX=XX
##    _POLYGONS = _DAGEO['polygons']
##    DAUID = []
##    Regions = {}
##    for nn,dauid in enumerate(_DAGEO['records']['DAUID']):
##        if int(dauid)//10000 in DACODES:
##            Regions[int(dauid)] = _POLYGONS[nn]
##            DAUID.append(dauid)
##    ###
##    _DADATA = load_data(POP_IN)
##    _DAUIDS = _DADATA['Geographic code']
##    _DAPop  = _DADATA['Population, 2016']
##    #Faulty?#_DAAkm2 = _DADATA['Land area in square kilometres, 2016']
##    Census = {}
##    for n, dauid in enumerate(_DAUIDS):
##        if dauid//10000 in DACODES:Census[dauid] = _DAPop[n]
##    ###
##    AREA = [];DAPOLY = [];POPDENS = [];POPCOUNT = []
##    for dauid in Regions:
##        r = Regions[dauid]
##        A = sPolygon(deg2m(None,Regions[dauid])).area*10**-6
##        AREA.append(A)
##        POPCOUNT.append(Census[dauid])
##        POPDENS.append(Census[dauid]/A)
##        DAPOLY.append(Regions[dauid])
##    max_POPDENS = 10000.0#max(POPDENS)
##    VIZDENS = [min(1.0, (d/max_POPDENS)**0.5) for d in POPDENS]
### Use this to save a subset of the larger DA .shp
####    LDA_OUT = ROOT+City+'.shp'
####    save_data({'polygons':[Regions[d] for d in DAUID],
####               'field_names':['DAUID','POPCOUNT','AREA','POPDENS'],
####               'records':{'DAUID':DAUID,'POPCOUNT':POPCOUNT,'AREA':AREA,'POPDENS':POPDENS}},
####              LDA_OUT)
##     
##    return {'Polygons':DAPOLY, 'PopCount':POPCOUNT, 'Polygon_Data':VIZDENS, 'PopDens':POPDENS}
##
