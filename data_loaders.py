## Format:
## Each line of the configuration file must contain only one parameter
## The parameter name must be entered exactly (case-sensitive) and without spaces.
## The parameter is expected to be followed by its assigned value, which may contain spaces.






from utils import *
from earth import *
from fileformats import *
from geometry2 import *
from requests import get as requests_get
from io import BytesIO as io_BytesIO
from zipfile2 import ZipFile as zipfile_ZipFile



# ZIP with encryption:
##import pyzipper
##secret_password = b'lost art of keeping a secret'
##with pyzipper.AESZipFile('new_test.zip',
##                         'w',
##                         compression=pyzipper.ZIP_LZMA,
##                         encryption=pyzipper.WZ_AES) as zf:
##    zf.pwd = secret_password
##    zf.writestr('test.txt', "What ever you do, don't tell anyone!")
##
##with pyzipper.AESZipFile('new_test.zip') as zf:
##    zf.pwd = secret_password
##    my_secrets = zf.read('test.txt')





def download_all_topog(dest_folder, tiles):
    DIRECTORIES = 0
    if DIRECTORIES: 
        directories = {tile['directory']:None for tile in tiles}
        print('Downloading topography directories...',end='')
        for d in directories:
            r = requests_get(d)
            if not r.ok: print(' failed',end='')
            else:
                print(' success',end='')
                directories[d] = r.text
        print('.')
    downloads = {}
    for tile in tiles:
        if tile['zip_file'] in downloads:
            if tile['url'] in downloads[tile['zip_file']]:
                downloads[tile['zip_file']][2].add(tile['topog_file'])
        else:
            downloads[tile['zip_file']] = (tile['directory'],tile['url'],set([tile['topog_file']]))
    for zipf,(directory,url,topogfs) in downloads.items():
        if DIRECTORIES:
            d = directories[directory]
            if d and zipf not in d:
                print('No such resource found online:', tile['zip_file'])
                continue # Tile does not exist (e.g., in ocean) 
        if sum([not os_path_isfile(dest_folder+topogf) for topogf in topogfs]):
            print('Downloading topography resource',zipf,'...',end='')
            time_sleep(1)
            r = requests_get(url)
            if not r.ok: print('failed.')
            else:
                z = zipfile_ZipFile(io_BytesIO(r.content))
                if z.testzip() is not None: print('failed.')
                else:
                    print('done.')
                    znl = z.namelist()
                    for topogf in topogfs:
                        if not os_path_isfile(dest_folder+topogf):
                            if not topogf in znl: print('Zip resource does not contain',topogf,'.')
                            else:
                                print('Found',topogf,'.')
                                z.extract(topogf,dest_folder)
    out = []
    for tile in tiles:
        F = dest_folder+tile['topog_file']
        L,R,B,T = tile['lrbt_deg']
        if os_path_isfile(F):
            out.append('TOPOGRAPHY      '+F+\
                       '\nTOPOG_LON       '+str(L)+'  '+str(R)+\
                       '\nTOPOG_LAT       '+str(B)+'  '+str(T))
    return '\n'.join(out)


def build_workplaces(data, fields):
    if isinstance(data,Table):
        coords = {}
        for C in 'LAT LON'.split():
            if fields.get(C) and fields[C] in data:
                coords[C] = data[fields[C]]
            else:
                print('Could not read '+C+' from workplaces.')
                return False
        out = Geometry(coords['LON'],coords['LAT'])
        if fields.get('COUNT') and fields['COUNT'] in data:
            out['count'] = Qf32(data[fields['COUNT']])
        else:
            print('Warning: no counts identified for workplaces: setting all counts to 1.')
            out['count'] = 1
        return out
    elif isinstance(data,dict):
        print('dict passed to data_loaders.build_workplaces()')
        print('this is probably a .shp - to implement')
        return False
    else:
        print('Invalid format passed to data_loaders.build_workplaces()')
        return False

def get_workplaces(config):
    C = {c[0]:c[1] for c in config}
    out = {}
    for k in 'LAT LON COUNT'.split():
        out[k] = C.get('WORKPLACES_'+k)
    return out
    
    

    





def get_topography(config):
    F = [c[0] for c in config]
    V = [c[1] for c in config]
    topogs = []
    for n,f in enumerate(F):
        if f == "TOPOGRAPHY": topogs.append((V[n],[None,None,None,None]))
        if f == "TOPOG_LON":
            if not topogs: return "TOPOG_LON must appear after TOPOGRAPHY."
            else:
                lons = get_all_floats(V[n])
                if len(lons) != 2: return 'Expecting exactly two values for TOPOG_LON, not "'+V[n]+'".'
                topogs[-1][1][0] = min(lons)
                topogs[-1][1][1] = max(lons)
        if f == "TOPOG_LAT":
            if not topogs: return "TOPOG_LAT must appear after TOPOGRAPHY."
            else:
                lats = get_all_floats(V[n])
                if len(lats) != 2: return 'Expecting exactly two values for TOPOG_LAT, not "'+V[n]+'".'
                topogs[-1][1][2] = min(lats)
                topogs[-1][1][3] = max(lats)
    return topogs

##def topography_polygons(topography, centre):
##    M = topography['data']
##    nx, ny = M.shape
##    x0 = topography['TOPO_MIN_LON']
##    y0 = topography['TOPO_MIN_LAT']
##    dx = (topography['TOPO_MAX_LON']-x0)/(nx-1)
##    dy = (topography['TOPO_MAX_LAT']-y0)/(ny-1)
##    print('Making topog. polygons...')
##    Polygons = []
##    Polygon_Data = []
##    X = [x0 + n*dx for n in range(1000,2001)]
##    Y = [y0 + m*dy for m in range(7000,9001)]
##    P = [deg2m(centre,[(x,y) for x in X]) for y in Y]
##    explode(P)
##    Polygon_Data = [float(M[1000+n,7000+m]/2000) for n in range(1000) for m in range(2000)]
##    explode(Polygon_Data)
##    Polygons = [[(P[m][n],P[m][n+1],P[m+1][n+1],P[m+1][n])] for n in range(1000) for m in range(2000)]
##    #        Polygons.append([[(cx,cy),(cx+dx,cy),(cx+dx,cy+dy),(cx,cy+dy)]])
##    #        Polygon_Data.append(X[n,m]/2000)
##    print('...done.')
##    return {'Polygons':Polygons,'Polygon_Data':Polygon_Data}



def get_outputs(config):
    F = [c[0] for c in config]
    V = [c[1] for c in config]
    if 'OUTPUTS' not in F: return 'outputs/'
    else: return V[F.index('OUTPUTS')]

##def get_map(config):
##    F = [c[0] for c in config]
##    V = [c[1] for c in config]
##    if 'OSM_XML_MAP' not in F: return False
##    else: return V[F.index('OSM_XML_MAP')]

def parse_config(config):
    out = []
    lines = config.splitlines()
    for l in lines:
        ls = l.strip()
        if ls:
            if ' ' in ls:
                i = ls.index(' ')
                out.append([ls[:i],ls[i:].lstrip()])
            else: out.append([ls,""])
    return out    

def get_paths(config):
    MANDATORY = []
    OPTIONAL = "WORKPLACES BUILDINGS BIKEWAYS_SHP TRUCK_ROUTES COLLISIONS TOPOGRAPHY OSM_MAP CITY_BOUNDARY CENSUS CENSUS_GEOMETRY".split()
    F = [c[0] for c in config]
    V = [c[1] for c in config]
    if "ROOT_PATH" not in F: ROOT_PATH = ""
    else: ROOT_PATH = V[F.index("ROOT_PATH")]
    missing = list(set(MANDATORY) - set(F))
    if missing:
        print("Missing mandatory files in configuration:")
        for m in missing: print(m)
        return False
    PATHS = {m:ROOT_PATH+V[F.index(m)] for m in MANDATORY}
    for O in OPTIONAL:
        if O in F: PATHS[O] = ROOT_PATH+V[F.index(O)]
    PATHS['ROOT_PATH'] = ROOT_PATH
    return PATHS
    
def get_cities(config, codes):
    F = [c[0] for c in config]
    V = [c[1] for c in config]
    Cities = []
    for n,f in enumerate(F):
        if f in codes:
            Cities.append([f,V[n]])
    return Cities

def get_prov_city(config, codes):
    F = [c[0] for c in config]
    V = [c[1] for c in config]
    for n,f in enumerate(F):
        if f in codes: return f,V[n]
    return None

def print_cities(Cities):
    return '+'.join(' '.join(C) for C in Cities)
















##def load_reswrk(City, DA_CODES, NW):
##    PD = load_population_density(City, DA_CODES)
##    NW.Map['ResPar'] = Geometry(PD['Polygons']) #mount
##    NW['ResPaP'] = PD['PopCount']
##    # Get Residence groups
##    if 1:
##        NW.ResParcels(res=50)
##        CR = m2deg(NW.Map.centre, NW['Res'])
##        WR = NW['ResPoP']
##    # Get Workplaces:
##    if City == 'Sherbrooke':
##        Emplois = load_data('data\Ville de Sherbrooke\Emplois.csv')
##        CW = list(zip(Emplois['Longitude'],Emplois['Latitude']))
##        WW = Emplois["Nombre"]
##    return locals()
