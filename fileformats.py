from utils import *
from shapefile import Reader as shapefile_Reader, Writer as shapefile_Writer
from pathlib import Path as pathlib_Path
import earth
from pickle import load as pickle_load, dump as pickle_dump
#from hickle import load as hickle_load, dump as hickle_dump
from xmltodict import parse as xmltodict_parse
from json import load as json_load, dump as json_dump
from csv import reader as csv_reader, writer as csv_writer
from tifffile import imread as tifffile_imread
from geometry2 import *
from pandas import read_excel #needs xlrd
from io import StringIO






def shp_reader(shp_file):
    return shapefile_Reader(shp_file, encoding = "ISO-8859-1")

def shp_get_EPSG(shp_file):
    i = shp_file[::-1].index('.')
    prj_file = shp_file[:-i]+'prj'
    if not os_path_isfile(prj_file): return False
    else:
        with open(prj_file, 'r') as fPRJ: sPRJ = ' '.join(fPRJ.readlines())
        PROJ = re_findall('"([^"]*)"', sPRJ)[0]
        if PROJ in earth.EPSG_CODES: return earth.EPSG_CODES[PROJ]
        else:
            print('ERROR: Projection "'+PROJ+'" not in database!')
            return False
           
def read_fiona(file):
    print("Deprecated - not using Fiona/GDAL.")
    return False
##    import fiona
##    C = fiona.open(file)
##    print('Fiona found',len(C),'entries.')
##    geo_types = []
##    geos = []
##    fields = {}
##    for E in C:
##        for F in E['properties']:
##            if F not in fields:
##                fields[F] = []
##    for E in C:
##        G=E['geometry']
##        geo_types.append(G['type'])
##        geos.append(G['coordinates'][0]) # Reads the first polygon of a multipolygon
##        P = E['properties']
##        for F in fields:
##            if F in P: fields[F].append(P[F])
##            else: fields[F].append(None)
##    out = {'field_names':list(fields.keys())}
##    out['records'] = fields
##    out['geometry'] = geos
##    out['geo_types'] = geo_types
##    return out 

# Reads the first polygon of a multipolygon
def read_multipolygon_fiona(file):
    print("Deprecated - not using Fiona/GDAL.")
    return False
##    import fiona
##    C = fiona.open(file)
##    print('Fiona read',len(C),'entries.')
##    geo_types = []
##    geos = []
##    fields = {}
##    for E in C:
##        for F in E['properties']:
##            if F not in fields:
##                fields[F] = []
##    for E in C:
##        G=E['geometry']
##        geo_types.append(G['type'])
##        if G['type'] == 'Polygon':
##            geos.append(G['coordinates'])
##        elif G['type'] == 'MultiPolygon': # only import first contour:
##            geos.append(G['coordinates'][0])
##        else: return False
##        P = E['properties']
##        for F in fields:
##            if F in P: fields[F].append(P[F])
##            else: fields[F].append(None)
##    out = {'field_names':list(fields.keys())}
##    out['records'] = fields
##    out['geometry'] = geos
##    out['geo_types'] = geo_types
##    return out 



def stream_csv_from_excel(excel_name):
    X = read_excel(excel_name)
    stream = StringIO()
    X.to_csv(stream)
    return stream
    
def save_data(data,name,form=None,opt=None):
    return _load_save_data(data,False,name,form,opt)
def load_data(name,form=None,opt=None):
    return _load_save_data(None,True,name,form,opt)
def _load_save_data(data,LOAD,name,form=None,opt=None):
    F = pathlib_Path(name)
    if form == None: FORMAT = F.suffix.strip('.')
    else: FORMAT = form.lower()
    if LOAD and not F.is_file():
            print('Could not open file:')
            print(name)
            return False
    #### WATCH FOR ELIF CHAIN WHEN ADDING NEW FORMATS!!!!
    if FORMAT == 'shp':
        if LOAD: return read_shp_to_dict(name,opt)
        else: return write_shp_from_dict(data,name) 
    else:
        if LOAD: mode = 'r'
        else: mode = 'w'
        # These formats require explicit file opening:
        if FORMAT == 'uic':
            mode += 'b'
            reader = ui_raw_read
            writer = ui_raw_write
## Deprecate p/hickle due to security vulnerabilities - replace by .ui*
##        if FORMAT == 'pickle': 
##            mode += 'b' # binary mode for pickle!
##            reader = pickle_load
##            writer = pickle_dump
##        elif FORMAT == 'hickle':
##            # note: hickle opens in text mode, not binary!
##            reader = hickle_load
##            writer = hickle_dump
        elif FORMAT == 'txt':
            reader = None #TO_DO
            writer = None #TO_DO
        elif FORMAT in ['xls','xlsx']:
            mode += 'b'
            reader = lambda f: read_csv_to_table(stream_csv_from_excel(f))
            writer = None #TO_DO
        elif FORMAT == 'gml':
            reader = lambda f: xmltodict_parse(f.read())
            writer = None #TO_DO
        elif FORMAT == 'csv':
            ###if mode == 'w': mode == 'wb' # !
            reader = read_csv_to_table
            writer = write_table_to_csv
        elif FORMAT in ['jsn','json','geojson']:
            reader = json_load
            writer = json_dump
        elif FORMAT in ['tif','tiff']:
            mode += 'b'
            reader = tifffile_imread
            writer = None
        else:
            print('ERROR: unknown format: '+FORMAT)
            return False
        out = False    
        with open(name, mode=mode) as file:
            if LOAD:
                if reader: out = reader(file)
                else: print("Don't know how to read from ."+FORMAT+'!')
            else:
                if writer: out = writer(data,file)
                else: print("Don't know how to write to ."+FORMAT+'!')
        return out
if 0: # Test smart parsing of CSV: 
    X = load_data("data/OCTranspo/gtfs/stops.txt",'csv')


def read_image(file): # untested
    return PIL_Image.open(file)

class Table():       
    def __init__(self, ROWS):
        def count_cols(los):
            c = 0
            for s in los:
                if not s: break
                c += 1
            return c
        if len(ROWS) < 2: return False
        TABLE = ROWS
        # Assumed first row is HEADER(TITLES), then DATA.
        COLS = count_cols(TABLE[0])
        if COLS < 1: return False
        # Number of columns is first N non-empty strings in first row
        # Number of rows (excluding row 0 head) is up to first row
        # that has empty strings within <COLS>
        LEN = len(ROWS) - 1
        r = Qr(COLS)
        COLUMNS = [['']*LEN for n in r]
        for i,R in enumerate(TABLE[1:]):
            if count_cols(R) < 1: break
            for n in r:
                v = R[n]
                if v is None: v=''
                else:
                    try:
                        if float(v) == int(v): v = int(v)
                        else: v = float(v)
                    except:
                        try: v = float(v)
                        except: pass
                COLUMNS[n][i] = v
        HEAD = TABLE[0][0:COLS]
        DATA = {}
        for n, COLUMN in enumerate(COLUMNS): DATA[HEAD[n]] = COLUMN
        self._data = DATA
        self.cols = COLS
        self.len = LEN
        self.head = HEAD
        self.types = None
    def make_types(self):
        TYPES = {int:0,float:1,str:2}
        INV_TYPES = {v:k for k,v in TYPES.items()}
        self.types = [INV_TYPES[max(TYPES[type(i)] for i in self[key])] for key in self.head]
    def __getitem__(self,key):
        if type(key) is int: return [self._data[h][key] for h in self.head] # give a row
        else: return self._data[key] # give a column (or fail hash)
    def __setitem__(self,key,data):
        self._data[key] = data
        if key not in self.head: self.head.append(key)
    def __len__(self): return self.len
    def __contains__(self, item): return item in self.head
    def summary(self, max_set=10):
        print('TABLE SUMMARY WITH',len(self),'DATA ROWS:')
        if not self.types: self.make_types()
        for n, key in enumerate(self.head):
            col = self[key]
            ty = self.types[n]
            if ty in [int,float]:
                print('<'+key+'>',ty.__name__,min(col),'to',max(col))
            else:
                SET = list(set([str(i) for i in col]))
                SET.sort()
                print('<'+key+'> set, size',len(SET),':',';'.join(SET[:max_set]),'.'+'..'*(len(SET)>max_set))



        
SHP_TYPES = {1:'points', 3:'polylines', 5:'polygons',15:'polygonZ',25:'polygonM'}
def write_shp_from_dict(data,file):
    TY = list(SHP_TYPES.keys())
    TY.sort()
    ty = None
    for k in TY:
        if SHP_TYPES[k] in data:
            ty = k
            break
    if ty in [3,5]:
        shpout = shapefile_Writer(file,ty) # creates empty file
        Geo = [[[[float(v) for v in c] for c in C] for C in P] for P in data[SHP_TYPES[ty]]]
        if ty == 5:
            for g in Geo: shpout.poly(g) # only works for polygons(5): g is G3, CW:solids, CCW:holes.
        if ty == 3:
            # seems desired depth for polyline is also 4, not 3.
            #Geo = [[[float(v) for v in c] for c in L] for L in data[SHP_TYPES[ty]]]
            for g in Geo: shpout.line(g)
        if 'field_names' in data:
            for fn in data['field_names']:
                shpout.field(fn, "C","20") # 80:size could be dynamic - TO_DO
            for n in Qr(Geo):
                shpout.record(**{fn:str(data['records'][fn][n])[:40] for fn in data['field_names']})
        else: # needs empty records to save
            print('Error: writing .shp without records not implemented yet!')
            for n in Qr(Geo):
                shpout.record()
        shpout.close() # end writing to file
    else:
        print('Do not know how to write to .shp with keys:',data.keys())
        print('Shapefile type '+str(ty)+' not implemented yet.')
        return False
    return file



def shp_polygon_parts(points, parts): # G2->G3
    L = list(parts) + [len(points)]
    return tuple(points[L[n]:L[n+1]] for n in Qr(parts))


        
def read_shp_to_dict(file,opt=None):
    R = shp_reader(file)
    F = R.fields[1:] # first field is 'DeletionFlag' - garbage?
    FNs = [f[0] for f in F]
    out = {}
    out['field_names'] = FNs
    out['field_idxs'] = {fn:ix for ix, fn in enumerate(FNs)}
    shptype = SHP_TYPES[int(R.shapeType)]
    if opt:
        trans={'3347':Canada2deg}[opt.lower()]
        out[shptype] = tuple(trans(S.points) for S in R.iterShapes())
    else:
        if shptype == 'polygons': G = tuple(tuple(clean_contour(C) for C in shp_polygon_parts(S.points,S.parts)) for S in R.iterShapes()) #G4
        elif shptype == 'polylines': G = tuple(shp_polygon_parts(S.points,S.parts) for S in R.iterShapes()) #G4
        else: return 'Shapefile type not supported yet: '+shptype
        EPSG = shp_get_EPSG(file)
        if not EPSG: out[shptype] = G
        else: out[shptype] = earth.EPSG2deg(G,EPSG)
    out['geometry'] = out[shptype]
    out['records'] = Table([FNs]+[S for S in R.iterRecords()])
    #print('Imported shapefile with bbox:\n',R.bbox)
    #print('TODO: find and print new bbox.')
    #if opt: print('Transformed via coordinate system: ',opt)
##    else:
##        if min(R.bbox)>180.0:
##            BB = [(R.bbox[0], R.bbox[1]), (R.bbox[2], R.bbox[3])]
##            print('Guessing at projection: NAD83(epsg:3347)...')
##            from earth import Canada2deg
##            GBB = Canada2deg(BB)
##            GBB = [GBB[0][0],GBB[0][1],GBB[1][0],GBB[1][1]]
##            if max(abs(v) for v in GBB)<=180.0:
##                print('possible match: valid bbox found to be:\n',GBB)
##            else:
##                print('Projection failed: bbox out of bounds!')
    return out

def write_table_to_csv(table, file):
    W = csv_writer(file, lineterminator='\n',delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    W.writerow(table.head)
    for n in Qr(table):
        W.writerow([table[h][n] for h in table.head])
    
def read_csv_to_table(file):
    return Table([x for x in csv_reader(file)])













## TO_DO: Implement byte order: https://docs.scipy.org/doc/numpy-1.15.0/user/basics.byteswapping.html 
############ Custom UI Data Format #################################
# Codes a dict of arrays/strings
############ .uia - no zip, no crypt - internal use only ###########
# TYPE uint8: =0 EOF; +3 to +6 - 2^n bits; +16: uint; +32: int; +48:float; +64:string-utf encoding (+3,+4,+5 - 8/16/32 encodings)
# NAME 15 * char: string name - chars 1 to 255, end-pad with chr(0)
# For arrays:
    # NDIM uint8: number of array dimensions; skip for strings
    # SHAP Ndim * uint64: dimension lengths as seen in ndarray.shape
# For strings:
# DATA - string of bytes


#### DO NOT CHANGE, UNDER PAIN OF PAIN!
UI_NAME_LEN = 63
UI_FORMAT_ENDIAN = '<' #'<' is Windows-native
UI_UINT64 = UI_FORMAT_ENDIAN+'u8'

def ui_raw_read(bfile):
    data = {}
    TYPE = ord(bfile.read(1))
    while(TYPE):
        NAME = bfile.read(UI_NAME_LEN)
        if 0 in NAME:
            NAME = NAME[0:NAME.index(0)]
        key = ''.join(chr(c) for c in NAME)
        TYPE0 = TYPE//16
        TYPE1 = TYPE%16
        if TYPE0 == 0: break
        elif TYPE0 == 4 and 3<=TYPE1<=5: # string
            STR_ENCODING = 'utf'+str(2**TYPE1)
            L = np_fromfile(bfile, dtype=UI_UINT64, count=1)[0]
            #if L > 2**31: return False # 2GB max per object.
            data[key] = bfile.read(L).decode(STR_ENCODING)
        elif 0<TYPE0<4 and 3<=TYPE1<=6: # np_array
            DTYPE = UI_FORMAT_ENDIAN + " uif"[TYPE0] + str(2**(TYPE1-3))
            NDIM = np_fromfile(bfile, dtype=UI_UINT64, count=1)[0]
            if NDIM>32: return False # max numpy array dim
            SHAPE = []
            for n in Qr(NDIM): SHAPE += [np_fromfile(bfile, dtype=UI_UINT64, count=1)[0]]
            L = np_prod(SHAPE)
            #if L > 2**(31 - (TYPE1-3)): return False # 2GB max per object.
            data[key] = np_fromfile(bfile, dtype=DTYPE, count=L).reshape(SHAPE)
        else: return False
        TYPE = ord(bfile.read(1))
    return data
    
def ui_raw_write(data,bfile): # takes a dict of ndarrays or strings
    BLOCK_SIZE = 10**6
    STR_ENCODING = 'utf8'
    for k in data:
        for c in k:
            if not 0<ord(c)<128:
                print('Invalid name passed to ui_raw_write:',k)
                bfile.write((chr(0)).encode(STR_ENCODING))
                return False
    for k in data:
        NAME = k[0:UI_NAME_LEN]
        NAME += chr(0)*(UI_NAME_LEN-len(NAME))
        t = type(data[k])
        if t in [str,list,tuple]: # string (or list thereof)
            TYPE = 64 + 3 #utf8 STR_ENCODING
        elif t is np_ndarray: 
            tyname =  data[k].dtype.name
            TYPE = {'u':16,'i':32,'f':48}[tyname[0]] +\
                   {'8':3,'6':4,'2':5,'4':6}[tyname[-1]]
        bfile.write((chr(TYPE)+NAME).encode(STR_ENCODING))
        if TYPE>=64: # string
            S = data[k]
            L = 0
            bfile.write(np_array([0],UI_UINT64).tobytes(order='C'))
            if type(S) is str: S = [S]
            for s in S:
                B = s.encode(STR_ENCODING)
                L += len(B)
                bfile.write(B)
            bfile.seek(-L-8,1)
            bfile.write(np_array([L],UI_UINT64).tobytes(order='C'))
            bfile.seek(+L,1)
        else: # array
            shape = data[k].shape
            bfile.write(np_array([len(shape)]+list(shape),UI_UINT64).tobytes(order='C'))
            bfile.write(data[k].astype(UI_FORMAT_ENDIAN+tyname[0]+str(2**((TYPE%16)-3))).tobytes(order='C'))###
    bfile.write(chr(0).encode(STR_ENCODING))
                
if 0: #test huge data import:
    with open('test.uia', mode='rb') as file:
        while(1):
            s = file.read(10*10**9)
            print(s)
            input()

if 0: # test .uia
    D = {'test1': Qu32([3,1,4]),
         'test2': Qi8([[5,-6],[-4,3]]),
         'test3': 'hello!',
         'test12345678901234567890': """QC_Montréal
QC_Lévis
QC_Quebec City
QC_Trois-Rivières
QC_Saint-Jérôme
""".split('\n')}
    save_data(D, 'test.uia')
    D_ = load_data('test.uia')
