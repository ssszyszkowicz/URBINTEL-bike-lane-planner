from utils import *
from fileformats import *

KAMLOOPS = make_text_dict("""
WORKPLACE_BUILDINGS_FILE City of Kamloops/Building/Building.shp
WORKPLACE_BUILDINGS_TYPE TYPE
WORKPLACE_BUILDINGS_HEIGHT_m HEIGHT
WORKPLACE_BUILDINGS_AREA_m2 SHAPE_Area
""")

PRINCE_GEORGE = make_text_dict("""
WORKPLACE_BUILDINGS_FILE City of Prince George/Building_Outlines/Building_Outlines.shp
WORKPLACE_BUILDINGS_TYPE BuildingTy
WORKPLACE_BUILDINGS_HEIGHT_m BuildingHe
WORKPLACE_BUILDINGS_AREA_m2 Shape__Are
WORKPLACE_BUILDINGS_FLOORS FloorCount
""")

NANAIMO = make_text_dict("""
WORKPLACE_BUILDINGS_FILE City of Nanaimo/Cadastre/BUILDINGS.shp
WORKPLACE_BUILDINGS_TYPE TYPE
WORKPLACE_BUILDINGS_HEIGHT_m HEIGHT
WORKPLACE_BUILDINGS_AREA_m2 AREA
WORKPLACE_BUILDINGS_FLOORS FLOORS
""")

CITY = NANAIMO
CITY['ROOT_PATH'] = "C:/Users/sebas/Desktop/"

WB_FILE = CITY['ROOT_PATH']+CITY['WORKPLACE_BUILDINGS_FILE']
print("WB file found:",os_path_isfile(WB_FILE))
X = load_data(WB_FILE)
R = X['records']
print(R.head)
T = R[CITY['WORKPLACE_BUILDINGS_TYPE']]
histo(T)
H = R[CITY['WORKPLACE_BUILDINGS_HEIGHT_m']]
A = R[CITY['WORKPLACE_BUILDINGS_AREA_m2']]
try: F = R[CITY['WORKPLACE_BUILDINGS_FLOORS']]
except: F = None

r = Qr(R)
Floors = [F[n] if F and F[n] else (int(H[n]/3.0) if H[n] and H[n]>0 else 1) for n in r]
Volume = [Floors[n]*A[n] for n in r]

