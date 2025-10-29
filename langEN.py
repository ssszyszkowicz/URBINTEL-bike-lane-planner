from utils import *

def lonlat(lon,lat,short=False):
    form = "%."+"52"[bool(short)]+"f"
    return (form%abs(lat)+chr(176)+'NS'[lat<0],
            form%abs(lon)+chr(176)+'EW'[lon<0])

text = make_text_dict("""
LEGEND legend
LTS_1 LTS 1
LTS_2 LTS 2
LTS_3 LTS 3
LTS_4 LTS 4
DISMOUNT Dismount
FORBIDDEN Forbidden
UNKNOWN Unknown

MENU menu

VIEW view
TOPOGRAPHY topography
POPULATION population
WORKPLACES workplaces
GRID grid
SLOPE slope
LTS LTS

ANALYSIS analysis
SINGLE_ROUTE single route
ISOCHRONE isochrone
ALL_ROUTES all routes

CONFIG. config.
VISUAL_SCHEME visual scheme
""")
