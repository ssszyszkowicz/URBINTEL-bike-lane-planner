from utils import *
from qcommands import *
import langEN
from colorsys import hsv_to_rgb, rgb_to_hsv
from PIL.Image import LANCZOS as PIL_Image_LANCZOS
from geometry2 import CL_HASH_LINES_CODE
from UrbNet import CL_ALL_TO_ALL_CODE
from pyopencl import Program as cl_Program, get_platforms as cl_get_platforms, Context as cl_Context
from shaders import _magma_data


URBINTEL_RGB = [197/255, 143/255, 86/255]
###[247/255, 181/255, 0] # Traffic Yellow
URBINTEL_BACK_RGB = [209/255, 198/255, 187/255]

### OPENCL CONFIG ###
CL_PLATFORM = cl_get_platforms()[0]
CL_DEVICE = CL_PLATFORM.get_devices()[0] # should be 2 - 1 GPU, 1 CPU, but I see only GPU now.
CL_CONTEXT = cl_Context([CL_DEVICE])
# ! IF you change the context, you have to recompile the programs (run-time)!





# CL sandbox for testing CL concepts:
CL_SANDBOX_CODE = ('SANDBOX', """
__kernel void A(
const uint L,
__global uint *V
)
{
uint I = get_global_id(0);
V[I] += I;
}""")

def build_all_cl_code(COMPUTE):
    for NAME,CODE in COMPUTE['CODE']:
        B = cl_Program(COMPUTE['CL_CONTEXT'],CODE).build()
        COMPUTE['BUILT'][NAME] = B

COMPUTE = {'CL_CONTEXT':CL_CONTEXT,
           'CODE': [CL_SANDBOX_CODE, CL_ALL_TO_ALL_CODE, CL_HASH_LINES_CODE]+CL_STACKS_CODES,
           'BUILT':{},
           'BUILD_ALL':build_all_cl_code,
           'START_TIME':time_time()}


##LANGUAGE = {'LANGS':'EN FR PL'.split(),
##            'LANG_NAMES': 'English français polski'.split(),
##            'LANG':'EN',
##            'TEXT': {'EN':langEN.text, 'FR':langFR.text, 'PL':langPL.text},
##            'FUNCS': {'EN':{'lonlat':langEN.lonlat},
##                      'FR':{'lonlat':langFR.lonlat},
##                      'PL':{'lonlat':langPL.lonlat}}}

LANGUAGE = {'LANGS':['EN'],
            'LANG_NAMES': ['English'],
            'LANG':'EN',
            'TEXT': {'EN':langEN.text},
            'FUNCS': {'EN':{'lonlat':langEN.lonlat}}}

GEOGRAPHY = {'ROAD_BOUNDARY_NAME':'analysis_boundary',
             'POPULATION_BOUNDARY_NAME':'buffer_boundary',
             'WORKPLACE_BOUNDARY_NAME':'municipal_boundary'}

def brand_colour(colour,sat=1.0,bright=1.0):# palette value 0-23
    return tuple(hsv_to_rgb(5/360+colour/24,sat,min(1,bright*0.93)))

class Layer:
    def __init__(self, TYPE, value, rgb, legend_name = '', button=None, visible = True, draw_order=0):
        self.type = TYPE
        self.value = value
        self.rgb = rgb
        self.legend_name = legend_name
        self.button = button
        self.visible = visible
        self.draw_order = draw_order
    def jsn(self):
        return (self.type, self.value, tuple(self.rgb), str(self.legend_name), str(self.draw_order))



LAYERS = {k:[] for k in """testingA testingB testingC LTS oneway has_duration 
hasHouses cycleInfra fwd_best_infra bwd_best_infra vis_best_infra HEAT8 SLOPE motor_allowed
shp_infra_coverage vis_candidate_infra vis_clamp vis_PRIORITY fwd_PRIORITY bwd_PRIORITY
vis_thread""".split()}

for T in 'ABC':
    LAYERS['testing'+T].append(Layer('road', 0, [0]*3, 'False', 1))
    LAYERS['testing'+T].append(Layer('road', 1, [1,0,0], 'True', 2))
    
def AddLayers(name, LAYERS, entries):
    LAYERS[name] = []
    for entry in entries:
        LAYERS[name].append(Layer('road', *entry))

AddLayers('passageType',LAYERS,
          [(0, [0]*3, 'Regular', 1), (1, [1,0,0], 'Multiple', 2)] +\
          [(ord(item[0]), brand_colour(6+n*6), item.capitalize(), n+3) \
           for n, item in enumerate('bridge ferry tunnel'.split())])

GENERIC_LAYER = [Layer('road',n,[[0,0,0],[0,0,1],[1,0,0]][n],str(n),n) for n in Qr(3)]
#LAYERS['__sidewalk'] = GENERIC_LAYER
#LAYERS['__bicycle_road'] = GENERIC_LAYER
LAYERS['bike_access'] = GENERIC_LAYER
LAYERS['motor_allowed'] = GENERIC_LAYER
LAYERS['force_dismount'] = GENERIC_LAYER
LAYERS['is_area'] = GENERIC_LAYER
LAYERS['has_name'] = GENERIC_LAYER
LAYERS['main_net'] = GENERIC_LAYER
LAYERS['vis_candidate_infra'] = GENERIC_LAYER

LAYERS['shp_infra_coverage'].append(Layer('road', 0, [0]*3, '0%%', 0))
for n in range(1,12):
    LAYERS['shp_infra_coverage'].append(Layer('road', n, brand_colour(16-2*n), '%d0%%'%n, 1))




LAYERS['has_duration'].append(Layer('road', 0, [0]*3, 'No Duration', 1))
LAYERS['has_duration'].append(Layer('road', 1, [1, 0.2, 0.2], 'Has Duration', 2))

LAYERS['hasHouses'].append(Layer('road', 0, [0]*3, 'No Houses', 1))
LAYERS['hasHouses'].append(Layer('road', 1, [0, 0.7, 0], 'Has Houses', 2))

LAYERS['cycleInfra'].append(Layer('road', 0, [0]*3, 'No Infra', 1))
LAYERS['cycleInfra'].append(Layer('road', 1, [0, 0, 1], 'Lane', 2))
LAYERS['cycleInfra'].append(Layer('road', 2, [0, 1, 0], 'Track', 3))
LAYERS['cycleInfra'].append(Layer('road', 3, [1, 0, 0], 'Both!?', 4))

LAYERS['fwd_best_infra'].append(Layer('road', 0, [0.5]*3, 'nothing', 0, draw_order=5))
LAYERS['fwd_best_infra'].append(Layer('road', 1, [0,0,1], 'shared', 1, draw_order=4))
LAYERS['fwd_best_infra'].append(Layer('road', 2, [0,0.5,1], 'permissive', 2, draw_order=3))
LAYERS['fwd_best_infra'].append(Layer('road', 3, [0,1,1], 'lane', 3, draw_order=2))
LAYERS['fwd_best_infra'].append(Layer('road', 4, [1,0,0], 'track', 4, draw_order=1))
LAYERS['fwd_best_infra'].append(Layer('road', 5, [1,1,0], 'crossing', 5, draw_order=0))
for k in Qmx('bwd_best_infra vis_best_infra fwd_osm_infra bwd_osm_infra fwd_shp_infra bwd_shp_infra'):
    LAYERS[k] = LAYERS['fwd_best_infra']

LAYERS['vis_clamp'].append(Layer('road', 5, [0,0,1], 'Good', 5))
LAYERS['vis_clamp'].append(Layer('road', 4, [0,0.75,0], 'Moderate', 4))
LAYERS['vis_clamp'].append(Layer('road', 3, [0.75,0.75, 0], 'Poor', 3))
LAYERS['vis_clamp'].append(Layer('road', 2, [1,0.5,0], 'Dismount', 2))
LAYERS['vis_clamp'].append(Layer('road', 1, [1,0,0], 'Demanding', 1))




LAYERS['oneway'].append(Layer('road', 0, [0]*3, 'Two Way', 1))
LAYERS['oneway'].append(Layer('road', 1, [0, 0.7, 0], 'One Way', 2))
LAYERS['oneway'].append(Layer('road', -1, [1,0,0], 'Reverse', 3))

LAYERS['LTS'].append(Layer('road', 1, brand_colour(14), 'LTS 1 (safest)', 1, draw_order=1))
LAYERS['LTS'].append(Layer('road', 2, brand_colour(6,bright=0.75), 'LTS 2', 2, draw_order=2))
LAYERS['LTS'].append(Layer('road', 3, brand_colour(2,bright=1.1), 'LTS 3', 3, draw_order=3))
LAYERS['LTS'].append(Layer('road', 4, brand_colour(23,bright=1.1), 'LTS 4 (hardest)', 4, draw_order=4))
LAYERS['LTS'].append(Layer('road', 0, brand_colour(16,0.2), 'Dismount', 0, draw_order=5))
LAYERS['LTS'].append(Layer('road', 6, brand_colour(18,0.5), 'Hard & Dismount', 6, draw_order=6))
LAYERS['LTS'].append(Layer('road', 5, brand_colour(17), 'Forbidden', 5, draw_order=7))
LAYERS['LTS'].append(Layer('road', -1, [0]*3, 'Unknown', 7, draw_order=8))
LAYERS['LTS'].append(Layer('road', 10, brand_colour(10,bright=0.75), 'Ferry', 8, draw_order=9))
LAYERS['LTS'].append(Layer('road', 127, brand_colour(4, bright=0.75), 'F/B Different', 9, draw_order=0)) # must be maximal value(127)
LAYERS['vis_LTS'] = LAYERS['LTS']
LAYERS['vis_LTS_lane'] = LAYERS['LTS']

NIL_GREY = [0.7]*3

LAYERS['BLANK'] = [Layer('road', 0, NIL_GREY, 'All', 0)]

LAYERS['HEAT8'].append(Layer('road', 0, NIL_GREY, 'SMALL', 0))
for N in Qr(1,8):
    LAYERS['HEAT8'].append(Layer('road', N,
        brand_colour((16.5-N*2.5)%24,
        bright=0.7 if n==5 else 1.0),
        'LOG10 ~ '+str(N), N))

LAYERS['vis_thread'].append(Layer('road', -1, NIL_GREY, 'SMALL', 0))
for N in Qr(24):
    LAYERS['vis_thread'].append(Layer('road', N, brand_colour(N), str(N), 1))



##LAYERS['LTS'].append(Layer('road', 1, brand_colour(14), 'LTS 1 (safest)', 1))
##LAYERS['LTS'].append(Layer('road', 2, brand_colour(6,bright=0.75), 'LTS 2', 2))
##LAYERS['LTS'].append(Layer('road', 3, brand_colour(2,bright=1.1), 'LTS 3', 3))
##LAYERS['LTS'].append(Layer('road', 4, brand_colour(23,bright=1.1), 'LTS 4 (hardest)', 4))

LAYERS['vis_PRIORITY'].append(Layer('road', 0, NIL_GREY, 'Lowest Priority', 0))
LAYERS['vis_PRIORITY'].append(Layer('road', 1, brand_colour(14), 'LTS 1', 1))
LAYERS['vis_PRIORITY'].append(Layer('road', 2, brand_colour(6,bright=0.75), 'LTS 2', 2))
LAYERS['vis_PRIORITY'].append(Layer('road', 5, brand_colour(3,bright=1.1), 'Medium Priority', 5))
LAYERS['vis_PRIORITY'].append(Layer('road', 6, brand_colour(1,bright=1.1), 'High Priority', 6))
LAYERS['vis_PRIORITY'].append(Layer('road', 7, brand_colour(22, bright=1.1), 'Highest Priority', 7))

LAYERS['fwd_PRIORITY'] = LAYERS['vis_PRIORITY']
LAYERS['bwd_PRIORITY'] = LAYERS['vis_PRIORITY']

LAYERS['SLOPE'].append(Layer('road', -127, [1,0,1], 'UNKNOWN', 0)) # -127 encodes NaN
for N in Qr(7):
    LAYERS['SLOPE'].append(Layer('road', N,
        #brand_colour((16.5-N*2.5)%24),
        _magma_data[N*42],
        '>6%' if N==6 else '<%d%%'%(N+1), N+1))
for k in 'min max avg'.split():
    LAYERS['slope_'+k] = LAYERS['SLOPE']

##LAYERS['LTS'].append(Layer('point', 1, (0.0,0.5,1.0)   , 'OSM LTS 1', 1))
##LAYERS['LTS'].append(Layer('point', 2, (0.0,0.75,0.0)  , 'OSM LTS 2', 2))
##LAYERS['LTS'].append(Layer('point', 3, (1,0.625,0.0)   , 'OSM LTS 3', 3))
##LAYERS['LTS'].append(Layer('point', 4, (1.0,0.0,1/16)   , 'OSM LTS 4', 4))
##LAYERS['LTS'].append(Layer('point', 0, (0.76,0.76,0.96), 'OSM Dismount', 0))



### COLOUR/VISUAL SCHEMES ###
_ = list(rgb_to_hsv(*URBINTEL_RGB))
_[2] = _[2]/4
URBINTEL_GLASS = tuple(hsv_to_rgb(*_)) 
DEFAULT = {'glass':URBINTEL_GLASS+(0.8,), 'line':URBINTEL_RGB, 'background':(0.95,0.95,0.95,1.0),
           'text_key':URBINTEL_RGB, 'text_value':brand_colour(1,0.082)}

GLASS = {'glass':(0.9,0.9,1.0,0.93), 'line':(0.5,0.5,1.0), 'background':(0.95,0.95,0.95,1.0),
         'text_key':(0.5,0.5,1.0),'text_value':(0.2,0.2,1.0)}

TACTICAL = {'glass':(75/255,83/255,32/255,0.93), 'line':(208/255,245/255,139/255), 'background':(0.95,0.95,0.95,1.0),
            'text_key':(208/255,245/255,139/255),'text_value':brand_colour(1,0.082)}



VISUAL = {'SCHEME':0,
          'SCHEMES':'default glass tactical'.split(),
          'COLOURS':{'default':DEFAULT,'glass':GLASS,'tactical':TACTICAL},
          'key_hold_fq': 10,
          'max_png_dim': 9000,
          'print_screen_resize': 4,
          'print_screen_downsample': 3, # seems to crash for 4, but not 3.
          'print_screen_aa_filter': PIL_Image_LANCZOS}


ALGORITHM = {'MAP_HASH_GRID_m':50.0,
             'MAP_MESO_GRID_m':1000.0, # unused
             'MAP_MACRO_GRID_m':500000.0} # unused, makes perfomance worse!!!

PHYSICAL = {'building_height':{'first_floor_m':3.2, 'next_floor_m':2.5},
            'road_width_m':5,'DRIVE_ON_RIGHT':True,
            'city_dilate_m':9000,'pop_res_sample_m':50,'workplace_group_m':50,
            'cell_hex_m':200, 'min_trip_s':600}


FILES = {'UI_EXT':'uic'}

gGLOBAL = {'PHYSICAL':PHYSICAL,
           'LANGUAGE':LANGUAGE,
           'COMPUTE':COMPUTE,
           'VISUAL':VISUAL,
           'GEOGRAPHY':GEOGRAPHY,
           'ALGORITHM':ALGORITHM,
           'LAYERS':LAYERS,
           'FILES':FILES}


