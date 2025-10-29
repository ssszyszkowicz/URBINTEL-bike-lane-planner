# Prov23: Buy truth; do not sell it.



# global os os.path imports
from os import name as os_name, mkdir as os_mkdir, makedirs as os_makedirs
from os import devnull as os_devnull, listdir as os_listdir, getcwd as os_getcwd, walk as os_walk
from os import system as os_system
from os.path import isfile as os_path_isfile, isdir as os_path_isdir, exists as os_path_exists
from os.path import splitext as os_path_splitext, join as os_path_join, getsize as os_path_getsize
from sys import exit as sys_exit

# global numpy imports
from numpy.random import random as np_random
if 0: # generate numpy import code:
    NUMPY_IMPORTS = """array ndarray zeros ones arange
int8 int16 int32 int64
uint8 uint16 uint32 uint64
dtype float32 float64
nan isnan nanstd nan_to_num
isinf inf isfinite
argmin argmax argwhere minimum maximum
cos log10 logical_or sign
radians degrees pi tan log exp atan
cumsum diff clip nanmean where
prod sum size mean
concatenate repeat pad
put stack hstack vstack
resize fliplr meshgrid
flatnonzero full empty bincount
fromfile seterr unique outer around"""
    for line in NUMPY_IMPORTS.split('\n'):
        print('from numpy import '+', '.join(n+' as np_'+n for n in line.split()))
from numpy import array as np_array, ndarray as np_ndarray, zeros as np_zeros, ones as np_ones, arange as np_arange
from numpy import int8 as np_int8, int16 as np_int16, int32 as np_int32, int64 as np_int64
from numpy import uint8 as np_uint8, uint16 as np_uint16, uint32 as np_uint32, uint64 as np_uint64
from numpy import dtype as np_dtype, float32 as np_float32, float64 as np_float64
from numpy import nan as np_nan, isnan as np_isnan, nanstd as np_nanstd, nan_to_num as np_nan_to_num
from numpy import isinf as np_isinf, inf as np_inf, isfinite as np_isfinite
from numpy import argmin as np_argmin, argmax as np_argmax, argwhere as np_argwhere, minimum as np_minimum, maximum as np_maximum
from numpy import cos as np_cos, log10 as np_log10, logical_or as np_logical_or, sign as np_sign
from numpy import radians as np_radians, degrees as np_degrees, pi as np_pi, tan as np_tan, log as np_log, exp as np_exp, atan as np_atan
from numpy import cumsum as np_cumsum, diff as np_diff, clip as np_clip, nanmean as np_nanmean, where as np_where
from numpy import prod as np_prod, sum as np_sum, size as np_size, mean as np_mean
from numpy import concatenate as np_concatenate, repeat as np_repeat, pad as np_pad
from numpy import put as np_put, stack as np_stack, hstack as np_hstack, vstack as np_vstack
from numpy import resize as np_resize, fliplr as np_fliplr, meshgrid as np_meshgrid
from numpy import flatnonzero as np_flatnonzero, full as np_full, empty as np_empty, bincount as np_bincount
from numpy import fromfile as np_fromfile, seterr as np_seterr, unique as np_unique, outer as np_outer, around as np_around

# global math imports
from math import log10, atan2, cos, pi, ceil, isnan

# global shapely imports
from shapely.geometry import GeometryCollection, LinearRing, LineString, MultiPoint, MultiPolygon, Point
from shapely.geometry import Polygon as sPolygon

# other global imports:
from subprocess import call as subprocess_call, run as subprocess_run
from random import random as random_random
from re import compile as re_compile, findall as re_findall
from time import time as time_time, sleep as time_sleep
from datetime import datetime as datetime_datetime #
from json import loads as json_loads, dumps as json_dumps
NOW = datetime_datetime.now


from warnings import filterwarnings
from collections import OrderedDict as collections_OrderedDict
import sys # needed for assigning to sys.stdout in printing()
from sys import __stdout__ as sys___stdout__
from pyopencl import enqueue_copy as cl_enqueue_copy, CommandQueue as cl_CommandQueue, Buffer as cl_Buffer, mem_flags as cl_mem_flags


########class LimitedWriter:
########    def __init__(self, limit):
########            self.limit = limit
########            self.old_stdout = sys_stdout
########    def write(self, value):
########            if len(value) > self.limit: value = value[:self.limit] + "..."
########            self.old_stdout.write(value)
########sys_stdout = LimitedWriter(1000) # limit to 1000 characters
########
########


    





def sorted_unique(itera, join=None):
    s = list(set(itera))
    s.sort()
    if join is None: return s
    else: return join.join(str(x) for x in s)
    

def circular(lst, shift = 1):
    sh = shift%len(lst)
    if sh < 0: return lst[-sh:] + lst[:-sh]
    if sh > 0: return lst[sh:]  + lst[:sh]
    return lst


def merge_sets(sets):
    l = sets
    out = []
    while len(l)>0:
        first, *rest = l
        first = set(first)
        lf = -1
        while len(first)>lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2
        out.append(first)
        l = rest
    return out


def make_text_dict(text):
    out = {}
    for line in text.split('\n'):
        l = line.strip()
        if l:
            i = l.index(' ')
            out[l[:i]] = l[i:].strip()
    return out


CL_DATA_TYPES = make_text_dict("""
uint8 uchar
uint32 uint
float32 float
""")


# ! Not sure if io buffers can be read afterwards!
def run_cl_program(program, cl_context, N_parallel, inputs, outputs, ios=[], workspaces=[]):    
    clQ = cl_CommandQueue(cl_context)
    i_buffers = [cl_input(i,cl_context) if type(i) is np_ndarray else i for i in inputs]
    o_buffers = [cl_output(o,cl_context) for o in outputs]
    io_buffers = [cl_io(io,cl_context) for io in ios]
    ws = [cl_io(ws,cl_context) for ws in workspaces]
    params = i_buffers+o_buffers+io_buffers+ws
    if type(N_parallel) not in (tuple,list): N_parallel = (N_parallel,)
    program(clQ, N_parallel, None,*params)
    for n,o in enumerate(outputs): cl_enqueue_copy(clQ, o, o_buffers[n])
    for n,io in enumerate(ios): cl_enqueue_copy(clQ, io, io_buffers[n])
    return o_buffers+io_buffers

def cl_input(np_a, CL_CONTEXT):
    mf = cl_mem_flags
    return cl_Buffer(CL_CONTEXT,
                     mf.READ_ONLY | mf.COPY_HOST_PTR,
                     hostbuf=np_a) 
def cl_output(np_a, CL_CONTEXT):
    mf = cl_mem_flags
    return cl_Buffer(CL_CONTEXT,
                     mf.WRITE_ONLY,
                     np_a.nbytes) 
def cl_io(np_a, CL_CONTEXT):
    mf = cl_mem_flags
    return cl_Buffer(CL_CONTEXT,
                     mf.READ_WRITE | mf.COPY_HOST_PTR,
                     hostbuf=np_a)




FLOATS_REGEX = re_compile(r"[-+]?\d*\.\d+|\d+")
INTS_REGEX = re_compile(r"[-+]?\d+")

def get_all_ints(string):
    return [int(s) for s in re_findall(INTS_REGEX,string)]

def get_all_floats(string):
    return [float(s) for s in re_findall(FLOATS_REGEX,string)]


def warnings(true):
    if true: filterwarnings("always")
    else: filterwarnings("ignore")

def printing(true):
    if true: sys.stdout = sys___stdout__
    else: sys.stdout = open(os_devnull, 'w')
    




def print_imports():
    for file in os_listdir():
        if file.endswith(".py"):
            with open(file, mode='r') as F:
                print('<',file,'>')
                for L in F:
                    if 'import' in L:
                        print(L.strip('\n'))
                

def get_depth(obj,rec=0):
    if isinstance(obj, LoL): return 2+rec
    if isinstance(obj, np_ndarray): return len(obj.shape)+rec
    try:
        len(obj)
        try: return get_depth(obj[0],rec+1)
        except: return rec+1 
    except: return rec

def make_dict(string, space_char=' '):
    out = {}
    for line in string.split('\n'):
        if line.strip():
            parts = line.split(space_char)
            if space_char: parts = [p.replace(space_char,' ') for p in parts]
            out[parts[0]] = ' '.join(parts[1:])
    return out

def str_to_dict(string, space_char=None):
    out = {}
    for line in string.split('\n'):
        if line.strip():
            parts = line.split(' ')
            if space_char: parts = [p.replace(space_char,' ') for p in parts]
            out[parts[0]] = parts[1:]
    return out

def str_to_inv_dict(string, space_char=None):
    out = {}
    for line in string.split('\n'):
        parts = line.split(' ')
        if space_char: parts = [p.replace(space_char,' ') for p in parts]
        for p in parts[1:]: out[p] = parts[0]
    return out


def install_all():
    PACKAGES = "js2py networkx"
    for package in PACKAGES.split():
        try: pass # import..."package"
        except: pass #subprocess.call("...")
    



def mean_std(vec):
    clean = vec #[v for v in vec if v is not None]
    return (np_nanmean(clean),np_nanstd(clean))


ILLEGAL_FILE_CHARS = "/\\:*?<>|"


TYPE_NAMES = {int:'I',float:'F',list:'L',
              tuple:'T',dict:'D',str:'$',set:'S',
              collections_OrderedDict:'oD'}


CHAR_RANGES = "AZA aza 090 ".split()
def char_type(c):
    for r in CHAR_RANGES:
        if c>=r[0] and c<=r[1]: return r[2]
    return c
    

def type_list_of_strings(LOS):
    types = set()
    for s in LOS:
        try:
            v = int(s)
            if v == float(s): t = int
            else: v == float
        except:
            try:
                v = float(s)
                t = float
            except:
                v = s
                t = str
        types.add(t)
    for t in [int, float, str]:
        if t in types: TYPE = t
    if TYPE is str: VALS = LOS
    else: VALS = [TYPE(s) for s in LOS]
    out = {}
    out['type'] = TYPE
    out['type_name'] = TYPE_NAMES[TYPE] 
    out['vals'] = VALS
    if TYPE in [int, float]:
        out['min'] = min(VALS)
        out['max'] = max(VALS)
    if TYPE is str:
        out['max_len'] = max(len(s) for s in VALS)
        char_set = set(c for s in S for c in s)
        char_type_set = list(set(char_type(c) for c in s))
        char_type_set.sort()
        out['char_types'] = ''.join(char_type_set) 
    return out



def wild_file(wild_name, root=None):
    if root==None: root = os_getcwd()
    pass

def read_file(wild_name, mode='r'):
    pass



def typename(OBJ):
    #if type(OBJ) in TYPE_NAMES:
    return TYPE_NAMES[type(OBJ)]
    
def key_to_str(key):
    if type(key) is str: return '"'+key+'"'
    else: return str(key)
def deeptype(OBJ, keys={}, strings={}):
    if OBJ is None: return '0'
    if OBJ is True: return 'B'
    if OBJ is False: return 'B'
    T = typename(OBJ)
    out = T
    if T in ['D','oD']:
        DT = [key_to_str(i)+':'+deeptype(OBJ[i],keys,strings) for i in OBJ]
        ST = set([dt for dt in DT])
        SL = list(ST)
        SL.sort()
        out += '(%s)'%','.join(SL)   
    if T in ['L','T','S']:
        DT = [deeptype(i,keys,strings) for i in OBJ]
        ST = set([dt for dt in DT])
        SL = list(ST)
        SL.sort()
        out += '(%s)'%','.join(SL)
    return out

def deepkeys(OBJ, keys = {}):
    T = typename(OBJ)
    if T in ['D','oD']:
        for k in OBJ.keys():
            if k in keys: keys[k] += 1
            else: keys[k] = 1
        for i in OBJ:
            for k in deepkeys(OBJ[i]):
                if k in keys: keys[k] += 1
                else: keys[k] = 1
    if T in ['L','T','S']:
        for i in OBJ:
            for k in deepkeys(i):
                if k in keys: keys[k] += 1
                else: keys[k] = 1
    return keys

def explode(OBJ, strlen=5000):
    S = deeptype(OBJ)
    if len(S) <= strlen:
        print(S)
        return S
    else:
        print('utils.explode() output is too long(%d characters)'%len(S))



def findattr(obj, attr, depth = 4, stack=''):
    for a in dir(obj):
        if a[0] != '_':
            if depth > 0:
                try: findattr(getattr(obj,a),attr,depth-1,stack+'.'+a)
                except: pass
            if a == attr: print(stack+'.'+a)

def histo(lst, silent=False):
    if not silent: print(str(type(lst))+' contains %d items according to the following distribution:'%len(lst))
    out = {}
    for i in lst:
        if i in out: out[i]+=1
        else: out[i]=1
    K = list(out.keys())
    K.sort()
    N = len(lst)
    if not silent:
        for k in K: print(k,' ---> ',out[k],'(%.2f%%)'%(out[k]/N*100.0))
        print('')
    return out

def dtypename(t):
    if type(t) is str: return t
    if hasattr(t, 'name'): return t.name
    else: return t.__name__

def ctypebytes(obj):
    if type(obj) is str: s=obj
    else:
        try: s = obj.dtype.name
        except: return None
    i = intlist(s)
    if len(i) != 1: return None
    else: return {8:1,16:2,32:4,64:8}[i[0]]
    
def deepctype(obj):
    O = list(deepnum(obj))
    if O[2] is float: return np_float64
    if O[0]<0 or O[1]<0:#signed
        for n in [0,1]:
            if O[n]<0: O[n]=-O[n]-1
        M = max(O[0:2])
        if O[1]<2**7: return np_int8
        if O[1]<2**15: return np_int16
        if O[1]<2**31: return np_int32
        return np_int64        
    else: #unsigned
       if O[1]<2**8: return np_uint8
       if O[1]<2**16: return np_uint16
       if O[1]<2**32: return np_uint32
       return np_uint64
       
##def mergectypes(a,b):
##    def to_str(o):
##        if type(o) is str: return o
##        else: return o.__name__
##    A = to_str(a)
##    B = to_str(b)
    
    
def deepnum(obj): # tuple is min, max, type
    if type(obj) in [int, float]:
        return (obj, obj,type(obj))
    if type(obj) is list:
        if not obj: return None
        L = [a for a in (deepnum(o) for o in obj) if a is not None]        
        FLOAT = False
        if float in (o[2] for o in L): FLOAT = True
        return (min(o[0] for o in L),max(o[1] for o in L), [int,float][FLOAT])




CL_STACKS_CODES = [('STACKS_'+dtype, """
__kernel void LoL2Stacks(
__global const """+CL_DATA_TYPES[dtype]+""" *i_data,
__global const uint *offsets,
__global """+CL_DATA_TYPES[dtype]+""" *o_data,
__global """+CL_DATA_TYPES[dtype]+""" *overflow,
__global uint *overflow_ix,
__global uint *SIZE)
{
    uint s = get_global_id(0);
    uint depth = SIZE[2];
    uint overflow_depth = SIZE[5];
    uint o = offsets[s];
    __global """+CL_DATA_TYPES[dtype]+""" *list = i_data + o;
    uint len = offsets[s+1] - o;
    __global """+CL_DATA_TYPES[dtype]+""" *w_data = o_data + (s*depth);
    uint ov; 
    for(uint n=0; n<len; n++) {
        if(n<depth) {w_data[n] = list[n];}
        if(n==depth) {ov = atomic_inc(SIZE+3); overflow_ix[s] = ov;}
        if(n>=depth) {overflow[(ov*overflow_depth)+n-depth] = list[n];}
    }
}
""") for dtype in 'uint8 uint32 float32'.split()]

# size array [6]:
# [0]: N: current number of stacks
# [1]: Nmax: max number of stacks
# [2]: depth: fixed stack depth
# [3]: O: number of overflowing stacks
# [4]: Omax: max number of overflowing stacks
# [5]: overflow_depth
# objects in [] can be connected to other Stacks (sync)
# data: size1 x size2
# overflow: size4 x size5
class Stacks:
    def to_LoL(self):
        pass
    def __init__(self, _LoL, depth, COMPUTE = False, expand_ratio = 2.0, extra_overflow = 0):
        self.dtype = _LoL.dtype
        lens = _LoL.lens()
        if lens.dtype.name != 'uint32': lens = lens.astype('uint32')
        offsets = _LoL.offsets
        if offsets.dtype.name != 'uint32': offsets = offsets.astype('uint32')
        self.size = [np_array([len(lens),
                               int(len(lens)*expand_ratio)+1,
                               depth,
                               0,
                               int(sum(lens>depth)*expand_ratio)+1,
                               lens.max()-depth+extra_overflow],'uint32')]
        self.lens = np_pad(lens, (0, self.size[0][1]-len(lens)), 'constant', constant_values=(0))
        self.data = np_empty((self.size[0][1],depth),self.dtype)
        self.overflow = np_empty((self.size[0][4],self.size[0][5]),self.dtype)
        self.overflow_ix = [np_empty(self.size[0][1],'uint32')]        
        if COMPUTE:
            inputs = (_LoL.data, offsets)
            outputs = (self.data, self.overflow, self.overflow_ix[0])
            io = (self.size[0],)
            self.COMPUTE = COMPUTE 
            PROGRAM = self.COMPUTE['BUILT']['STACKS_'+dtypename(self.dtype)]
            CL_CONTEXT = self.COMPUTE['CL_CONTEXT']
            RUN_PROGRAM(PROGRAM.LoL2Stacks, CL_CONTEXT, len(lens), inputs, outputs, io)
    def __call__(self,stacks): # sync to other Stacks
        stacks.size = self.size
        stacks.lens = self.lens
        stacks.overflow_ix = self.overflow_ix
    def __len__(self):
        return self.size[0][0]
    def add_stacks(self,num_new_stacks):
        self.size[0][1] += num_new_stacks
        self.data.resize((self.size[0][1],self.size[0][2]))
        self.lens[0].resize((self.size[0][1],))
        self.overflow_ix[0].resize((self.size[0][1],))
    def increase_overflow(self, new_overflows=0, deeper_overflow=0):
        self.size[0][4] += new_overflows
        self.size[0][5] += deeper_overflow
        self.overflow.resize((self.size[0][4],self.size[0][5])) # TO_DO: repair this
    def __getitem__(self,si): # slow - for reference/testing only
        l = self.lens[0][si]
        M = self.size[0][2]
        out = [self.data[si,n] for n in range(min(l,M))]
        if l>M:
            oix = self.overflow_ix[0][si]
            out += [self.overflow[oix][n] for n in range(l-M)]
        return out
    def __setitem__(self,si,val):
        if si>=self.size[0][1]:
            print('Stacks segfault')
            return False
        l = self.lens[0][si]
        M = self.size[0][2]
        if l>=self.size[0][5]+M:
            print('Stacks overflow_overflow')
            return False
        if l<M: self.data[si,l] = val
        else:
            if l==M:
                o = self.size[0][3]
                if o < self.size[0][4]: self.size[0][3] += 1
                else:
                    print('out_of_overflows')
                    return False
                self.overflow_ix[0][si] = o
            else: o = self.overflow_ix[0][si]
            self.overflow[o,l-M] = val
        self.lens[0][si] += 1
        self.size[0][0] = max(self.size[0][0],si+1)
    def new_stack(self):
        if self.size[0][1]<self.size[0][2]: self.size[0][1]+=1
        else:
            print('Stacks: out_of_stacks')
            return False
    def expose(self):
        return (self.size[0], self.lens[0], self.overflow_ix[0])
# Numerical version needs to implement errors: out_of_stacks, out_of_overflows, overflow_overflow :)

if 0:
    from gGlobals import *
    build_all_cl_code(gGLOBAL['COMPUTE'])
    x=LoL([[3,4,5],[2,1],[8,7,9,10,11],[12,13,14,15,16,17]],'uint32')
    s=Stacks(gGLOBAL['COMPUTE'], x, 4)


        
class Block:
    def __init__(self, lol, datatype):
        N = len(lol)
        self.N = N
        lens = tuple(len(l) for l in lol)
        M = max(lens)
        self.M = M
        self.lens = np_array(lens,dtype=deepctype(M))
        self.data = np_array([l+[0]*(M-len(l)) for l in lol],datatype) # empty values=0 so that there is no conversion overflow
    def __getitem__(self,ix): return self.data[ix,0:self.lens[ix]]
    def __len__(self): return self.N
    def lol(self): return tuple(tuple(self[i]) for i in range(len(self))) 





class List:
    def __init__(self, numlist = None, dtype = None):
        if numlist is None:
            self.data = []
            if dtype == None: self.dtype = 'float64'
            else: self.dtype = dtype
        else:
            if dtype == None:
                if isinstance(numlist, np_ndarray): self.dtype = numlist.dtype
                else: self.dtype = 'float64'
            else: self.dtype = dtype
            self.data = [self._np_array(numlist)] #sic
            self.dtype = self.data[0].dtype
    def _np_array(self, numlist):
        if isinstance(numlist,np_ndarray):
            if numlist.dtype == self.dtype:
                return numlist
        return np_array(numlist,self.dtype)
    def __len__(self): return sum(a.shape[0] for a in self.data)
    def lens(self): return tuple(a.shape[0] for a in self.data)
    def array(self):
        if len(self.data) == 0: return self._np_array([]) #sic
        if len(self.data) == 1: return self.data[0]
        return np_concatenate(self.data,axis=None)
    def merge(self): self.data = [self.array()]
    def extend(self, numlist): self.data.append(self._np_array(numlist)) #sic
    def insert(self, numlist, index): self.data.insert(index,self._np_array(numlist)) #sic
    def pop(self, index=-1): return self.data.pop(index)
    def copy(self):
        new = List()
        new.dtype = self.dtype
        new.data = self.data.copy()
    def tolist(self): return self.array().tolist()
    def min(self):
        mins = [a.min() for a in self.data if a.shape[0]]
        if mins: return min(mins)
        else: return None
    def max(self):
        maxs = [a.max() for a in self.data if a.shape[0]]
        if maxs: return max(maxs)
        else: return None

class ListofLists:
    def __init__(self, lol = None, dtype = None):
        if lol == None:
            self.data = []
            if dtype == None: self.dtype = 'float64'
            else: self.dtype = dtype
        else:
            self.data = [LoL(lol,dtype)]
            self.dtype = self.data[0].dtype          
    def __len__(self): return sum(len(a) for a in self.data)
    def list_of_arrays(self): return sum((lol.list_of_arrays() for lol in self.data),[])
    def LoL(self):
        if len(self.data) == 1: return self.data[0]
        new = LoL([],dtype=self.dtype)
        if len(self.data) == 0: return new
        Ns = [d.N for d in self.data]
        cumN = np_cumsum(Ns)
        new.N = cumN[-1]
        datalens = [d.datalen for d in self.data]
        new.datalen = sum(datalens)
        new.lens = np_concatenate([d.lens for d in self.data])
        new.dtype = self.dtype
        new.data = np_concatenate([d.data for d in self.data])
        offsets = np_concatenate([d.offsets[:-1] for d in self.data]+[self.data[-1].offsets[-1]])
        offset = np_cumsum(datalens)
        for n in range(len(datalens)-1):
            offsets[cumN[n]:cumN[n+1]] += offset[n]
        new.offsets = offsets 
        return new
    def merge(self): self.data = [self.LoL()]
    def extend(self, lol): self.data.append(LoL(lol, self.dtype))
    def insert(self, lol, index): self.data.insert(index,LoL(lol,self.dtype))
    def pop(self, index=-1): return self.data.pop(index)
    def copy(self):
        new = ListofLists()
        new.dtype = self.dtype
        new.data = self.data.copy()
    def tolol(self): return self.LoL().tolol()    
    def min(self):
        mins = [lol.data.min() for lol in self.data if lol.datalen]
        if mins: return min(mins)
        else: return None
    def max(self):
        maxs = [lol.data.max() for lol in self.data if lol.datalen]
        if maxs: return max(maxs)
        else: return None            

def lol_to_tuple(lol):
    return tuple(e for l in lol for e in l)


def dtype_short(obj):
    t = str(obj.dtype)
    if t=='bool':return t
    return t[0] + ('8' if t[-1]=='8' else t[-2:])

class LoL:
    def filter(self, cond): # TO_DO: optimize ?
        if len(self) != len(cond): raise Exception('LoL.filter len mismatch')
        return LoL(lol = [v for n,v in enumerate(self.tolol()) if cond[n]], dtype = self.dtype)
    def __Qstr__(self):
        s = '*LoL'+dtype_short(self.data)
        s += '[%d]'%len(self.data)
        return s + '(%d)'%(len(self.offsets)-1)
    def load(self, data, offsets):
        self.data = data
        self.offsets = offsets
        self.N = len(offsets)-1
        self._lens = [False]
        self.datalen = self.data.size
        self.dtype = data.dtype
        return self
    def __init__(self,lol=[],dtype=None):
        if isinstance(lol, LoL):
            self.N = lol.N
            self._lens = lol._lens
            self.offsets = lol.offsets
            self.datalen = lol.datalen
            if dtype is None:
                self.data = lol.data
                self.dtype = lol.dtype
            else:
                if dtypename(dtype)==dtypename(lol.dtype):
                    self.data = lol.data
                else:
                    self.data = np_array(lol.data,dtype)
                self.dtype = dtype
        else:
            cumsum_type = 'uint32'
            self.N = len(lol)
            lens = np_array(tuple(len(l) for l in lol),dtype=cumsum_type)
            flat = lol_to_tuple(lol)
            self.data = np_array(flat,dtype=dtype)
            self.datalen = len(flat)
            self.dtype = self.data.dtype
            cumsum = np_cumsum(lens,dtype=cumsum_type)
            self.offsets = np_concatenate((np_zeros(1,dtype=cumsum_type),cumsum))
            self._lens = [False]
    def lens(self):
        if self._lens[0] is False: self._lens[0] = np_diff(self.offsets) # "is", not "==" !
        return self._lens[0]
    def __getitem__(self,ix): return self.data[self.offsets[ix]:self.offsets[ix+1]]
    def list_of_arrays(self): return [self[n] for n in range(len(self))]
    def __len__(self): return self.N
    def tolol(self): return tuple(tuple(l.tolist()) for l in self)
    def __call__(self, lol):
        self.offsets = lol.offsets
        self._lens = lol._lens
        self.N = lol.N
        self.datalen = lol.datalen
def LoLu8(lol=[]): return LoL(lol, 'uint8')
def LoLu32(lol=[]): return LoL(lol, 'uint32')
def LoLf32(lol=[]): return LoL(lol, 'float32')


if 0:#test LoL
    input('Testing LOL - press enter...')
    Z = tuple(tuple(range(int(random_random()*10))) for n in range(1*2**20))
    input(len(Z)); L  = LoL(Z,'uint8'); print(L)
    # with ~45M elements (in 8 seconds) Z-to-L achieves:
    # - over x14 compression ratio over lists,
    # - over x9 compression ratio over tuples
    # ;ALSO with 4.5M elements.
    # - about x10 compression ratio over lists,
    # - over x9 compression ratio over tuples


class HugeVec:
    def __init__(self, ctype, block_size = 10**6):
        self.ctype = ctype
        self.len = 0
        self.block_size = block_size
        self.block = 0
        self.index = 0
        self.mem = [np_empty(block_size, ctype)]
        self.len = 0 # keep this for speed
    def __call__(self, v):
        i = self.index
        b = self.block
        self.mem[b][i] = v
        i += 1
        self.len += 1
        if i == self.block_size:
            i = 0; b += 1
            self.mem.append(np_empty(self.block_size, self.ctype))
        self.index = i
        self.block = b
    def __getitem__(self, ix):
        if ix<self.len:
            bs = self.block_size
            return self.mem[ix//bs][ix%bs]
        else: raise StopIteration()
    def __len__(self):
        return self.len
    def annihilate_to_set(self):
        S = set(self.mem.pop()[:self.index])
        while(self.mem): S.update(set(self.mem.pop()))
        self.__init__(self.ctype,self.block_size) # reset object to empty
        return S


    
class HugeDict:
    def __init__(self, ctype, hash_size = 10**7, vec_size = 10): 
        self.hash_size = hash_size
        self.vec_size = vec_size
        self.len = 0
        self.counts = np_zeros(hash_size,deepctype(vec_size))
        self.keys = np_empty((hash_size,vec_size),'uint64')
        self.values = np_empty((hash_size,vec_size),ctype)
        self.extra = {}
    def __setitem__(self, key, val):
        self.len += 1
        i = key%self.hash_size
        c = self.counts[i]
        if c < self.vec_size:
            self.keys[i][c] = key
            self.values[i][c] = val
            self.counts[i] += 1
        else:
            self.extra[key] = val
    def __getitem__(self, key):
        i = key%self.hash_size
        k = self.keys[i]
        h = np_where(k == key)[0]
        if not len(h): return self.extra.get(key)
        else: return self.values[i][h[0]]
    def __len__(self): return self.len

class IndexDict:
    def __init__(self, keys, keytype=None):
        if keytype==None:
            #try:
            keytype = keys.dtype.name
            #except: keytype = deepctype(keys)
        if 'int' not in str(keytype):
            print('ERROR: class IndexDict can only hash ints!')
            return None
        self.N = len(keys)
        COMPACT_FACTOR = 1.0
        self.HashLen = int(self.N / COMPACT_FACTOR) + 1
        hashkeys = tuple([] for _ in range(self.HashLen))
        indexes = tuple([] for _ in range(self.HashLen))
        for ix,key in enumerate(keys):
            hsh = int(key%self.HashLen) # note: % can return floats: int.0
            indexes[hsh].append(ix)
            hashkeys[hsh].append(int(key))
        self.indexes = LoL(indexes,deepctype(self.N))
        self.hashkeys = LoL(hashkeys,keytype)
        self.hashkeys(self.indexes)
    def __getitem__(self,key):
        hsh = int(key%self.HashLen) # note: % may return floats: int.0
        hks = self.hashkeys[hsh]
        if len(hks) == 0: return None
        if len(hks) == 1:
            hk = hks[0]
            if hk == key: return self.indexes[hsh].item(0) # HIT!
            else: return None
        # If multiple hash hits, i.e., len(hks)>1, continue:
        wh = np_argwhere(hks==key) 
        if len(wh) == 0: return None
        return self.indexes[hsh].item(wh.item(-1)) # HIT!
        # Multiple hits allowed: get last insertion. 
    def __len__(self): return self.N
    def mem(self):
        o = self.hashkeys.offsets
        return self.indexes.mem() + self.hashkeys.mem() - len(o)*ctypebytes(o)
class __SimpleIndexDict: # Backup to basic python dict with .get()
    def __init__(self, keys, keytype=None):
        self.dict = {int(n):int(i) for i,n in enumerate(keys)}
    def __getitem__(self,key):
        return self.dict.get(key)
    def __len__(self): return len(self.dict)
    def mem(self): return len(self.dict)*120 #approx.



## This may automate freezing lists and LoLs (etc.) into numpy objects, but autodetection of type migth just be
## too slow for this to be worthwhile in dev code.
##def freeze(obj, form=None, ctype=None):
##    if form==None: form = deeptype(N)
##    if form in 'IF': return obj
##    #...if 
    
    
def intlist(string):
    return [int(s) for s in re_findall(r'\d+', string)]

def short_str(obj, max_len=100):
    s = str(obj)
    if len(s) > max_len:
        try: return s[:max_len]+' ... (%d items.)'%len(obj)
        except: return s[:max_len]+' ... (%d chars.)'%len(s)
    else: return s


def disp(Obj,C=20,idx=None):
    try:
        try: print(str(type(Obj))+" has %d items."%len(Obj))
        except: pass
        try:
            if len(Obj)<=10**6: print('min =',min(Obj),'max =',max(Obj))
        except: pass
        if not isinstance(Obj,(set,list,tuple,dict,str)):
            for i in dir(Obj):
                if not i.startswith('__'):
                    a = getattr(Obj,i)
                    if not callable(a): print(i,':',a)
        if hasattr(Obj,'head'): print(short_str(Obj.head))
        c=0
        for I in Obj:
            if c%C==0: print('Showing entries %d'%c+" to %d:"%(min(len(Obj),c+C)-1))
            if type(Obj) is dict:
                if idx is None: print(short_str(I),' --> ',short_str(Obj[I]))
                elif type(idx) is int or type(idx) is str: print(short_str(I),' --> ',short_str(Obj[I]['idx']))
                elif type(idx) is list: print(short_str(I),' --> ',short_str([Obj[I][k] for k in idx]))            
            else:
                if idx is None: print(short_str(I))
                elif type(idx) is int or type(idx) is str: print(short_str(I[idx]))
                elif type(idx) is list: print(short_str([I[k] for k in idx]))
            c+=1
            if c%C==0:
                if(len(input())): break
    except:
        print(short_str(Obj,C*80))
        try: print('Variables: ',end='');disp(vars(Obj),C)
        except: pass#print('Could not find any vars().')


        
def get_all_inclusions(root, language = 'py'):
    pattern = re_compile("[a-zA-Z./]+")
    out = set()
    include = {'py':'import','c':'#include','cpp':'#include'}[language]
    for file in get_all_files(root, language):
        with open(file,'r') as FILE:
            for line in FILE:
                if line.strip().startswith(include):
                    out.update(re_findall(pattern,line))
    return out
    

def get_all_files(root, extension = 'py'):
    FILES = []
    for path, subdirs, files in os_walk(root):
        for name in files:
            f, ext = os_path_splitext(name)
            if extension == None or ext[1:] == extension:
                file = os_path_join(path, name)
                FILES.append(file)
    return FILES

#disp(get_all_inclusions("C:/Users/S/Desktop/General Urban/depthmapX","cpp"),30)
#disp(get_all_inclusions("C:/Users/S/Desktop/urb metr/osmnx-master","py"),40)



###import numpy 
##class Coordinates():
##    def __init__(self,coords,centre=None):
##        pass
##    

def heat_colour(v):
	if (v<0): return (0.0,0.0,0.0)
	if (v>1): return (1.0,1.0,1.0)
	if (v < 0.25): return (0.0, 4.0*v,1.0)
	if (v < 0.5): return (0.0, 1.0, 2.0-4.0*v)
	if (v < 0.75): return (4.0*v-2.0, 1.0, 0.0)
	return (1.0, 4.0-4.0*v,0.0);




class Selection:
    def __len__(self): return len(self.select)
    def __getitem__(self, item):
        return [v for n,v in enumerate(self.items[item]) if self.select[n]]
    def __setitem__(self, item, obj): self.items[item] = obj
    def __init__(self, N):
        self.items = {}
        self.select = np_ones(N)
    def __call__(self, sel): self.select *= np_array(sel,'uint8')


class Values:
    def sum(self): return sum(self.s)
    def mean(self): return sum(self.s)/sum(self.w)
    def means(self): return self.s/self.w
    def sums(self): return self.s
    def __init__(self, N):
        self.s = np_zeros(N, 'float64')
        self.w = np_zeros(N, 'float64')
    def __call__(self,v,w,n=None):
        if n is None:
            for i in range(v): self(v[i],w[i],i) # TO_DO:vectorize
        else:
            if np_isfinite(v):
                self.s[n] += v*w
                self.w[n] += w
            
            
class Value:
    def sum(self): return self.s
    def mean(self): return self.s/self.w
    def __init__(self):
        self.s = 0
        self.w = 0
    def __call__(self,v,w):
        d = get_depth(v)
        if d == 1:
            self.s += sum(v[n]*w[n] for n in range(len(w)) if np_isfinite(v[n]))
            self.w += sum(w[n] for n in range((w)) if np_isfinite(v[n]))
        elif d==0:
            if np_isfinite(v):
                self.s += v*w
                self.w += w

def tab(columns):
    for n in range(min(len(c) for c in columns)):
        for c in columns:
            s = str(c[n])
            l = max(0, 30-len(s))
            print(s+' '*l,end='')
        print('\n',end='')
        if (n+1)%20==0:
            if input(): return None
    

def xor(x, y):
    return bool((x and not y) or (not x and y))


class Bible:
    def __init__(self, book = 'PSA'):
        self.book = book.upper()
        path = '/Bible DR/'
        self.text = {}
        for f in get_all_files(path,'sfm'):
            tag = f[(len(path)+3):(len(path)+6)]
            with open(f,'r',encoding='utf8') as ff: self.text[tag] = ff.read().split('\\')
    def __lt__(self, n): self.__gt__(n)
    def __gt__(self, n):
        if type(n) is str:
            if n in self.text: self.book = n.upper()
            else: print('Unknown book:',n)
        if type(n) is int:
            p = False
            ks = "c %d"%n
            ke = "c %d"%(n+1)
            for s in self.text[self.book]:
                s = s.strip('\n')
                if s == ke: p = False
                if p:
                    if s.startswith('v'): print(s[2:])
                    elif s.startswith('cl'): print(s[3:],end='')
                    elif s.startswith('s1'): print(' ~~ '+s[3:])
                    elif s.startswith('cd'): print('['+s[3:]+']')
                if s == ks: p = True
                
B = Bible('PSA')
