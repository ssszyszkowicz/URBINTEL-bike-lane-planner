import pickle
from numba import njit
from qcommands import *
from pyopencl import *

# not debugged!
CL_ALL_TO_ALL_CODE = ('ALL_TO_ALL', """
__kernel void A(
const uint NO, // u32
const uint ND, // u32
__global const float *MX_times, // f32
const uint MX_N, // u32
const float min_s, // f32
__global const float *OA_s_DATA, // f32
__global const uchar *OA_s_LENS, // u8
const uint OA_s_M, // u32
__global const float *DA_s_DATA, // f32
__global const uchar *DA_s_LENS, // u8
const uint DA_s_M, // u32
__global const uint *OATN_DATA, // u32
__global const uint *DATN_DATA, // u32
__global const uint *OTNix, // u32
__global const uint *DTNix, // u32
__global float *times, // f32
__global char *Oroutes, // i8
__global char *Droutes, // i8
__global float *O_s, // f32
__global uint *OA, // u32
__global float *D_s, // f32
__global uint *DA // u32
){
uint nD = get_global_id(0);
uchar dl = DA_s_LENS[nD];    
for(uint nO = 0; nO<NO; nO++) {
    uchar ol = OA_s_LENS[nO];
    float mini = INFINITY;
    uchar d_min = -1, o_min = -1;
    for(uchar o=0; o<ol; o++) {
        ulong ix0 = nO*OA_s_M+o; // u64
        O_s[o] = OA_s_DATA[ix0];
        OA[o] = OATN_DATA[ix0];
        for(uchar d=0; d<dl; d++) {
            ulong ix1 = nD*DA_s_M+d; // u64
            D_s[d] = DA_s_DATA[ix1];
            DA[d] = DATN_DATA[ix1];
            // Note Dijkstra matrix must be read: [D,O]
            float trip = MX_times[DTNix[DA[d]]*MX_N+OTNix[OA[o]]]; /// + O_s[o] + D_s[d];
            if(trip < mini) {
                mini = trip;
                d_min = d;
                o_min = o;
    }}}
    if(mini < min_s) mini = INFINITY; // this line eliminates short paths. 
    ulong ix2 = nO*ND+nD; // u64
    times[ix2] = mini;
    Oroutes[ix2] = o_min; // these are indices.
    Droutes[ix2] = d_min; // "            
}}""")


### OPENCL CONFIG ###
CL_PLATFORM = get_platforms()[0]
CL_DEVICE = CL_PLATFORM.get_devices()[0] # should be 2 - 1 GPU, 1 CPU, but I see only GPU now.
CL_CONTEXT = Context([CL_DEVICE])

COMPUTE = {'CL_CONTEXT':CL_CONTEXT,
           'CODE': [CL_ALL_TO_ALL_CODE],
           'BUILT':{}}

for NAME,CODE in COMPUTE['CODE']:
    B = Program(COMPUTE['CL_CONTEXT'],CODE).build()
    COMPUTE['BUILT'][NAME] = B






def _route_all_to_all(use_CL, MX_times, min_s, OA_s, DA_s,
                      OATN, DATN, OTNix, DTNix, COMPUTE=None):
    P = ((len(OATN), len(DATN)),None)
    times = Qf32(*P)
    Oroutes = Qi8(*P)
    Droutes = Qi8(*P)
    PARAMS  =  [times,Oroutes,Droutes, # outputs
                np_uint32(len(OATN)), np_uint32(len(DATN)),
                MX_times, np_uint32(MX_times.shape[1]), np_float32(min_s),
                OA_s.data, OA_s.lens, np_uint32(OA_s.M),
                DA_s.data, DA_s.lens, np_uint32(DA_s.M),
                OATN.data, DATN.data,
                OTNix, DTNix]
    if use_CL: return _CL_all_to_all(*PARAMS, COMPUTE)
    else: return [_JIT_all_to_all(*PARAMS), Oroutes, Droutes]



### ! uint8 is valid if there are at most 127 anchors per Res/Wrk -  should be ok:) (currently max=2)
@njit("float32[:,:](float32[:,:],"+"int8[:,:],"*2+\
      "uint32,"*2+"float32[:,:],uint32,float32,"+\
      "float32[:,:],uint8[:],uint32,"*2+\
      "uint32[:,:],"*2+\
      "uint32[:],uint32[:])")
# TO_DO: add local variable types.
def _JIT_all_to_all(times, Oroutes, Droutes, # outputs
                    NO, ND, MX_times, MX_N, min_s, #MX_N is ignored in JIT version.
                    OA_s_DATA, OA_s_LENS, OA_s_M,
                    DA_s_DATA, DA_s_LENS, DA_s_M,
                    OATN_DATA, DATN_DATA,
                    OTNix, DTNix):
##    for nD in range(len(DD)):
##        D = DD[nD]
##        dl = DA_s_LENS[D]
##        for nO in range(len(OO)):
##            O = OO[nO]
##            ol = OA_s_LENS[O]
##            times[nO,nD] = 1.0
##    for nD in range(len(DD)):
##        for nO in range(len(OO)):
##            times[nO,nD] = 1.0
##    return times
    O_s = np_empty(OA_s_M,np_float32) # must be literal for jit...
    OA = np_empty(OA_s_M,np_uint32)
    D_s = np_empty(DA_s_M,np_float32)
    DA = np_empty(DA_s_M,np_uint32) #... .
    for nD in range(ND):
        dl = DA_s_LENS[nD]
        for nO in range(NO):
            ol = OA_s_LENS[nO]
            mini = np_float32(np_inf)
            d_min = np_int32(-1)
            o_min = np_int32(-1)
            for o in range(ol):
                O_s[o] = OA_s_DATA[nO,o]
                OA[o] = OATN_DATA[nO,o]
                for d in range(dl):
                    D_s[d] = DA_s_DATA[nD,d]
                    DA[d] = DATN_DATA[nD,d]
                    # Note Dijkstra matrix must be read: [D,O]
                    trip = MX_times[DTNix[DA[d]],OTNix[OA[o]]] ###+O_s[o] + D_s[d] 
                    if trip < mini:
                        mini = trip
                        d_min = d
                        o_min = o
            if mini < min_s: mini = np_float32(np_inf) # this line eliminates short paths. 
            times[nO,nD] = mini
            Oroutes[nO,nD] = o_min # these are indices.
            Droutes[nO,nD] = d_min # "
    return times




def _CL_all_to_all(times, Oroutes, Droutes, # outputs
                    NO, ND, MX_times, MX_N, min_s,
                    OA_s_DATA, OA_s_LENS, OA_s_M,
                    DA_s_DATA, DA_s_LENS, DA_s_M,
                    OATN_DATA, DATN_DATA,
                    OTNix, DTNix, COMPUTE):
    #Qer("This code is buggy. Please use JIT version for now.")
    O_s = Qf32(OA_s_M,None)
    OA = Qu32(OA_s_M,None)
    D_s = Qf32(DA_s_M,None)
    DA = Qu32(DA_s_M,None)
    i = [NO, ND, MX_times, MX_N, min_s, OA_s_DATA, OA_s_LENS, OA_s_M,
        DA_s_DATA, DA_s_LENS, DA_s_M, OATN_DATA, DATN_DATA, OTNix, DTNix]
    o = [times, Oroutes, Droutes]
    w = [O_s, OA, D_s, DA]
    #test# for v in i+o+w: print(Qstr(v))
    run_cl_program(COMPUTE['BUILT']['ALL_TO_ALL'].A,
                   COMPUTE['CL_CONTEXT'], (ND,), i, o, [], w)
##    print('OA_s_DATA',Qstr(OA_s_DATA))
##    print('OATN_DATA',Qstr(OATN_DATA))
##    print('DA_s_DATA',Qstr(DA_s_DATA))
##    print('DATN_DATA',Qstr(DATN_DATA))
##    print('times',Qstr(times))
##    print('NO',NO,'ND',ND)
    return [times, Oroutes, Droutes]




def call(func, params=[]):
    if (Qtype(func) in Qmx('function method') or hasattr(func,'__call__')) \
       and Qtype(params) in Qmx('tuple list dict'):
        print('Calling',Qstr(func),' with','['+', '.join(Qstr(p) for p in params)+']...')
        t0 = time_time()
        out = func(**params) if Qtype(params)=='dict' else func(*params)
        t = time_time()-t0
        print('...',Qstr(func),'done in %.2f'%t,'seconds.')
        return out
    Qer('Bad QcF call!')


with open('test.pic','rb') as file:
    params = pickle.load(file) + [COMPUTE]
route0 = call(_route_all_to_all, params)
params[0] = 1
print()
route1 = call(_route_all_to_all, params)

print(sum(Ql(route0[0]!=route1[0])))


