#################################
# Avoid Premature Optimization !#
#################################

############# BUGS ###################
# ? What to do with Wrk/Res points too far away from road NW. 
# ? What to do with hash misses in points outside lrbt.
########################################

# TO_DO:
# Netowrk will be meltable: xy_LoLs in TRoads - should be split to take advantage of Q!


import pickle
from utils import *
from qcommands import *
from geometry2 import HashGrid, dist
from earth import Geometry
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from numba import njit

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
    #for P in PARAMS: print(type(P),P.dtype,P.shape)
    if use_CL: return _CL_all_to_all(*PARAMS, COMPUTE)
    else: return [_JIT_all_to_all(*PARAMS), Oroutes, Droutes]
    
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
            float trip = O_s[o] + D_s[d] + MX_times[DTNix[DA[d]]*MX_N+OTNix[OA[o]]]; 
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


def _CL_all_to_all(times, Oroutes, Droutes, # outputs
                    NO, ND, MX_times, MX_N, min_s,
                    OA_s_DATA, OA_s_LENS, OA_s_M,
                    DA_s_DATA, DA_s_LENS, DA_s_M,
                    OATN_DATA, DATN_DATA,
                    OTNix, DTNix, COMPUTE):
    Qer("This code is buggy. Please use JIT version for now.")
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
    


# ! uint8 is valid if there are at most 127 anchors per Res/Wrk -  should be ok:) (currently max=2)
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
    d_min = np_float32(0)
    o_min = np_float32(0)
    for nD in range(ND):
        dl = DA_s_LENS[nD]
        for nO in range(NO):
            ol = OA_s_LENS[nO]
            mini = np_float32(np_inf)
            d_min = -1; o_min = -1
            for o in range(ol):
                O_s[o] = OA_s_DATA[nO,o]
                OA[o] = OATN_DATA[nO,o]
                for d in range(dl):
                    D_s[d] = DA_s_DATA[nD,d]
                    DA[d] = DATN_DATA[nD,d]
                    # Note Dijkstra matrix must be read: [D,O]
                    trip = O_s[o] + D_s[d] + MX_times[DTNix[DA[d]],OTNix[OA[o]]]
                    if trip < mini:
                        mini = trip
                        d_min = d
                        o_min = o
            if mini < min_s: mini = np_float32(np_inf) # this line eliminates short paths. 
            times[nO,nD] = mini
            Oroutes[nO,nD] = o_min # these are indices.
            Droutes[nO,nD] = d_min # "
    return times

class Segments:
    def __init__(self, TNs, TRs):
        # find minimum modulo hash that can differentiate all TN neighbours: 
        neighb_modulo = Qu32(len(TNs.neighbTNs), 0)
        for n,s in enumerate(TNs.neighbTNs):
            v = 1
            m = {0}
            s = set(s)
            while(len(m)<len(s)):
                v += 1
                m = set([x%v for x in s])
            neighb_modulo[n] = v
        self.neighb_modulo = neighb_modulo
        self.neighb_modulo_max = max(neighb_modulo)     
        # Place TRs in modulo entry for each TN:
        tn2tr = [[0]*nm for nm in neighb_modulo]
        tn2trbwd = [[0]*nm for nm in neighb_modulo]
        startTN = TRs.startTN
        endTN = TRs.endTN
        for tr in Qr(TRs):
            if 1:# TO_DO: This will overwrite TRs with same end-points - only write if this is shortest link.
                stn = startTN[tr]
                etn = endTN[tr]
                E = etn%neighb_modulo[stn]
                S = stn%neighb_modulo[etn]
                tn2tr[stn][E] = tr
                tn2tr[etn][S] = tr
                tn2trbwd[stn][E] = 0
                tn2trbwd[etn][S] = 1
        self.tn2tr = Block(tn2tr,'uint32')
        self.tn2trbwd = Block(tn2trbwd,'uint8')
            
            

##def route_centrality____SLOW(NW):
##    TRcount = np_zeros(len(NW.TRs),'float64')
##    SEG = Segments(NW.TNs,NW.TRs)
##    neighb_modulo = SEG.neighb_modulo
##    tn2tr = SEG.tn2tr
##    Oweight = NW.Res.pop
##    Dweight = NW.Wrk.pop
##    G = NW.graph['Work']
##    OATN = G.OATN
##    DATN = G.DATN
##    Oroute = G.Oroute
##    Droute = G.Droute
##    A2Aroute = G.A2Aroute
##    MX_prede = G.MX_prede
##    DTNix = G.DTNix
##    OTNix = G.OTNix
##    NO, ND = A2Aroute.shape
##    for o in range(NO):
##        if not o%100: print(o)
##        for d in range(ND):
##            if A2Aroute[o,d]<1801.0:
##                #print('\n@',o,d)
##                STN = OATN[o][Oroute[o,d]]
##                ETN = DATN[d][Droute[o,d]]
##                #print(STN,ETN)
##                TN1 = STN
##                TN2 = MX_prede[DTNix[ETN]][OTNix[STN]]
##                if STN!=ETN:
##                    while 1:
##                        TN2 = MX_prede[DTNix[ETN]][OTNix[TN1]]
##                        tr = tn2tr[TN1][TN2%neighb_modulo[TN1]]
##                        TRcount[tr] += Oweight[o] * Dweight[d]
##                        #print(tr,end='-')
##                        if TN2 == ETN: break
##                        TN1 = TN2
##                    #input()
##    return TRcount



@njit("float64[:,:](float64[:,:],uint32[:],uint32[:,:],uint8[:,:],"+\
      "float32[:],"*2+"uint32[:,:],"*2+"int8[:,:],"*2+"float32[:,:],"+\
      "int32[:,:],"+"uint32[:],"*2+"uint64,"*2+"float64, float64)")
def _JIT_route_centrality(TRcount,neighb_modulo,tn2tr,tn2trbwd,
                          Oweight,Dweight,OATN,DATN,Oroute,Droute,A2Aroute,
                          MX_prede,DTNix,OTNix,NO,ND, max_s, min_s):
### !!! Commented code does not have distinction TRfwdcount/TRbwdcount.

##    if 0: # unweighed alg. 
##        for o in range(NO):
##            for d in range(ND):
##                if A2Aroute[o,d]<max_s:           
##                    weight = Oweight[o] * Dweight[d]
##                    STN = OATN[o][Oroute[o,d]]
##                    ETN = DATN[d][Droute[o,d]]
##                    TN1 = STN
##                    TN2 = MX_prede[DTNix[ETN]][OTNix[STN]]
##                    if STN!=ETN:
##                        while 1:
##                            TN2 = MX_prede[DTNix[ETN]][OTNix[TN1]]
##                            tr = tn2tr[TN1][TN2%neighb_modulo[TN1]]
##                            TRcount[tr] += weight
##                            if TN2 == ETN: break
##                            TN1 = TN2
    if 1: # weighed alg. - both O and D normalized
##        Osumweight = np_ones(NO, dtype=np_float32)
##        Dsumweight = np_ones(ND, dtype=np_float32)
##        if 0: # turn on sum normalization (if needed)
##            # find sum weights from each origin to its reachability neighbourhood:
##            for o in range(NO):
##                osw = 0.0
##                for d in range(ND):
##                    a2a = A2Aroute[o,d]
##                    if a2a < max_s and a2a > min_s:
##                        osw += Dweight[d]
##                Osumweight[o] = osw
##            # same, for destinations:
##            for d in range(ND):
##                dsw = 0.0
##                for o in range(NO):
##                    a2a = A2Aroute[o,d]
##                    if a2a < max_s and a2a > min_s:
##                        dsw += Oweight[o]
##                Dsumweight[d] = dsw
        # find path weight and add it to all path segments:
        for o in range(NO):
            for d in range(ND):
                a2a = A2Aroute[o,d]
                if a2a < max_s and a2a > min_s: 
                    weight = Oweight[o] * Dweight[d] #/ Dsumweight[d] / Osumweight[o]
                    STN = OATN[o][Oroute[o,d]]
                    ETN = DATN[d][Droute[o,d]]
                    TN1 = STN
                    DTNixETN = DTNix[ETN]
                    TN2 = MX_prede[DTNixETN][OTNix[STN]]
                    if STN!=ETN:
                        while 1:
                            TN2 = MX_prede[DTNixETN][OTNix[TN1]]
                            m = TN2%neighb_modulo[TN1]
                            TRcount[tn2tr[TN1][m]][tn2trbwd[TN1][m]] += weight
                            if TN2 == ETN: break
                            TN1 = TN2
    return TRcount




        

                


class Graph:
    def __len__(self): return self._len
    def __init__(self, _len, starts, ends, Fs, Bs, Fmps, Bmps, walk_mps, max_s, min_s, COMPUTE):
        Qsave(self, locals(), '*')
    def build_mx(self, OTNs=True, DTNs=True, Predecessors=False):# Build inter-TN time matrix
        if OTNs is True: OTNs = tuple(Qr(self))
        else:
            print('Graph.build_mx(OTNs not True) not implemented yet...')
            return False
        if DTNs is True: DTNs = tuple(Qr(self))
        def array_dict(vec):
            a = Qu32(max(vec)+1, 0)
            for i,n in enumerate(vec): a[n] = i
            return a
        self.OTNix = array_dict(OTNs)
        self.DTNix = array_dict(DTNs)
        print(len(self),'x',len(DTNs),'[float64]')
        ty = 'float32'
        sh = (self._len,self._len)
        MX_links = csr_matrix((self.Fs,(self.starts,self.ends)),shape=sh,dtype=ty) +\
                   csr_matrix((self.Bs,(self.ends,self.starts)),shape=sh,dtype=ty)
        D = dijkstra(MX_links,indices=DTNs,
            directed=True,unweighted=False,
            return_predecessors=Predecessors,
            limit = self.max_s)
        if Predecessors:
            self.MX_times = Qf32(D[0]) # cast for openCL
            self.MX_prede = D[1]
        else: self.MX_times = Qf32(D) # cast for openCL
    def build_anchors(self, TRs, side, locations, indexes=True):
        if indexes is True: indexes = Qr(locations)
        ATN = []; A_s = []
        w_mps = self.walk_mps
        Fmps = self.Fmps
        Bmps = self.Bmps
        for i in indexes:
            w_m = locations.walk_m[i]
            N = len(w_m)
            Anchors = {}
            for n in Qr(N):
                w_s = w_m[n]/w_mps
                tr = locations.anchorTR[i][n]
                time1 = w_s + locations.anchor_m1[i][n]/Fmps[tr]   # FWD and BWD speeds on that road
                time2 = w_s + locations.anchor_m2[i][n]/Bmps[tr]
                for t,A in ((time1, locations.anchor1[i][n]),(time2, locations.anchor2[i][n])):
                    if A not in Anchors: Anchors[A]=t
                    else: Anchors[A] = min(Anchors[A],t)
            Anchors = [(A,t) for A,t in Anchors.items()]
            ATN.append([A[0] for A in Anchors])
            A_s.append([A[1] for A in Anchors])
        if side == 0: # Origins
            self.OATN = Block(ATN,'uint32')
            self.OA_s = Block(A_s,'float32')
            self.OTNs = tuple(set(lol_to_tuple(ATN)))
            return self.OTNs
        else: # Destinations
            self.DATN = Block(ATN,'uint32')
            self.DA_s = Block(A_s,'float32')    
            self.DTNs = tuple(set(lol_to_tuple(ATN)))
            return self.DTNs
    def all_to_all(self):
        for JIT_CL in [0]: # CL is not debugged
            params = [JIT_CL, self.MX_times, self.min_s, 
                  self.OA_s, self.DA_s, self.OATN, self.DATN,
                  self.OTNix, self.DTNix, self.COMPUTE]
##        with open('test.pic','wb') as file:
##            pickle.dump(params[:-1], file)
            route = QcFs(_route_all_to_all, params)
        #print('SQerror:', max((np_nan_to_num((route[0][0]-route[1][0]),nan=0)**2).flatten()))
        #print('Oroute:',sum(route[0][1]!=route[1][1]))
        #print('Droute:',sum(route[0][2]!=route[1][2]))
        #if not hasattr(self, 'route'): self.route = route ###
        #route = route[0] ###
        self.A2Aroute = route[0]
        self.Oroute = route[1]
        self.Droute = route[2]
        return self.A2Aroute
##        for nD, D in enumerate(DD):
##            for nO, O in enumerate(OO):                
##                O_s = self.OA_s[O]
##                D_s = self.DA_s[D]
##                OA = self.OATN[O]
##                DA = self.DATN[D]
##                # Note Dijkstra matrix must be read: [D,O]
##                times[nO,nD] = min(min(O_s[o] + D_s[d]
##                    + self.MX_times[self.DTNix[DA[d]],self.OTNix[OA[o]]]
##                    for o in range(len(OA))) for d in range(len(DA)))        
##        return times
    def __call__(self, O, D): # Route one-to-one one by one: OTN to DTN
        O_s = self.OA_s[O]
        D_s = self.DA_s[D]
        OA = self.OATN[O]
        DA = self.DATN[D]
        # Note Dijkstra matrix must be read: [D,O]
        return min(min(O_s[o] + D_s[d]
            + self.MX_times[self.DTNix[DA[d]],self.OTNix[OA[o]]]
            for o in range(len(OA))) for d in range(len(DA)))
    def BACKUPone_to_one(self, OO, DD=None): # Route OTN to DTN
        if DD is None: DD = OO
        return [self(OO[n],DD[n]) for n in range(len(OO))]
    def one_to_one(self, OO, DD=None): # O-D study: Route OTN to DTN
        if DD is None: DD = OO
        N = len(OO)
        OA_s = self.OA_s
        DA_s = self.DA_s
        OATN = self.OATN
        DATN = self.DATN
        MX_times  = self.MX_times
        M = self._len
        OTNix = self.OTNix
        DTNix = self.DTNix
        return [min(min(OA_s[n][o] + DA_s[n][d] + MX_times[DTNix[DATN[n][d]],OTNix[OATN[n][o]]]
            for o in Qr(OATN[n])) for d in Qr(DATN[n])) for n in Qr(N)]



class TNodes: # topological (routing) nodes
    def xy(self): return tuple(zip(self.x,self.y))
    def __init__(self, Roads=None):
        self.lols = ['neighbTNs TRconn TRsens TRazim'.split()]
        self.arrays = 'x y'.split()
        self.roads = Roads
        if Roads:
            # STEP 1: identify all TNodes:
            print('NodeTypes')
            Nn = Roads.node_x.size
            NodeType = Qi8(Nn, 0)
            for l in Roads.lines:
                NodeType[l[0]] += 2
                NodeType[l[-1]] += 2
                for n in l: NodeType[n] += 1
            IDs = [n for n in Qr(Nn) if NodeType[n]>1]
            self.N = len(IDs)
            IDs = Qu32(IDs)
            print('IndexDict(IDs)')
            self.IXs = IndexDict(IDs,keytype='uint32')
            # STEP 2: find TN neighbour Lists:
            print('get_TN_neighbours()...',end='')
            t0 = time_time()
            self.neighbTNs = [[] for n in Qr(self)]
            #self.TRconn = [[] for n in Qr(self)]
            for l in self.roads.lines:
                nd = l[0]
                ix = self.IXs[nd]
                for nd2 in l[1:]:
                    ix2 = self.IXs[nd2]
                    if ix2 is not None:
                        self.neighbTNs[ix2].append(ix)
                        self.neighbTNs[ix].append(ix2)
                        nd = nd2
                        ix = ix2
            self.neighbTNs = LoL(self.neighbTNs,'uint32')
            #self.TRconn = LoL(self.TRconn,'uint32')
##            # STEP 3: for each TN, find minimum modulo hash that can differentiate all TN neighbours:
##            neighb_modulo = np_zeros(len(self.neighbTNs), 'uint32')
##            for n,s in enumerate(self.neighbTNs):
##                v = 1
##                m = {0}
##                s = set(s)
##                while(len(m)<len(s)):
##                    v += 1
##                    m = set([x%v for x in s])
##                neighb_modulo[n] = v
##            self.neighb_modulo = neighb_modulo
            # STEP 4: TN coordinates in MAP space:
            self.x = Roads.node_x[IDs]
            self.y = Roads.node_y[IDs]
            print('done in %.2f seconds.'%(time_time()-t0))
    def __len__(self): return self.N





class TRoads: # topological road lines
    def __len__(self): return self._len
    def _getnames(self):
        L = [self.roads.property, self.extra]
        return sum([list(x.keys()) for x in L],[])
    def find_prop(self, name):
        f = [name in self.roads.property, name in self.extra]
        if sum(f) == 2: Qer('Name collision',name,'in TRoads.')
        if sum(f) == 0: return False
        return 1 if f[0] else 2
    def get(self, name, default=None):
        if name in self: return self[name]
        return default
    def __contains__(self, name): return bool(self.find_prop(name))
    def __getitem__(self, name_id):
        if type(name_id) is int: return self.roads[self.osm_road[name_id]]
        f = self.find_prop(name_id)
        if not f: Qer('Name',name_id,'not found in TRoads.')
        if f == 1: return self.get_osm_attribute(name_id)
        else: return self.extra[name_id]
    def __setitem__(self, name_id, value):
        if name_id in self.roads.property: Qer('Cannot override OSM-based property',name_id,'.')
        self.extra[name_id] = value
    def get_osm_attribute(self,attribute):
        if attribute not in self.osm_attributes:
            self.osm_attributes[attribute] = self.roads.property[attribute][self.osm_road]
        return self.osm_attributes[attribute]
    def get_xy_LoLs(self):
        if not self.xy_LoLs:
            data = [v[self.lines.data] for v in (self.roads.node_x, self.roads.node_y)]
            lols = [LoL() for n in range(2)]
            for n in range(2): lols[n].load(data[n],self.lines.offsets)
            self.xy_LoLs = lols
        return self.xy_LoLs
    def get_LineStrings(self):
        if not hasattr(self,'LineStrings'):
            x = self.roads.node_x
            y = self.roads.node_y
            self.LineStrings = [LineString([(x[n],y[n]) for n in line]) for line in self.lines]
        return self.LineStrings
    def __init__(self, TNs=None, roads=None):
        self.osm_attributes = {}
        self.extra = {}
        self.xy_LoLs = None
        self.TNs = TNs
        self.roads = roads
        self.lols = ['lines'.split(), 'seg_len_m'.split()]
        self.arrays = 'startTN endTN len_m osm_road'.split()
        if TNs:
            TNIX = TNs.IXs
            x = roads.node_x
            y = roads.node_y
            rl = roads.lines
            rlsm = roads.seg_len_m
            lines = []
            startTN = []
            endTN = []
            seg_len_m = []
            len_m = []
            osm_road = []
            for R in Qr(roads):
                L = rl[R]
                LS = rlsm[R]
                TN = [TNIX[n] for n in L]
                segs = [i for i,tn in enumerate(TN) if tn is not None]
                TN = [tn for tn in TN if tn is not None]
                N = len(segs)-1
                startTN += TN[:-1]
                endTN += TN[1:]
                osm_road += [R]*N
                for i in Qr(N):
                    lines.append(L[segs[i]:(segs[i+1]+1)])
                    lsm = LS[segs[i]:segs[i+1]]
                    seg_len_m.append(lsm)
                    len_m.append(sum(lsm))
            # save:
            self.startTN = Qu32(startTN)
            self.endTN = Qu32(endTN)
            self.lines = LoLu32(lines)
            self.osm_road = Qu32(osm_road)
            self.seg_len_m = LoLf32(seg_len_m)
            self.len_m = Qf32(len_m)
            self._len = len(self.len_m) # replaced by:(doesn't work!):
##            Qsave(self, locals(), 'startTN@u32 endTN@u32 lines@Lu32 osm_road@u32 seg_len_m@Lf32 len_m@f32')
            # about 160 chars vs 100 - not huge saving, but may avoid typos?
            

    
class Network:
    def find_joining_TRs(self, tn1, tn2):
        if tn1==tn2: return []
        TRconn = self.TNs.TRconn
        return list(set(TRconn[tn1]) & set(TRconn[tn2]))
    def export_to_shp(self, centre):
        X = LoL(); Y = LoL()
        data = self.TRs.lines.data #Q
        offsets = self.TRs.lines.offsets #Q
        X.load(self.TRs.roads.node_x[data], offsets) #node_x Q
        Y.load(self.TRs.roads.node_y[data], offsets) #"
        X = X.tolol(); Y = Y.tolol()
        C = [[list(zip(X[n],Y[n]))] for n in Qr(X)]
        C = Geometry(C, form='xy')
        C(centre)
        out = {'polylines':C.lonlat()}
        out['records']= self.TRs.roads.property #Q
        out['field_names'] = list(self.TRs.roads.property.keys()) #Q
        out.update()
        return out
    def export_ui(self):
        out = {}
        for item in Qmx('TNs TRs'):
            ITEM = getattr(self, item)
            for arr in ITEM.arrays:
                if hasattr(ITEM, arr): out[item+':'+arr] = getattr(ITEM, arr)
            for lols in ITEM.lols:
                lol_map = {lol:lols[0] for lol in lols}
                for lol in lols:
                    if hasattr(ITEM, lol):
                        attr = getattr(ITEM, lol)
                        out[item+':'+lol+':lol'] = attr.data
                        out[item+':'+lol_map[lol]+':off'] = attr.offsets
        return out
    def import_ui(self, data):
        for item in Qmx('TNs TRs'):
            ITEM = getattr(self, item)
            for arr in ITEM.arrays:
                if item+':'+arr in data: setattr(ITEM, arr, data[item+':'+arr])
            for lols in ITEM.lols:
                lol_map = {lol:lols[0] for lol in lols}
                for lol in lols:
                    if item+':'+lol+':lol' in data and item+':'+lol_map[lol]+':off' in data:
                        v = LoL()
                        v.load(data[item+':'+lol+':lol'], data[item+':'+lol_map[lol]+':off'])
                        setattr(ITEM, lol, v)
        self.TNs.N = len(self.TNs.neighbTNs)
        self.TRs._len = len(self.TRs.lines)
                         
    def route_centrality(self,G,Oweight,Dweight):
        print('Graph G.MX_times:',Qstr(G.MX_times))
        TRcount = Qf64((len(self.TRs), 2), 0) # route centrality output
        if not self.Segments: self.Segments = QcF(Segments,[self.TNs,self.TRs]) # lazy loading
        NO, ND = G.A2Aroute.shape
        PARAMS = [TRcount,self.Segments.neighb_modulo,
                  self.Segments.tn2tr.data,self.Segments.tn2trbwd.data,
                  Qf32(Oweight),Qf32(Dweight),
            G.OATN.data,G.DATN.data,G.Oroute,G.Droute,G.A2Aroute,
            G.MX_prede,G.DTNix,G.OTNix,NO,ND, G.max_s, G.min_s]
        return _JIT_route_centrality(*PARAMS)
    def __setitem__(self, item, value): self.optional[item] = value
    def __getitem__(self, item): return self.optional[item]
    def __call__(self, profile, threeD, min_s=None): # takes cyclist profile, returns a Graph
        # How is speed calculated?
        # BOTH WAYS:
        # - LTS-table speed
        # - (material) clamp speed
        # EACH WAY:
        # - oneway conditions clamps to LTS0/1
        # - slope penalty should work on LTS 1 speed
        # Load LTS speeds
        epsilon = 10**-20
        LTS_mps = [profile['lts'+str(n)] for n in range(5)] # LTS 0-4 - 0 is walking.
        #walk_mps = LTS_mps[0] # LTS[0] is walking speed, assumed for walking against one-way traffic.
        LTS_mps += [0]*6 # LTS5(forb.) - 10(transit): infinite impedance  
        LTS_mps.append(profile['lts-1']) # LTS-1 (unknown road type)
        LTS_mps = Qf32(LTS_mps) + epsilon
        print('LTS_mps = ', ['%.2f'%v for v in LTS_mps])
        # Find clamped speed in both dections
        TRs = self.TRs
        MPS = {}; P = {}
        oneway = TRs['oneway']
        OW = {'f':oneway>=0, 'b':oneway<=0}
        print('Preparing %dD graph.'%[2,3][threeD]) #TRs.extra #!
        LTS_mps1 = LTS_mps[1]
        Super = TRs.get('Super')
        if Super is not None:
            print(Qstr(Super))
            print('Super count:',sum(Super>0))
        for d in 'fb':
            MPS[d] = TRs['clampSpeed_mps']
            d_danger = d+'_danger'
            P[d_danger] = LTS_mps[TRs[d+'wd_LTS']] #TRs.extra
            if Super is not None:
                for n in Qr(TRs):
                    if Super[n]: P[d_danger][n] = LTS_mps1
            P[d+'_oneway'] = Qf32([LTS_mps[0], LTS_mps1])[OW[d].astype('uint8')] # must be astype, in {0,1}
            if threeD:
                P[d+'_hill'] = LTS_mps[1]/(1+TRs[d+'wd_climb']) #TRs extra
            for k in P:
                if k[0] == d: MPS[d] = np_minimum(MPS[d], P[k])
        FB_s = [TRs.len_m/MPS[d] for d in 'fb'] 
        if 1: #durations (public transport: ferries)
            DFs, DBs = [TRs[d+'wd_dur_s'] for d in 'fb']
            # ferry code:
            ferry = TRs['passageType']==ord('f')
            for n, v in enumerate(DFs):
                if v and ferry[n]: FB_s[0][n] = v
            for n, v in enumerate(DBs):
                if v and ferry[n]: FB_s[1][n] = v
        min_time = profile['min_time'] if min_s is None else min_s # hotwired to 0 for RW study for now.
        G = Graph(len(self.TNs), TRs.startTN, TRs.endTN,
                  FB_s[0], FB_s[1], MPS['f'], MPS['b'],
                  LTS_mps[0], profile['max_time'],
                  min_time,self.COMPUTE)
        self.graph[profile['profile']] = G
    def __init__(self, Map, GLOBAL, build = True):
        self.COMPUTE = GLOBAL["COMPUTE"]
        self.Map = Map
        self.graph = {}
        self.optional = {}
        self.terminals = {}
        self.Segments = None
        if build:
            self.TNs = QcF(TNodes, [self.Map.roads])
            self.TRs = QcF(TRoads, [self.TNs, self.Map.roads])
            del self.TNs.IXs
            QcF(self.make_topology_info,[])
            print('Network done: %d TNs, %d TRs.'%(len(self.TNs),len(self.TRs)))
        else:
            self.TNs = TNodes()
            self.TRs = TRoads()
            self.TRs.TNs = self.TNs
            self.TRs.roads = self.Map.roads
    def spatial_hash(self, GLOBAL):
        if 1:
            print('Spatial-hashing...')
            ###X,Y = Q(self, "TNs nodes x<y")
            X = self.TNs.x
            Y = self.TNs.y
            self.grid_m = GLOBAL['ALGORITHM']['MAP_HASH_GRID_m']
            S_m = self.grid_m
            if 'analysis_boundary' in self.Map:
                map_lrbt = self.Map['analysis_boundary'].get_lrbt('m')
            else:
                RX = self.TRs.roads.node_x
                RY = self.TRs.roads.node_y
                ####in Q, this becomes: RX,RY = Q(self,'TRs roads node_x<node_y')
                map_lrbt = [RX.min(),RX.max(),RY.min(),RY.max()]
            self.hashgrid = HashGrid(GLOBAL['COMPUTE'],
                [map_lrbt[0]-S_m, map_lrbt[1]+S_m,
                 map_lrbt[2]-S_m, map_lrbt[3]+S_m], S_m)
        if 0:
            self.hashgrid.hash_points('TNs', tuple(zip(X,Y)))
        if 1:
            XY = self.TRs.get_xy_LoLs()
            self.hashgrid.hash_lines('TRs', *[C.data for C in XY],
                                     self.TRs.lines.offsets,
                                     self.TRs.seg_len_m.data)
    def ResParcels(self, polygons=None, populations=None, res_m=50):
        if polygons: self.Map['ResPar'] = Geometry(polygons) #mount
        self.Prc = self.Map['ResPar']
        if populations: self.Prc.pop = populations
        else: self.Prc.pop = self['ResPaP']
        #####self.Prc.NW = self
        (self['Res'], self['ResPoP']) = self._place_residences(self.Prc, res_m)
    def _place_residences(self, parcels, res_m=50):
        HG = self.hashgrid
        TRs_h = HG.hashed['TRs']
        TRLS = self.TRs.get_LineStrings()
        hasHouses = self.TRs['hasHouses']
        POINTS = []; WEIGHTS = [];
        print('len(parcels):',len(parcels))
        for n,P in enumerate(parcels.xy()):
            if parcels.pop[n] > 0.0:
                #print(n,len(parcels), len(P), parcels.pop[n])
                print('%d[%d]'%(n,len(P)),end='')
                #######if len(P)>200: continue###
                trs = np_unique(np_concatenate([TRs_h[hp]//(2**32) for hp in HG._hash_polygon(P)])) ### trs = set(sum([list(TRs[hp]//2**32) for hp in HP],[]))
                SP = sPolygon(P).buffer(0.01).simplify(0.01) # fix broken polygons (tol=1cm)
                LS = []
                for tr in trs:
                    ls = TRLS[tr]
                    if not hasHouses[tr]: pass
                    elif SP.covers(ls): LS.append(ls)
                    elif SP.crosses(ls):
                        i = SP.intersection(ls)
                        t = i.geom_type
                        if t == 'LineString': LS.append(i)
                        else: LS += [l for l in i.geoms] # extract LS's from MultiLinestring
                PTs = [];WGs = []
                for ls in LS:
                    ln = ls.length
                    if ln < res_m:
                        PTs.append(ls.interpolate(ln/2).coords[0])
                        WGs.append(ln)
                    else:
                        d = (ln%res_m)/2
                        M = int(ln//res_m)+1
                        for m in Qr(M):
                            PTs.append(ls.interpolate(d+m*res_m).coords[0])
                            if m==0 or m==M: WGs.append(d+res_m/2)
                            else: WGs.append(res_m)
                sWGs = sum(WGs)
                if sWGs:
                    K = parcels.pop[n]/sWGs
                    POINTS += PTs
                    WEIGHTS += [wg*K for wg in WGs]
        print('\n')
        return (POINTS, WEIGHTS)
    def make_topology_info(self):
        ### Azimuth code is wrong.
        TRs = self.TRs
        TNs = self.TNs
        lines = TRs.lines
        startTN = TRs.startTN
        endTN = TRs.endTN
        X = TRs.roads.node_x
        Y = TRs.roads.node_y
        TRconn = [[] for _ in Qr(TNs)]
        TRazim = [[] for _ in Qr(TNs)]
        TRsens = [[] for _ in Qr(TNs)]
        for tr in Qr(TRs):
            sn = startTN[tr]
            en = endTN[tr]
            TRconn[sn].append(tr)
            TRconn[en].append(tr)
            TRsens[sn].append(0) #start of segment
            TRsens[en].append(1) #end of segment
            x = [X[n] for n in lines[tr]]
            y = [Y[n] for n in lines[tr]]         
            TRazim[sn].append(atan2(y[1]-y[0],x[1]-x[0]))
            TRazim[en].append(atan2(y[-2]-y[-1],x[-2]-x[-1]))
        neighbTNs = []
        for tn in range(len(TNs)):
            order = Qr(TRazim[tn])
            order = [x for _,x in sorted(list(zip(TRazim[tn],order)),key=lambda p:p[0])]
            TRazim[tn] = [TRazim[tn][o] for o in order]
            TRconn[tn] = [TRconn[tn][o] for o in order]
            TRsens[tn] = [TRsens[tn][o] for o in order]
            # ! neighbTNs have a different order and must be rebuilt!
            neigh = []
            for n,tr in Qe(TRconn[tn]):
                if TRsens[tn][n]==1:neigh.append(startTN[tr]) # append the opposite!
                else:neigh.append(endTN[tr])
            neighbTNs.append(neigh)
        #Qsave(TNs, locals(), "TRconn@Lu32 TRsens@Lu8 TRazim@Lf32 neighbTNs@Lu32") # replaces:
        TNs.TRconn = LoLu32(TRconn)
        TNs.TRsens = LoLu8(TRsens)
        TNs.TRazim = LoLf32(TRazim)
        TNs.neighbTNs = LoLu32(neighbTNs) # new: reordered
        for i in [TNs.TRconn, TNs.TRsens, TNs.TRazim]: i(TNs.neighbTNs) # save on mem
        
##    def subNetwork(self, nodeIDs): # TO_DO: finish TNs.subset()
##        SN = copy(self)
##        SN.TNs = self.TNs.subset(nodeIDs)
##        return SN 

# rewrite similarly to Workplaces():
##    def Residences(self, points=None, PoP=None):
##        if points: self.Map['Res'] = Geometry(points)#mount
##        self.Res = self.Map['Res']
##        if PoP: self.Res.pop = PoP
##        else: self.Res.pop = self['ResPoP']
##        self._anchor_geometry(self.Res)
##        self.Res.NW = self

    # Should be modified to use dijkstra distance
    def groupTerminals(self, name, d_m):
        print('Grouping terminals',name,'...')
        T = self.terminals[name]
        # assumes there is only one anchor per point (as is the case right now)
        # some points are too far and don't have an anchor (TODO: debug these points)
        L = T.anchor_x.lens()
        C = [None]*2
        for i in Qr(2):
            C[i] = Qf32(c[0] if L[n] else float('inf') for n,c in Qe([T.anchor_x,T.anchor_y][i]))
        XY = np_stack(C,axis=1)
        neighb = []
        AR = np_arange(len(XY))
        for n,xy in Qe(XY):
            D = cdist(XY, xy.reshape((1,2)))
            S = set(AR[D[:,0]<d_m])
            S.discard(n)
            neighb.append(S)
        # TODO: add dijkstra filter here
        A = set(AR)
        replace = {}
        for s in neighb:
            a = []
            for v in s:
                if v in A:
                    A.discard(v)
                    a.append(v)
            if a: replace[a[0]] = a[1:]
        print('Grouping ratio achieved:', len(XY)/len(replace))
        if 0: # for debugging: check grouping assignment visually:
            # do not change walk_m:
            for prop in Qmx("anchor1 anchor2 anchor_m1 anchor_m2 anchor_x anchor_y anchorTR"):
                var = getattr(T,prop) # all LoL
                for h,b in replace.items():
                    for v in b: var[v][0] = var[h][0]
        if 1: # reduces terminals to cluster heads only:
            heads = Qu8(len(XY),0)
            for k in replace.keys(): heads[k] = 1
            pop = []
            geo = []
            p = T.pop
            g = T.xy()
            for n,v in Qe(heads):
                if v:
                    pop.append(sum(p[x] for x in [n]+replace[n]))
                    geo.append(g[n])
            NEW = Geometry(geo, form='xy')
            NEW.pop = Qf32(pop)
            for prop in Qmx("walk_m anchor1 anchor2 anchor_m1 anchor_m2 anchor_x anchor_y anchorTR"):
                p = getattr(T,prop)
                lol = LoL([v for n,v in Qe(p.tolol()) if heads[n]], p.dtype) # all LoL
                setattr(NEW,prop,lol)
                getattr(NEW,prop)(NEW.walk_m) # save memory
            self.terminals[name] = NEW
    def Terminals(self, name, G2):
        self.terminals[name] = G2
        self.terminals[name].pop = G2['count']
        print('Anchoring points in "%s"...'%name)
        #Q(G2)
        self._anchor_geometry(self.terminals[name])
        print('Done!')
    def _anchor_geometry(self, Geo, max_walk = 10000.0):
        if Geo.shape == 2: #points
            # Q...(locals(),"]anchor1 anchor2 anchor_m1 anchor_m2 anchor_x anchor_y walk_m anchorTR") # 87 chars vs 105 - not very useful?
            anchor1 = [];anchor2 = []
            anchor_m1 = [];anchor_m2 = []
            anchor_x = [];anchor_y = []
            walk_m = []
            anchorTR = []
            # Q...(locals(),self, "TRs len_m<startTN<endTN") # 46 chars vs 69 - 1/3 better.
            len_m = self.TRs.len_m
            startTN = self.TRs.startTN
            endTN = self.TRs.endTN
            #
            LineStrings = self.hashgrid.geometry['TRs']
            for count, P in enumerate(Geo.xy()):
                TR = self.hashgrid.find_one_line(P, 'TRs', max_walk)
                if (count+1)%10000==0: print(count+1,'/',len(Geo))
                if TR is not None:
                    anchor1.append([startTN[TR]])
                    anchor2.append([endTN[TR]])
                    line = LineStrings[TR]
                    point = Point(P)
                    walk_m.append([line.distance(point)])
                    proj = line.project(point)
                    anchor_m1.append([proj])
                    anchor_m2.append([len_m[TR] - proj])
                    xy = line.interpolate(proj)
                    anchor_x.append([xy.x])
                    anchor_y.append([xy.y])
                    anchorTR.append([TR])
                else:
                    for v in [anchor1, anchor2, walk_m, anchor_m1, anchor_m2, anchor_x, anchor_y, anchorTR]:
                        v.append([])
##                    print('Fatal Error in importing OD:')
                    print('Point',P,'is not within',max_walk,'m of any known road.')
##                    return False
            Qsave(Geo, locals(),"anchor1@Lu32 anchor2@Lu32 anchor_m1@Lf32 anchor_m2@Lf32 anchor_x@Lf32 anchor_y@Lf32 walk_m@Lf32 anchorTR@Lu32")
            for O in [Geo.anchor1, Geo.anchor2, Geo.anchor_m1, Geo.anchor_m2, Geo.walk_m]: 
                Geo.anchorTR(O) # saves on memory - common LoL shape
        else:
            print('ERROR, anchoring not implemented!')
            return False
    def get_anchor_links_G3(self,Geo):
        LINKS = []
        #ax, ay, gx, gy = Q(Geo, 'anchor_x, anchor_y, data x array!<<y array!')
        ax = Geo.anchor_x
        ay = Geo.anchor_y
        gx = Geo.data['x'].array()
        gy = Geo.data['y'].array()
        for n in Qr(Geo):
            for a in Qr(ax[n]):
                LINKS.append(((gx[n],gy[n]),(ax[n][a],ay[n][a])))
        return Geometry(LINKS,form='xy')
    def export_simple_network(self): # for testing and otehr projects, COVID etc.
        out = {}
        for k in 'xy':
            out[k] = getattr(self.TNs,k).tolist()
        out['neighb'] = n.TNs.neighbTNs.tolol()
        return out
        
        
        




##class DistanceMatrix_OBSOLETE_______________:
##    def __init__(self, NW, mode = 'path', unit_m=1.0, w_max=2**16-2): # note, 2**16-2 is the highest encodable distance with unit_m=1, as 2**16-1 represents infinity        
##        self.unit_m = unit_m
##        self.w_max = w_max
##        self.N = NW.NN
##        self.D = np_empty([self.N,self.N],dtype=np_uint16)
##        if mode == 'path': # shortest paths distances:
##            # Build sparse one-hop distance matrix H:
##            def linearize(ll, dtype):
##                return np_array([i for sl in ll for i in sl], dtype=dtype)
##            data = linearize(NW.Elenm,np_float32)
##            rows = linearize(NW.Eadjup,np_int32)
##            columns = linearize([[n]*len(NW.Eadjup[n]) for n in range(NW.NN)],np_int32) 
##            H = scipy.sparse.csr_matrix((data, (rows,columns)), shape=(NW.NN, NW.NN), dtype=np_float32)
##            H = H + H.transpose()
##            engine.path_distances(self, H)
##
##
##class Network_OBSOLETE___:
##    def export(self, filename, add={}):
##        RN = range(self.NN)
##        centre = ((max(self.Plon)+min(self.Plon))/2, (max(self.Plat)+min(self.Plat))/2)
##        xy = deg2m(centre, [(self.Plon[n], self.Plat[n]) for n in RN])
##        #data = {'Plon':self.Plon,'Plat':self.Plat}
##        data = {'x':[xy[n][0] for n in RN], 'y':[xy[n][1] for n in RN]}
##        data.update(add)
##        data['edges'] = self.edges
##        pad = 100 # meters border around map
##        data['lrbt'] = [min(data['x'])-pad, max(data['x'])-pad, min(data['y'])-pad, max(data['y'])-pad] 
##        with open(filename, mode='w') as file:   
##            json.dump(data,file,sort_keys=True)
##
##    def __init__(self, G):
##        osmN = G.nodes(data=True)
##        osmE = G.edges(data=True, keys=True)
##        ### Trying to clean up the graph.
##        ### What about https://github.com/mikolalysenko/clean-pslg  ? This is js, not py!
##        ##osmEdupl = G.edges(data=True, keys=True)
##        ##
##        ##osmways = set()
##        ##osmE = []
##        ##for edge in osmEdupl:
##        ##    osmid = edge[3]['osmid']
##        ##    if type(osmid) is list:
##        ####        osmid.sort()
##        ####        osmid = ' '.join([str(n) for n in osmid])
##        ####        print(osmid)
##        ##        osmE.append(edge)
##        ##    elif True:#osmid not in osmways:
##        ##        osmways.add(osmid)
##        ##        osmE.append(edge)
##        NN = len(osmN)
##        NE = len(osmE)
##        osmNid2ix = {osmN[n][0]:n for n in range(NN)}
##        Plat = [osmN[n][1]['y'] for n in range(NN)]
##        Plon = [osmN[n][1]['x'] for n in range(NN)]
##        Elenm = [[] for n in range(NN)]
##        Eadjup = [[] for n in range(NN)]
##        Eadjdown = [[] for n in range(NN)]
##        Elines = [set() for n in range(NN)]
##        Ecurves = [[] for n in range(NN)]
##        for n in range(NE):
##            e = osmE[n]
##            lm = e[3]['length']
##            A = osmNid2ix[e[0]]
##            B = osmNid2ix[e[1]]
##            m = min(A,B)
##            M = max(A,B)
##            if 'geometry' not in osmE[n][3]: Elines[m].add(M) 
####            else:
####                xy = osmE[n][3]['geometry'].coords.xy
####                nex = len(xy[0])-2
####                idx = len(Plat)
####                path = [A]+list(range(idx,idx+nex))+[B]
####                if A<B: path.reverse()
####                duplicate = False
####                for i in range(n):
####                    if Epaths[i][0] == min(A,B) and Epaths[i][-1] == max(A,B):
####                        lp = len(path) 
####                        if lp == len(Epaths[i]):
####                            if A<B: dd = [latlon_dist(xy[1][lp-k],xy[0][lp-k], Plat[Epaths[i][k]], Plon[Epaths[i][k]]) for k in range(lp)]
####                            else: dd = [latlon_dist(xy[1][k],xy[0][k], Plat[Epaths[i][k]], Plon[Epaths[i][k]]) for k in range(lp)]
####                            ########print(dd)
####                            if max(dd) < 1:
####                                duplicate = True
####                if not duplicate:
####                    Epaths[n] = tuple(path)
####                    Plat += xy[1][1:-1]
####                    Plon += xy[0][1:-1]
##            if A != B:
##                if A<B: A,B = B,A
##                if B not in Eadjup[A]:
##                    Eadjup[A].append(B)
##                    Eadjdown[B].append(A)
##                    Elenm[A].append(lm)
##                else:
##                    i = Eadjup[A].index(B)
##                    if lm < Elenm[A][i]: Elenm[A][i] = lm
##        edges = []
##        for n in range(NN):
##            for m in list(Elines[n]):
##                edges.append((n,m))
##        # Structure:
##        self.Eadjup = Eadjup
##        self.Elenm = Elenm
##        self.edges = edges
##        self.osmNid2ix = osmNid2ix
##        self.Plat = Plat
##        self.Plon = Plon
##        self.NN = NN
##        self.NE = len(self.edges)
##        self.W = None
##        
##
##    def all_walk(self, w_max=65000):
##        if self.W is not None: return self.W
##        self.W = DistanceMatrix(self, w_max=w_max)
##        return self.W
##
##class DistanceMatrix_OBSOLETE_______________:
##    def __init__(self, NW, mode = 'path', unit_m=1.0, w_max=2**16-2): # note, 2**16-2 is the highest encodable distance with unit_m=1, as 2**16-1 represents infinity        
##        self.unit_m = unit_m
##        self.w_max = w_max
##        self.N = NW.NN
##        self.D = np_empty([self.N,self.N],dtype=np_uint16)
##        if mode == 'path': # shortest paths distances:
##            # Build sparse one-hop distance matrix H:
##            def linearize(ll, dtype):
##                return np_array([i for sl in ll for i in sl], dtype=dtype)
##            data = linearize(NW.Elenm,np_float32)
##            rows = linearize(NW.Eadjup,np_int32)
##            columns = linearize([[n]*len(NW.Eadjup[n]) for n in range(NW.NN)],np_int32) 
##            H = scipy.sparse.csr_matrix((data, (rows,columns)), shape=(NW.NN, NW.NN), dtype=np_float32)
##            H = H + H.transpose()
##            engine.path_distances(self, H)



if False: # In development: GRAPH TOPOLOGY PARSING
    ######## MAKE NETWORK GRAPH ###########
    # xy: local coords in m
    # nn: list of neighbouring nodes
    # Ordered as nn:
    # nb: list of next branching node for each branch, None if 'antenna'
    # az: atan2 azimuth from node to nn
    # pa: list of adjacent patches
    if True:
        nodes_passed = set()
        nodes_routing = set() 
        BIKEABLE_NODES = {}
    # Build nodes used in bike network, find neighbour node sets
    for w,W in BIKEABLE_WAYS.items():
        seg = W['nodes']
        last = len(seg)-1
        if last: # Skip rare one-node ways, which should not exist according to OSM
            nodes_routing.add(seg[0])
            nodes_routing.add(seg[-1])
            for i, nd in enumerate(seg):
                if i == 0: nix = [1]
                else:
                    if seg[i-1] == nd: print('duplicate node in way:',nd,w) # should not happen [OSM]
                    if i == last: nix = [last-1]
                    else: nix = [i-1,i+1]
                if nd not in nodes_passed: # First time seeing this node ID 
                    nodes_passed.add(nd)
                    try: BIKEABLE_NODES[nd] = {'x':NODES[nd]['x'], 'y':NODES[nd]['y'], 'nn':[seg[ix] for ix in nix]}
                    except: print(w,seg)
                else:
                    nodes_routing.add(nd)
                    BIKEABLE_NODES[nd]['nn'] += [seg[ix] for ix in nix]
    # Now we have nodes classed by neighbours:
    # 1 - end of antenna
    # 2 - transition node
    # >2 - branching node: these nodes only are taken to build patch topology
    nodes_branching = {nd for nd in nodes_routing if len(BIKEABLE_NODES[nd]['nn'])>2}
    # For each branching
    # Totally ignore "antennae" - not topologically relevant (maybe later?)
    if False:
        for nd in nodes_branching:
            BN = BIKEABLE_NODES[nd]
            BN['nb'] = []
            BN['az'] = []
            BN['pa'] = set()
            for nn in BN['nn']:
                BNN = BIKEABLE_NODES[nn]
                BN['az'].append(atan2(BNN['y']-BN['y'],BNN['x']-BN['x']))
                nc,nx = nd,nn
                while nx not in nodes_branching:
                    xx =  set(BIKEABLE_NODES[nx]['nn']).remove(nc)
                    if not xx:
                        nx = None
                        break
                    if len(xx)> 1:
                        Qer('!!!')
                        break
                    else: nc,nx = nx, list(xx)[0]
                BN['nb'].append(nx)
            # sort neighbour info according to azimuth
            # this is not elegant (maybe use zip?)
            keys = 'az nn nb'.split()
            ordered = [(BN['az'][n],BN['nn'][n],BN['nb'][n]) for n in Qr(BN['nn'])]
            ordered.sort()
            for i, k in enumerate(keys): BN[k] = [o[i] for o in ordered]
    # Build patches:
    # nb: osm id's of branching nodes
    # nd: osm id's of all nodes
    # am: area in m^2
    # or: orientation cw/ccw (or winding?)

    if False:
        PATCHES = {}
        for nd in nodes_branching:
            BN = BIKEABLE_NODES[nd]
            for inn, nn in enumerate(BN['nn']):
                patch = []
                nc,nx = nd,nn
                while nx != nd:
                    patch.append(nc)
                    BNX = BIKEABLE_NODES[nx]['nn']
                    nc,nx = nx, BNX[BNX.index(nc)-1]
                ID = len(PATCHES)
                PATCHES[ID] = {'nb':patch}
                for n in patch:
                    if 'pa' in BIKEABLE_NODES[n]: BIKEABLE_NODES[n]['pa'].add(ID)
                    else: BIKEABLE_NODES[n]['pa'] = set([ID])



    if False:
        # TO_DO: Remove duplicate patches here...
        for _,p in PATCHES.items():
            p['nd'] = []
            pnb = p['nb']
            for i,b in enumerate(pnb):
                BN = BIKEABLE_NODES[b]
                nc, nx = b, BN['nn'][BN['nb'].index(pnb[(i+1)%len(pnb)])]
                p['nd'].append(nc)
                while nx not in nodes_branching:
                    xx =  set(BIKEABLE_NODES[nx]['nn']).remove(nc)
                    if len(xx) != 1: Qer('!!!')
                    else:
                        nc,nx = nx, list(xx)[0]
                        p['nd'].append(nc)

    data['points_x'] = []
    data['points_y'] = []
    data['points_data'] = []
    for _,BN in BIKEABLE_NODES.items():
        if True: #'nn' in BN:
            data['points_x'].append(BN['x'])
            data['points_y'].append(BN['y'])
            #data['points_data'].append(sum(bool(i) for i in BN['nb'])/5)
            data['points_data'].append((len(BN['nn'])-2)/5)
            
    ###data['node_data'] = [(-1)**(n not in nodes_intersections) for n in range(len(NODES))]

            
        

##class Node(lat,lon):
##    self.lat = lat
##    self.lon = lon
##    
