from geometry2 import *
from qcommands import *
from popdens import *
import osmfilter
from fileformats import ui_raw_write, ui_raw_read
from numba import njit
import pickle
from shaders import *
import seaborn
from matplotlib import pyplot

SIM_SIZE = 2e8 # approximate amount of items per block simulation


def draw_rw_curves(curves):
    for n, curve in curves.items():
        x = [v[0]/1000 for v in curve]
        y = [v[1] for v in curve]
        x = np_cumsum(x)
        y = Qf32(y)
        y/= max(y)
        seaborn.lineplot(x=x,y=y) # colours change automatically.
    seaborn.set(style='ticks')
    seaborn.set_style("darkgrid")
    pyplot.xlabel('Bike tracks built (km)')
    pyplot.ylabel('Cycling potential gained (unitless)')
    pyplot.show()

def hex_grid_pop(GLOBAL):
    MAP = GLOBAL['MAP']
    grid = Qf32(list(MAP['municipal_boundary'].get_lrbt('m'))+[GLOBAL['PHYSICAL']['cell_hex_m']])
    HexGrid = hex_grid(grid[0:4],grid[4])
    rk = [k for k in MAP.optional if k.startswith('residences')]
    if len(rk)!=1: Qer('Bad residences in hex_grid_pop().')
    rk = rk[0]
    hashgrid = HashGrid(GLOBAL['COMPUTE'], grid[0:4], grid[4])
    hashgrid.hash_points('residences', MAP[rk].xy())
    count = MAP[rk].optional['count']
    HexPop = Qf32(len(HexGrid),0)
    for n,Hex in enumerate(HexGrid):
        HexPop[n] = sum(count[i] for i in hashgrid.find_all_on_polygon('residences',Hex))
    Hex = Geometry(HexGrid,form='xy')
    Hex['pop'] = HexPop
    MAP['hex_grid_pop'] = Hex



def prepare_cells(MAP, NWK, HexGrid, COMPUTE):
    rk = [k for k in MAP.optional if k.startswith('residences')]
    if len(rk)!=1: Qer('Bad residences in prepare_cells().')
    rk = rk[0]
    wk = [k for k in MAP.optional if k.startswith('workplaces')]
    if len(wk)!=1: Qer('Bad workplaces in prepare_cells().')
    wk = wk[0]
    RT = NWK.terminals[rk]
    Rbike = QcF(Cells, [HexGrid])
    QcF(Rbike.make_hashgrid, [COMPUTE])
    QcF(Rbike.locate, [RT])
    WT = NWK.terminals[wk]
    Wbike = QcF(Cells, [HexGrid])
    Wbike.hashgrid = Rbike.hashgrid
    QcF(Wbike.locate, [WT])
    return RT, WT, Rbike, Wbike

def main_planner_ai(GLOBAL, VARS):
    PLA_FILES = GLOBAL['PATHS']['PLA_FILES']
    NWK = GLOBAL['NWK']
    TRs = NWK.TRs
    seg_selector_ai(GLOBAL)
    MAP = GLOBAL['MAP']
    Hex = MAP.get('hex_grid_pop')
    if not Hex:
        hex_grid_pop(GLOBAL)
        Hex = MAP['hex_grid_pop']
    RT, WT, Rbike, Wbike = prepare_cells(MAP, NWK, Hex.xy(), GLOBAL['COMPUTE'])
    params = [NWK, RT, WT, Rbike, Wbike, osmfilter.PROFILES,
              'SLOPE' in GLOBAL, GLOBAL['PHYSICAL']['min_trip_s']]
    FILE = PLA_FILES+'00'
    if os_path_isfile(FILE):
        with open(FILE, 'rb') as f: Scores, base = pickle.load(f)
    else:
        # baseline:
        R = QcF(RW_Study,params)
        Rw = R['Rw']
        V = Hex['pop']*R['Rs']/Rw
        base = sum(v for i,v in Qe(V) if Hex['pop'][i] and Rw[i])
        # individual thread improvement access:
        print('Anlyzing',len(TRs.threads),'candidate TR threads for improvement...')
        Scores = []
        for n,thread in Qe(TRs.threads):
            Sup = Qbool(len(TRs),0)
            for tr in thread: Sup[tr] = 1
            len_m = sum(TRs.len_m[tr] for tr in thread)
            TRs['Super'] = Sup
            Rbike.reset() # cell counting reset
            Wbike.reset() # "
            R = QcF(RW_Study,params)
            Rw = R['Rw']
            V = Hex['pop']*R['Rs']/Rw
            improv = base - sum(v for i,v in Qe(V) if Hex['pop'][i] and Rw[i])
            Scores.append([improv/len_m, n, len_m]) # must be mutable for next step
            print(len(Scores),'/',len(TRs.threads),'threads checked.\n')
        Scores.sort(reverse=True)
        with open(FILE, 'wb') as f: pickle.dump((Scores, base), f)
    # cumulative thread improvement access iterative algorithm:
    Curves = {}
    for iteration in range(1,GLOBAL['ALGORITHM']['iterations']):
        FILE = PLA_FILES+'0'*(iteration<10)+str(iteration)
        print('Planning iteration',iteration,'\n')
        if os_path_isfile(FILE):
            with open(FILE, 'rb') as f: Curves, Scores = pickle.load(f)
        else:
            Sup = Qbool(len(TRs),0)
            Curve = [(0.0,0.0)]
            for n in Qr(Scores):
                print('Building curve:',n+1,'/',len(Scores),'iter',iteration,'\n')
                for tr in TRs.threads[Scores[n][1]]: Sup[tr] = 1
                TRs['Super'] = Sup
                Rbike.reset() # cell counting reset
                Wbike.reset() # "
                R = QcF(RW_Study, params)
                Rw = R['Rw']
                V = Hex['pop']*R['Rs']/Rw
                M = base - sum(v for i,v in Qe(V) if Hex['pop'][i] and Rw[i])
                Curve.append((Scores[n][2],M))
            Curves[iteration] = Curve
            # update Scores:
            for n, d in Qe(np_diff([x[1] for x in Curve])):
                Scores[n][0] = d/Scores[n][2]
            Scores.sort(reverse=True)
            with open(FILE, 'wb') as f: pickle.dump((Curves, Scores), f)
    GLOBAL['STUDY']['rw_curves'] = Curves
    #GLOBAL['STUDY']['rw_scores'] = Scores
    # populate TRs['vis_PRIORITY'] with category values
    ten = sum(x[2] for x in Scores)/10
    cum_len_m = 0
    category = 7
    prior = Qu8(len(TRs), 0)
    for _, thread, len_m in Scores:
        for tr in TRs.threads[thread]:
            prior[tr] = category
        cum_len_m += len_m
        if cum_len_m>ten: category = 6
        if cum_len_m>ten*3: category = 5
    for n, lts in Qe(TRs['vis_LTS']):
        if lts == 1 or lts == 2: prior[n] = lts
    TRs['vis_PRIORITY'] = prior
    with open(GLOBAL['PATHS']['PRI_FILE'],'wb') as f:
        pickle.dump(prior, f)
    
        


# TO_DO:
# - maybe sort TRs in thread by connectivity.
# - also find TN endpoints of each thread, needed for graph colouring (optional)
def seg_selector_ai(GLOBAL):
    TRs = GLOBAL['NWK'].TRs
    endTN = TRs.endTN
    startTN = TRs.startTN
    TNs = GLOBAL['NWK'].TNs
    TRconn = TNs.TRconn
    ####neighbTNs = TNs.neighbTNs
    vPR = TRs['vis_PRIORITY'] * TRs['is_upgradeable']
    N = len(TRs)
    print(N, 'TRs,')
    print(sum(vPR>4), 'prioritized,') # TO_DO: make '4' a system variable
    TRsrem = set(np_arange(N)[vPR>4])
    TRspri = TRsrem.copy()
    threads = []
    while TRsrem:
        trhome = TRsrem.pop()
        thread = [trhome]
        for tn in (startTN[trhome], endTN[trhome]):
            tr = trhome
            while 1:
                c = set(TRconn[tn]) & TRspri
                c.remove(tr)
                if len(c) != 1: break
                prio = vPR[tr]
                tr = c.pop()
                s = set((startTN[tr], endTN[tr]))
                s.remove(tn)
                tn = s.pop()
                if tr in TRsrem and prio==vPR[tr]:
                    thread.append(tr)
                    TRsrem.remove(tr)
                else: break
        threads.append(thread)
    print(len(threads),'priority threads.')
    vt = Qu32(N,0)
    th = Qu32(N,0)
    for i,thread in enumerate(threads):
        v = i%24
        for tr in thread:
            vt[tr] = v
            th[tr] = i
    TRs['vis_thread'] = vt
    TRs['num_thread'] = th
    TRs.threads = LoL(threads,'uint32')


##def main_rw_residences_curve(GLOBAL, VARS):
##    MAP = GLOBAL['MAP']
##    NWK = GLOBAL['NWK']
##    TRs = NWK.TRs
##    COMPUTE = GLOBAL['COMPUTE']
##    if 1:
##        Seg = GLOBAL['STUDY']['Seg']
##        Seg = Seg.get('SxB_3D', Seg['SxB_2D'])
##        Seg = np_maximum(Seg[:,0],Seg[:,1]) ### priority set as highest of both directions, for now.
##        SegPri = list(zip(Seg, np_arange(len(TRs)), TRs.len_m))
##        SegPri.sort(reverse=True)
##    Hex = MAP.get('hex_grid_pop')
##    if not Hex:
##        hex_grid_pop(GLOBAL)
##        Hex = MAP['hex_grid_pop']
##    Sup = Qbool(len(TRs),0)
##    cum_len = 0.0
##    R = QcF(RW_Study,[NWK, rk, wk, osmfilter.PROFILES, Hex.xy(), COMPUTE])
##    Rw = R['Rw']
##    V = Hex['pop']*R['Rs']/Rw
##    base = sum(v for i,v in Qe(V) if Hex['pop'][i] and Rw[i])
##    Curve = [(0.0,0)]
##    N = sum(TRs['vis_PRIORITY']>4)### 7-3 
##    print('Anlyzing',N,'candidate TRs for improvement.')
##    for n in Qr(N):
##        Sup[SegPri[n][1]] = 1
##        cum_len += SegPri[n][2]
##        if cum_len > 1000.0 or n==N-1:
##            TRs['Super'] = Sup
##            R = QcF(RW_Study,[NWK, rk, wk, osmfilter.PROFILES, Hex.xy(), COMPUTE])
##            Rw = R['Rw']
##            V = Hex['pop']*R['Rs']/Rw
##            M = base-sum(v for i,v in Qe(V) if Hex['pop'][i] and Rw[i])
##            P = (cum_len,M)
##            Curve.append(P)
##            print(Curve)
##            cum_len = 0.0
##    del TRs['Super']
##    GLOBAL['STUDY']['seg_curve'] = Curve

##    rw_vis = Cells(HexGrid, hex_grid(grid[0:4],grid[4],forGPU=True))
##    v = Values(len(HexGrid))
##    sb = rw_studies.get('S*B_3D',rw_studies['S*B_2D']) 
##    v.s = sb['Rs']
##    v.w = sb['Rw']
##    rw_vis.values = v
##    QcF(rw_vis.make_graphics)
##    GLOBAL['STUDY'] = {'RW':rw_studies, 'RW_vis':rw_vis}
##    MAP['cells'] = rw_vis # mount non-Geo






def main_rw_study(GLOBAL, VARS):
    MAP = GLOBAL['MAP']
    NWK = GLOBAL['NWK']
    TRs = NWK.TRs
    RW_FILE = GLOBAL['PATHS']['RW_FILE']
    if os_path_isfile(RW_FILE):
        with open(RW_FILE, 'rb') as f: rw_studies = pickle.load(f)
        grid = rw_studies['hex_grid']
        HexGrid = hex_grid(grid[0:4],grid[4])
    else:
        grid = Qf32(list(MAP['municipal_boundary'].get_lrbt('m'))+[GLOBAL['PHYSICAL']['cell_hex_m']])
        HexGrid = hex_grid(grid[0:4],grid[4])
        rw_studies = {'hex_grid':grid}
        threeD = 'SLOPE' in GLOBAL
        for rk in [k for k in MAP.optional if k.startswith('residences')]:
            for wk in [k for k in MAP.optional if k.startswith('workplaces')]:
                for direction in [(rk,wk), (wk,rk)]:
                    A, B = direction
                    print('='*40)
                    print('Beginning Study...')
                    print(len(MAP[A]),'x',len(MAP[B]))
                    StudyName = A[10:]+'*'+B[10:]+'_'+'23'[threeD]+'D'
                    print(StudyName,'RW_Study')
                    RT, WT, Rbike, Wbike = prepare_cells(MAP, NWK, HexGrid, GLOBAL['COMPUTE']) 
                    rw_studies[StudyName] = RW_Study(NWK, RT, WT, Rbike, Wbike,
                        osmfilter.PROFILES, 'SLOPE' in GLOBAL, GLOBAL['PHYSICAL']['min_trip_s'])
        with open(RW_FILE, 'wb') as f: pickle.dump(rw_studies, f)
    print(rw_studies.keys())
    rw_vis = Cells(HexGrid, hex_grid(grid[0:4],grid[4],forGPU=True))
    v = Values(len(HexGrid))
    sb = rw_studies.get('S*B_3D')
    if sb is None: sb = rw_studies.get('S*B_2D')
    v.s = sb['Rs']
    v.w = sb['Rw']
    rw_vis.values = v
    QcF(rw_vis.make_graphics)
    GLOBAL['STUDY'] = {'RW':rw_studies, 'RW_vis':rw_vis}
    MAP['cells'] = rw_vis # mount non-Geo
            

    

def main_seg_study(GLOBAL, VARS):
    MAP = GLOBAL['MAP']
    NWK = GLOBAL['NWK']
    TRs = NWK.TRs
    COMPUTE = GLOBAL['COMPUTE']
    SEG_FILE = GLOBAL['PATHS']['SEG_FILE']
    seg_studies = {}
    if os_path_isfile(SEG_FILE):
        with open(SEG_FILE, 'rb') as file: seg_studies = ui_raw_read(file)
    else:
        for threeD in [False]+[True]*('SLOPE' in GLOBAL):
            for rk in [k for k in MAP.optional if k.startswith('residences')]:
                for wk in [k for k in MAP.optional if k.startswith('workplaces')]:
                    seg = []
                    print('='*40)
                    StudyName = rk[10:]+'x'+wk[10:]+'_'+'23'[threeD]+'D'
                    print('Beginning Seg Study',StudyName)
                    print(len(MAP[rk]),'x',len(MAP[wk]))
                    for A,B in [(rk,wk), (wk,rk)]:
                        seg.append(Seg_Study(NWK, NWK.terminals[A], NWK.terminals[B],
                                             osmfilter.PROFILES, threeD))
                    seg_studies[StudyName] = seg[0]+seg[1]
        with open(SEG_FILE, 'wb') as file: ui_raw_write(seg_studies, file)
    draw_seg_studies = {}
    MAX_L = 7
    LTS_HIGH = {d: (TRs[d+"wd_LTS"]!=1) & (TRs[d+"wd_LTS"]!=2) for d in 'fb'} 
    for k,v in seg_studies.items():
        V = {'f':v[:,0],'b':v[:,1]}
        V = {a:np_log10(b+10**-20) for a,b in V.items()}
        maxV = max(max(V['f']),max(V['b']))
        V = {a:Qu8(np_clip((b + (MAX_L+0.5 - maxV)).round(),0,MAX_L)) for a,b in V.items()}
        for a,b in V.items():
            draw_seg_studies['L'+a+'wd'+k] = b * LTS_HIGH[a]
    print(draw_seg_studies.keys())
    # Visible PRIORITY defaults to SxB:
    for d in 'fb':
        H = draw_seg_studies.get('L'+d+'wdSxB_3D',draw_seg_studies.get('L'+d+'wdSxB_2D'))
        H = H*(H>(MAX_L-3))
        L = TRs[d+'wd_LTS']
        H += L==1
        H += Qu8((L==2)*2)
        TRs[d+'wd_PRIORITY'] = H
    TRs['vis_PRIORITY'] = np_maximum(*[TRs[d+'wd_PRIORITY'] for d in 'fb'])
    GLOBAL['STUDY'] = {'Seg':seg_studies,'LSeg':draw_seg_studies}##'RW':rw_studies,
    



def Seg_Study(NWK, RT, WT, PROFILES, threeD=False):
    Profile = 'Work'
    print(Profile,'profile')
    print('1) Attaching R-W points...')# 1) Make a graph for this profile.
    NWK(PROFILES[Profile], threeD=threeD)
    G = NWK.graph[Profile]
    print('Running Seg_Study with',len(RT),'RTs,',len(WT),'WTs.')
    print(int(len(RT)*len(WT)/1e6),'M routes,',
          'estimated time %.2f minutes'%float(len(RT)*len(WT)/240e6)) #~4M routes/second
    print('2) Building Res anchors...') # 2) Add anchors for all residence points.
    RTN = QcF(G.build_anchors,[NWK.TRs, 0, RT]) 
    # RTN lists all TNs adjacent to a Residences.
    BLOCK_SIZE = int(SIM_SIZE/max(len(RT),len(WT)))
    Seg = Qf64((len(NWK.TRs),2),0)
    for b in Qr(ceil(len(WT)/BLOCK_SIZE)):
        # Group workplaces by blocks so that simulation fits reasonably into memory
        BLOCK = list(Qr(b*BLOCK_SIZE,min((b+1)*BLOCK_SIZE,len(WT))))
        print('='*16,'\nBLOCK size:',len(BLOCK))
        print('3) Building Wrk anchors...') # 3) Add anchors for subset of workplace points
        WTN = QcF(G.build_anchors,[NWK.TRs, 1, WT, BLOCK])
        # WTN lists all TNs adjacent to a Workplace.
        print(Profile)
        print('4) Routing network graph...') # 4) TN routing: Route all RTNs to selected WTNs
        QcF(G.build_mx,[True,WTN,True])
        print('5) Routing R-W...') # 5) Route all Res to all Wrk.
        QcF(G.all_to_all) 
        print('6) Tracing routes...') # 6) Count all segs with multiplicity.
        Seg += QcF(NWK.route_centrality, [G,RT.pop,WT.pop])
        print('DONE!')
    print('Study DONE.')
    return Seg # (len(TR) x 2) matrix
    


def RW_Study(NWK, RT, WT, Rbike, Wbike, PROFILES, threeD, min_s):
    for Profile in 'Work Super'.split():
        #print('---',Profile,'profile ---')
        #print('1)Making Graph...')# 1) Make a graph for this profile.
        # min_s is set to 0 here, and short routes are eliminated later:
        QcFs(NWK, {'profile':PROFILES[Profile], 'min_s':0.0, 'threeD':threeD})
        #print('2) Building Res anchors...') # 2) Add anchors for all residence points.
        RTN = QcFs(NWK.graph[Profile].build_anchors, [NWK.TRs, 0, RT])
        # RTN lists all TNs adjacent to a Residence.
    BLOCK_SIZE = int(SIM_SIZE/max(len(RT),len(WT)))
    for b in Qr(ceil(len(WT)/BLOCK_SIZE)):
        # Group workplaces by blocks so that simulation reasonably fits into memory
        BLOCK = list(Qr(b*BLOCK_SIZE,min((b+1)*BLOCK_SIZE,len(WT))))
        #print('='*16);print('BLOCK size:',len(BLOCK))
        Travel_s = {}
        for Profile in ['Work', 'Super']:
            G = NWK.graph[Profile]
            #print('---',Profile,'profile ---')
            #print('3) Building Wrk anchors...') # 3) Add anchors for subset of workplace points
            WTN = QcFs(G.build_anchors, [NWK.TRs, 1, WT, BLOCK])
            # WTN lists all TNs adjacent to a Workplace.
            #print('4) Routing network graph...') # 4) TN routing: Route all RTNs to selected WTNs
            QcFs(G.build_mx,[True,WTN,False])
            #print('5) Routing R-W...') # 5) Route all Res to all Wrk.
            Travel_s[Profile] = QcFs(G.all_to_all)
            #print('DONE!')
        BikeRatios = BikeabilityMetric(Travel_s['Work'], Travel_s['Super'],
                                       PROFILES['Work']['max_time'], min_s) 
        Weights = QcFs(np_outer, [RT.pop, [WT.pop[b] for b in BLOCK]]) # weight of each trip = Res_weight * Wrk_weight
        # Count bikeability metric average for each residence:
        R = [len(RT),None]
        Rv = Qf32(*R)
        Rw = Qf32(*R)
        ResVals = finite_averages(Rv, Rw, len(RT), BikeRatios, Weights)
        QcFs(Rbike.set, [Rv, Rw])
        # Count bikeability metric average for each workplace:
        W = [len(BLOCK), None]
        Wv = Qf32(*W)
        Ww = Qf32(*W)
        WrkVals = finite_averages(Wv, Ww, len(BLOCK),
                                  BikeRatios.transpose(), Weights.transpose())
        QcFs(Wbike.set, [Wv, Ww, BLOCK])
    return {'Rs':Rbike.values.s, 'Rw':Rbike.values.w,
            'Ws':Wbike.values.s, 'Ww':Wbike.values.w}

@njit("float32[:]("+"float32[:],"*2+"uint32,float32[:,:],float64[:,:])")
def finite_averages(outV, outW, M, Values, Weights):
    for m in range(M):
        SW = 0
        V = 0
        for n in range(len(Weights[m,:])):
            if np_isfinite(Values[m,n]):
                SW += Weights[m,n]
                V += Weights[m,n]*Values[m,n]
        outV[m] = V/SW if SW else np_nan
        outW[m] = SW
    return outV

def BikeabilityMetric(Work, Super, max_s, min_s):
    value = (Super>min_s) & (Super<max_s) & (Work<max_s)
    out = Qf32(Work.shape,float('Nan'))
    out[value] = Work[value]/Super[value] - 1.0
    return out
        
















##def __DEPRECATED_OD_Study(OD, NW, Regions):
##    print('Attaching O-D points...')
##    NW.Residences(OD['O'], [1]*len(OD['N']))
##    NW.Workplaces(OD['D'], OD['N'])
##    Ohist = Cells(Regions)
##    Dhist = Cells(Regions)
##    for n, P in enumerate(NW.Res.xy()): Ohist(*P, NW.Res.pop[n], 1)
##    for n, P in enumerate(NW.Wrk.xy()): Dhist(*P, NW.Wrk.pop[n], 1)
##    ####
##    #NW.hashgrid.hash_polygons('OD Regions',Regions)
##    Travel_s = {}
##    RODN = list(Qr(OD['N']))
##    for Profile in 'Work Super'.split():
##        print('Building graph for profile:', Profile)
##        NW(PROFILES[Profile])
##        print('Building attachments...')
##        RTN=NW.graph[Profile].build_anchors(NW.TRs, 0, NW.Res)
##        WTN=NW.graph[Profile].build_anchors(NW.TRs, 1, NW.Wrk)
##        print('Routing network graph...')
##        NW.graph[Profile].build_mx(True,WTN)
##        print('... routed graph.')
##        print(NOW())
##        print('Routing O-D points!')
##        Travel_s[Profile] = NW.graph[Profile].one_to_one(RODN)
##        print(NOW())
##        print('DONE!')
##        del NW.graph[Profile]
##    Sel = Selection(len(OD['N']))
##    Sel['Flight'] = [dist(NW.Res[n],NW.Wrk[n])/(18/3.6) for n in RODN]
##    Sel['Work'] = Travel_s['Work']
##    Sel['Super'] = Travel_s['Super']
##    Sel['Weight'] = OD['N']
##    Sel['index'] = RODN
##    Sel([s<1800.0 for s in Travel_s['Work']])
##    WORK = Sel['Work']
##    SUPER = Sel['Super']
##    WEIGHT = Sel['Weight']
##    Obike = Cells(Regions)
##    XY = NW.Res.xy()
##    for n,ix in enumerate(Sel['index']):
##        Obike(*XY[ix], WORK[n]/SUPER[n]-1, WEIGHT[n])
##    Dbike = Cells(Regions)
##    XY = NW.Wrk.xy()
##    for n,ix in enumerate(Sel['index']):
##        Dbike(*XY[ix], WORK[n]/SUPER[n]-1, WEIGHT[n])
##    # Bike Mode Share
####    BMS = Cells(Regions)
####    XY = NW.Res.xy()
####    for n, B in enumerate(OD['B']):
####        BMS(*XY[n], B, OD['N'][n])
##    return locals()







class Cells:
    def set_canvas(self, canvas):
        self.canvas = canvas
        self.reshape()
    def set_alpha(self,alpha):
        if alpha != self.alpha:
            self.alpha = alpha
            self.program['u_alpha'] = self.alpha
    def draw(self, alpha=None, max_density_m2=None):
        if alpha is not None: self.set_alpha(alpha)
        if max_density_m2 is not None: self.set_max_density(max_density_m2)
        self.program.draw('triangles',self.index_buffer)
    def reshape(self):
        self.program['u_pan'] = self.canvas.pan
        self.program['u_scale'] = [self.canvas.scale/self.canvas.size[0],
                                   self.canvas.scale/self.canvas.size[1]]
    def make_graphics(self):
        self.program = Program(MAGMA_SHADER+POLYGON_VERT_SHADER,POLYGON_FRAG_SHADER)
        M = 1.0-np_nan_to_num(Qf32(self.values.means()),nan=-10.0)
        self.alpha = None
        self.set_alpha(1.0)
        #T = [sum(triangulate_contour(C),tuple()) for C in self.polygons]
        #print(get_depth(T))
        #TA = Qf32(sum((tuple(t) for t in T),tuple()))
        #print(TA.shape)
        self.program['a_position'] = VertexBuffer(self.positionVB)
        IA = np_arange(self.positionVB.shape[0],dtype='uint32')
        #print(IA.shape)
        self.index_buffer = IndexBuffer(IA)
        self.program['u_max_val'] = 1.0
        self.program['a_val'] = VertexBuffer(np_repeat(Qf32(M),18)) # each hex is made up of 6 triangles.
        self.program['a_alpha'] = VertexBuffer(np_repeat(Qf32((M<10.0)*0.8),18)) # set NaN to transparent
    def __len__(self): return self.len
##    def means(self):
##        V = self.values.means()
##        sel = Selection(self.len)
##        sel['polygons'] = self.polygons
##        sel['values'] = V
##        sel([not np_isnan(v) for v in V])
##        return {k:sel[k] for k in sel.items}        
##    def sums(self):
##        V = self.values.sums()
##        sel = Selection(self.len)
##        sel['polygons'] = self.polygons
##        sel['values'] = V
##        sel([not np_isnan(v) for v in V])
##        return {k:sel[k] for k in sel.items}
##    def shpdict(self, method, field):
##        out = {}
##        out['polygons'] = self.polygons
##        out['field_names'] = [field]
##        out['records'] = {field:method()['values']}
##        return out
    def __init__(self, Polygons, vertex_buffer=None):
        self.polygons = Polygons
        self.positionVB = vertex_buffer
        self.len = len(Polygons)
        self.values = Values(self.len)
        self.lrbt = lrbt(lol_to_tuple(Polygons))
    def make_hashgrid(self, COMPUTE):
        self.hashgrid = HashGrid(COMPUTE, self.lrbt)
        self.hashgrid.hash_polygons('Polygons', self.polygons)
    def set(self, v, w, nn=None):
        ixs = self.indices
        if nn is None: nn = np_arange(len(v))
        for i,j in Qe(nn):
            cell = ixs[j]
            if cell is not None: self.values(v[i],w[i],cell)
    def reset(self): self.values = Values(self.len)
    def locate(self, G2):
        self.indices = [self.hashgrid.find_one_polygon(xy,'Polygons') for xy in G2.xy()] # ! may contain None's
