from utils import *
from qcommands import *
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from geometry2 import contours_to_solids, triangulate_contour
from earth import Geometry
from shaders import *

# offline function, need not be efficient.
# outputs must be json-compatible
def pop_geo_remake_geometry(POP_GEO):
    out = {}
    out['parcels'] = [contours_to_solids(P) for P in POP_GEO['parcels']] # remove holes, make into solids only: G4->G4
    out['areas'] = [[sPolygon(c).area for c in mp] for mp in out['parcels']]
    print('Triangulating population parcels...',end='')
    ### FAST_BAD should be false - make a better triangulation function.
    out['triangles'] = [sum((sum(triangulate_contour(c,fast_bad=True),tuple()) for c in mp),tuple()) for mp in out['parcels']] # G4->G3
    print('done!')
    return out


class PopDens:
    def give_G3(self):
        G4 = self.parcels.xy()
        outpops = []
        outG3 = []
        for n,g in enumerate(G4):
            if len(g) == 1:
                outpops.append(self.pops[n])
                outG3.append(g[0])
            else:
                outpops += [self.pops[n]*self.partial_areas[n][k]/self.areas[n] for k in Qr(g)]
                outG3 += g
        out = Geometry(outG3, form='xy')
        out.pop = outpops
        return out
    def __init__(self, POP_GEO, pops, max_density_m2 = 0.005):
        self.parcels = Geometry(POP_GEO['parcels'], form='xy')
        self.pops = Qf32(pops)
        self.partial_areas = LoLf32(POP_GEO['areas'])
        self.areas = Qf32([sum(a) for a in POP_GEO['areas']]) #L(L(F))->L(F)
        self.density_m2 = self.pops/self.areas
        # visuals:
        self.program = Program(MAGMA_SHADER+POLYGON_VERT_SHADER,POLYGON_FRAG_SHADER)
        self.max_density_m2 = None
        self.set_max_density(max_density_m2)
        triangles = POP_GEO['triangles']
        TA = Qf32(sum((tuple(t) for t in triangles),tuple()))
        print('TA.shape=',TA.shape)
        self.program['a_position'] = VertexBuffer(TA)
        lens = tuple(len(v) for v in triangles)
        cs_lens = np_cumsum(lens)
        s = cs_lens[-1]
        #print('s=',s)
        self.index_buffer = IndexBuffer(np_arange(s,dtype='uint32'))
        v_density = Qf32(s,None)
        self.vertex_len = s
        for n in Qr(triangles):
            if n == 0: a=0; b = cs_lens[0]
            else: a = cs_lens[n-1]; b = cs_lens[n]
            v_density[a:b] = self.density_m2[n]
        self.program['a_val'] = VertexBuffer(v_density)
        self.alpha = None
        self.set_alpha(1.0)
    def set_max_density(self, max_density_m2):
        if self.max_density_m2 != max_density_m2:
            self.max_density_m2 = max_density_m2
            self.program['u_max_val'] = self.max_density_m2
    def set_canvas(self, canvas):
        self.canvas = canvas
        self.reshape()
    def set_alpha(self,alpha):
        if alpha != self.alpha:
            self.alpha = alpha
            self.program['a_alpha'] = Qf32(self.vertex_len, alpha)
    def draw(self, alpha=None, max_density_m2=None):
        if alpha is not None: self.set_alpha(alpha)
        if max_density_m2 is not None: self.set_max_density(max_density_m2)
        self.program.draw('triangles',self.index_buffer)
    def reshape(self):
        self.program['u_pan'] = self.canvas.pan
        self.program['u_scale'] = [self.canvas.scale/self.canvas.size[0],
                                   self.canvas.scale/self.canvas.size[1]]
