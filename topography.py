from utils import *
#from png import from_array as png_from_array 
from geometry2 import lrbt_and_lrbt, make_box
from earth import Geometry
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from scipy.interpolate import RectBivariateSpline
from qcommands import *







TOPOG_VERT_SHADER = """
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;
uniform float u_max;
uniform float u_min;
varying float v_max;
varying float v_min;
uniform float u_alpha;
varying float v_alpha;
uniform vec2 u_pan;
uniform vec2 u_scale;

void main() {
    gl_Position = vec4(u_scale * (a_position + u_pan), 0.0, 1.0);
    v_texcoord = a_texcoord;
    v_alpha = u_alpha;
    v_max = u_max;
    v_min = u_min;
}
"""

TOPOG_FRAG_SHADER = """
varying float v_max;
varying float v_min;
varying float v_alpha;
uniform sampler2D u_heightmap;
varying vec2 v_texcoord;

void main()
{
    float c = texture2D(u_heightmap, v_texcoord)[0];
    if(c<0.0) c = 0.0;
    //float c = h/10.0;//(h-v_min)/(v_max-v_min);
    c = c*8.0; // increase brightness at lower end.
    if(c<0.33333) {gl_FragColor = vec4(0,0,0.5+c*1.5,v_alpha);}
    else {gl_FragColor = vec4((c-0.33333)*1.5,(c-0.33333)*1.5,1.0,v_alpha);}
}

"""

class TopogRect:
    def __init__(self, lons=None, lats=None, data=None):
        if type(data) is type(None): self.empty = True;return None
        print('One Topog',data.shape)
        self.data = Qf32(data) # data is int16 in CDEM
        if np_prod(data.shape)==0: self.empty = True;return None
        self.empty = False
        self.max = self.data.max()
        if self.max <-500.0:
            self.max = None
            self.min = None
            self.empty = True
            return None
        else:
            self.min = self.data.min(initial = self.max, # if map is full of -32767, it will default to empty.
                                     where=self.data>-500.0) # lowest point on Earth surface is -420m
        #print(self.min,self.max)
        # Keep "empty" tiles for now (e.g. Ottawa ON), complicates code:
        #if self.min==self.max: self.empty = True;return None # this will happen if all map is -32767
        self.centre = None
        self.h, self.w = self.data.shape #Note order
        self.data_lrbt_deg = (min(lons),max(lons),min(lats),max(lats))
        self.data_shBox = sPolygon(make_box(lrbt=self.data_lrbt_deg))
        self.pixel_w = (self.data_lrbt_deg[1]-self.data_lrbt_deg[0])/(self.w-1)
        self.pixel_h = (self.data_lrbt_deg[3]-self.data_lrbt_deg[2])/(self.h-1)
        self.draw_lrbt_deg = (self.data_lrbt_deg[0]-self.pixel_w/2,
                              self.data_lrbt_deg[1]+self.pixel_w/2,
                              self.data_lrbt_deg[2]-self.pixel_w/2,
                              self.data_lrbt_deg[3]+self.pixel_w/2)
        self.draw_min = self.min
        self.draw_max = self.max
        self.program = Program(TOPOG_VERT_SHADER, TOPOG_FRAG_SHADER)
        self.program['u_min'] = self.draw_min
        self.program['u_max'] = self.draw_max
        self.program['u_alpha'] = 1.0
        self.program['u_pan'] = (0.0,0.0)
        self.program['u_scale'] = (1.0,1.0)
        ### interpolation 2D sampling:
        ix = np_arange(0, self.w, 1) 
        iy = np_arange(self.h-1, -1, -1) # y-flip
        ixx, iyy = np_meshgrid(ix, iy)
        z = self.data[iyy,ixx] # #transpose
        x = np_arange(0,self.w,1)*self.pixel_w + self.data_lrbt_deg[0]
        y = np_arange(0,self.h,1)*self.pixel_h + self.data_lrbt_deg[2]
        self.interp2d = RectBivariateSpline(x, y, z)#, fill_value=float('Nan')
    def sample_one(self, lon, lat):
        return self.interp2d(lon,lat)[0]
    def __sample_many(self, lons, lats): # to do: fix interp2d output
        N = len(lons)
        return Qf32([self.interp2d(lons[n],lats[n]) for n in range(N)])
    def crop(self, lrbt_deg):
        lrbt = lrbt_and_lrbt(self.data_lrbt_deg, lrbt_deg)
        if not lrbt_deg: return TopogRect()# empty object
        l,r,b,t = lrbt
        pl = int((l - self.data_lrbt_deg[0])/self.pixel_w + 0.5)
        pr = int((r - self.data_lrbt_deg[0])/self.pixel_w + 0.5) #sic[0]
        pt = int((self.data_lrbt_deg[3] - t)/self.pixel_h + 0.5) # note axis reversal
        pb = int((self.data_lrbt_deg[3] - b)/self.pixel_h + 0.5) # sic[3], note axis reversal
        new_data = self.data[pt:pb+1][pl:pr+1]
        new_lrbt = (self.data_lrbt_deg[0]+self.pixel_w*(pl-0.5),
                    self.data_lrbt_deg[0]+self.pixel_w*(pr+0.5),#sic[0]
                    self.data_lrbt_deg[3]-self.pixel_h*(pt-0.5),
                    self.data_lrbt_deg[3]-self.pixel_h*(pb+0.5))#sic[3]
        return TopogRect(new_lrbt[0:2], new_lrbt[2:4], new_data)
    def __call__(self, centre):
        if not self.empty:
            self.centre = centre
            G = Geometry(make_box(lrbt=self.draw_lrbt_deg))
            self.draw_trapez_m = G(centre).get_trapez_m()
        return self
    def set_draw_min_max(self, draw_min, draw_max):
        if not self.empty:
            if draw_min != self.draw_min:
                self.draw_min = draw_min
                self.program['u_min'] = draw_min
                #print('New draw_min',draw_min)
            if draw_max != self.draw_max:
                self.draw_max = draw_max
                self.program['u_max'] = draw_max
                #print('New draw_max',draw_max)
    def load_heightmap(self):
        if not self.empty:
            gray = ((self.data-self.draw_min)/(self.draw_max-self.draw_min))
            #print(gray.shape)
            #alpha = (gray-gray)+255
            #print(gray.min(),gray.max())
            self.program['u_heightmap'] = np_fliplr(gray.transpose())  ###np_stack([gray,gray,gray,alpha])#self.data
            self.program["u_heightmap"].interpolation = 'linear'
    def set_frame(self, frame):
        if not self.empty:
            #self.program['a_position'] = Qf32(frame)
            #self.program['a_texcoord'] = Qf32([(0, 0),(1, 0), (1, 1),(0, 1)])
            self.program['a_position'] = Qf32([frame[n] for n in [1,0,2,3]])
            self.program['a_texcoord'] = Qf32([(0, 1),(0, 0), (1, 1),(1, 0)])
            #print('set_frame')
    def draw(self):
        if not self.empty:
            self.program.draw('triangle_strip')


class Topography:
    def get_altitude_tile(self, lon, lat):
        if not self.tiles: return None
        out = None
        for t,tile in enumerate(self.tiles):
            h = tile[1].sample_one([lon],[lat])
            if not isnan(h):
                out = (h,t)
                break
        return out
    def __init__(self, g3xyBoundary=None):
        self.tiles = []
        self.centre = None
        self.programs = []
        self.canvas = None
        self.shBoundary = MultiPolygon(sPolygon(C) for C in g3xyBoundary) if g3xyBoundary else None
        self.min = None
        self.max = None
        self.empty = True
    def place_on_map(self):
        if not self.centre:
            print('Error - Topography must be mounted before it can be placed on map!')
            return False
        if not self.empty:
            for _,tr in self.tiles:
                if not tr.empty:
                    tr.set_frame(tr.draw_trapez_m)
                    tr.load_heightmap()
    def set_min_max(self):
        if not self.empty:
            m = [tr.min for _,tr in self.tiles if not tr.empty and tr.min is not None]
            self.min = min(m) if m else None
            M = [tr.max for _,tr in self.tiles if not tr.empty and tr.max is not None]
            self.max = max(M) if M else None
            for _,tr in self.tiles: tr.set_draw_min_max(self.min, self.max)
        else: self.min = None; self.max = None
    def add(self, name, data, lons, lats):
        tr = TopogRect(lons, lats, data)
        if tr:
            self.tiles.append([name,tr])
            if self.centre: tr(self.centre)
            if not tr.empty: self.empty = False
        self.set_min_max()
    def crop_all(self):
        for n in Qr(self.tiles): self.crop(n)
    def crop(self, ixTile=-1):
        if self.shBoundary:
            tr = self.tiles[ixTile][1]
            shI = tr.data_shBox.intersection(self.shBoundary)
            if shI.area:
                l,b,r,t = shI.bounds
                self.tiles[ixTile][1] = tr.crop((l,r,b,t))
            else:
                tr.data = None
                tr.empty = True
            if self.centre: tr(self.centre)
        self.empty = not bool(sum(not tr.empty for _,tr in self.tiles))
        self.set_min_max()
    def __call__(self, centre):
        self.centre = centre
        for _,tr in self.tiles: tr(centre)
        return self
    def draw(self):
        if not self.canvas:
            print('Error - Topography must have a canvas before it can draw!')
            return False
        for _,tr in self.tiles:
            if not tr.empty: tr.draw()
    def set_canvas(self, canvas):
        self.canvas = canvas
        self.reshape()
    def reshape(self):# For drawing on interactive map
        if self.centre:
            for _,tr in self.tiles:
                if not tr.empty:
                    tr.program['u_pan'] = self.canvas.pan
                    tr.program['u_scale'] = [self.canvas.scale/self.canvas.size[0],
                                             self.canvas.scale/self.canvas.size[1]]





##if 0: # Download all Surface files
##    import urllib.request, time
##    url = "http://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdsm_mnsc/082/"
##    file_format = "082E%s_cdsm_final.zip"
##    for n in range(1,17):
##        s = str(n)
##        if len(s) == 1: s = '0'+s
##        file = file_format%s
##        print('Downloading:', file)
##        urllib.request.urlretrieve(url+file,'data/BC/Kelowna/'+file)
##
##def read_tif(file):
##    return np_array(PIL_Image.open(file),'float32')
##    
##def write_png(file, data):
##    X = data
##    m = X.min()
##    M = X.max()
##    Y = ((X-m)/(M-m)*255.9999).astype('uint8')
##    png_from_array(Y,'L').save(file)
##    os_system("outputs/elevation.png")
##
##PATH = "data/Canada/Topography - Kelowna/"
##if 1:
##    A = read_tif(PATH+"cdem_dem_082E.tif")
##    #A = A.clip(min=0) # Fix -32767 values near bottom
##    #A = A[:3000,:3000]
##    #write_png("outputs/elevation.png",A)
##if 0:
##    PATTERN = '12,13,14,15,11,10,9,8,4,5,6,7,3,2,1,0'.split(',')
##    Y = {}
##    for i,n in enumerate(PATTERN):
##        s = str(int(n)+1)
##        if len(s) == 1: s = '0'+s
##        E = read_tif(PATH+"082E%s_cdsm_final_e.tif"%s)
##        W = read_tif(PATH+"082E%s_cdsm_final_w.tif"%s)
##        Y[i] = np_hstack((W,E))
##    Z = {}
##    for m in range(4):
##        Z[m] = np_hstack(tuple(Y[n] for n in range(m*4,4+m*4)))
##    X = np_vstack(tuple(Z[n] for n in range(4)))
##    #X = X[:3000,:3000]
##    write_png("outputs/surface.png",X)
##if 0:
##    write_png("outputs/S-A.png",np_clip(X-A,a_min=0,a_max=10**10))
##    write_png("outputs/A-S.png",np_clip(A-X,a_min=0,a_max=10**10))
##    

