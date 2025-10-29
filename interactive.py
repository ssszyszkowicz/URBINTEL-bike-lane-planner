# to do: break up this module and put into subfolder(?)
# Main map engine, keys and layers
# GUI buttons, highlight, etc.
# Roads objects.
# ? Low-level graphic primitives and shaders
#### TO_DO: Fix, maybe: Still crashes when passed empty layer?
# Always remaking the buttons makes it impossible to turn them on!

# http://jcgt.org/published/0002/02/08/paper.pdf - interesting thick line shaders (arrows, dashes, etc.)
from time import strftime as time_strftime, time_ns as time_time_ns
from osmfilter import WAY_TYPES
from drawfont import *
from utils import *
from qcommands import *
from geometry2 import HashGrid, make_box, dist, Frame
from colorsys import hsv_to_rgb, rgb_to_hsv
from earth import Geometry
from numba import njit
from vispy.io import imsave
from vispy.gloo.util import _screenshot
from vispy.gloo import Program, VertexBuffer, IndexBuffer, clear, set_viewport, set_state, Texture2D, RenderBuffer
from vispy.app import use_app as app_use_app, Canvas as app_Canvas, Timer as app_Timer
import vispy.app.backends._pyglet # needed for pyinstaller
app_use_app('pyglet') # needed for pyinstaller
from pyglet.window import key as pyglet_window_key
from OpenGL.GL import glEnable, glDisable, glFenceSync, glDeleteSync, glViewport
from OpenGL.GL import GL_MULTISAMPLE, GL_CULL_FACE, GL_SYNC_GPU_COMMANDS_COMPLETE, GL_SYNC_STATUS, GL_SIGNALED, GL_UNSIGNALED
from OpenGL.GL.ARB.sync import glGetSync
from OpenGL.GLUT import glutInit, glutInitDisplayMode, GLUT_SINGLE, GLUT_RGB, GLUT_MULTISAMPLE
#import OpenGL_accelerate
##from OpenGL import *
##from OpenGL.GL import *
from scipy.ndimage import gaussian_filter
from webbrowser import open as webbrowser_open
from fileformats import write_shp_from_dict
import auto_odt
from os import mkdir as os_mkdir
from os.path import exists as os_path_exists
try: from imageio.v2 import imread as imageio_imread
except: from imageio import imread as imageio_imread
from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw
from gGlobals import URBINTEL_RGB, URBINTEL_BACK_RGB
from shaders import *
# Summary:
# class:
# ThinLines
# ThickLines, thick_lines_helper(@njit)
# BigPoints
# Box
# Button
# GeometryOverlay
# HUDdisplay, MAPdisplay
# HeatMap
# RoadDisplay
###############
# class Canvas
# 




PUBLIC_NAMES  = {'vis_PRIORITY':'Priority',
                 'inst_slope':'Slope',
                 'vis_LTS':'Level of Traffic Stress',
                 'S*B_3D|B*S_3D':'Centrality',
                 'S*B_2D|B*S_2D':'Centrality',
                 '!S*B_3D|B*S_3D':'Prioritize',
                 '!S*B_2D|B*S_2D':'Prioritize',
                 'S*B':'AM Commute', '!S*B':'AM Prioritize',
                 'B*S':'PM Commute', '!B*S':'PM Prioritize',
                 'S*B|B*S':'Both Commute', '!S*B|B*S':'Both Prioritize',
                 'S*B-B*S':'Diff Commute', '!S*B-B*S':'Diff Prioritize'}
def public_name(name):
    return PUBLIC_NAMES.get(name,name)



POINT_TEXTURE_FILE = 'point.png' #Amir: we'll use point texture

def AntiAliasing(inputTexture,outputFbo,image_name,shape):
    #logic : primary object draw --> high res x4 fbo texture --> project that texture to quad with original resolution with linear interpolation
    vertices = Qf32([[1.0,  1.0],  [1.0, -1.0],  [-1.0, -1.0],  [-1.0,  1.0]]) #quad
    indices = Qu32([0, 1, 2, 3]) #triangle_fan indices, can be normal triangles also
    aaProgram = Program(ANTI_ALIASING_VERT_SHADER, ANTI_ALIASING_FRAG_SHADER)
    aaProgram['a_position'] = VertexBuffer(vertices)
    # aaProgram = IndexBuffer(indices)
    inputTexture.interpolation = 'linear'
    aaProgram['u_texture1'] = inputTexture
    with outputFbo:
        glViewport(0,0,shape[0],shape[1])
        aaProgram.draw(mode='triangle_fan',indices= IndexBuffer(indices))
        image = _screenshot((0,0,shape[0],shape[1]))
    imsave(image_name,image)

##########################################################
class ThinLines:
    def __init__(self,canvas):
        self.canvas = canvas
        program = Program(LINE_VERT_SHADER, LINE_FRAG_SHADER)
        program['u_alpha'] = 1.0 # throw away
        #Amir : the thin line program geometry was change to 'lines' instead of line_strip. in a line_strip all lines are joined , thus it adds excesses geomtery. 'lines' is made of seperate lines
        self.programs = [[program, 'lines', None]]
    def set_pos(self, x_y_lens):
        if x_y_lens is not None:
            program = self.programs[0][0]
            XY = np_stack([Qf32(x_y_lens[d]) for d in [0,1]],axis=1)
            program['a_position'] = VertexBuffer(XY)
            totalIndices = 0
            #Amir : this for loop calculates the total amount of indices required. each polyline of size l has l-1 line. each line requires 2 indices
            for n,l in enumerate(x_y_lens[2]):
                totalIndices+= (l-1)*2
            #Amir : the index buffer array is alloacted here
            indices = Qu32(totalIndices, None)
            #Amir : the index count is calculated increamentally
            indexCount = 0
            start = 0
            for n,l in enumerate(x_y_lens[2]):
                for m in Qr(l-1): 
                    indices[indexCount] = start + m # each poly line has a start index. m is the current position within the polyline
                    indices[indexCount+1] = start + m+1 # m+1 is the next part of the polyline
                    indexCount += 2
                start +=l # at the end of each polyline we add the polyline size to the start index , so it now points to the beginning of the next polyline
            self.programs[0][2] = IndexBuffer(indices)


@njit("uint8(uint32[:],float32[:,:],float32[:,:],uint32[:],"+
      "float32[:,:],uint32[:],float32)")
def _JIT_thick_line_helper(resize, corners, textCoords, indexes, #outputs
                       xy, lens, width): # inputs
    corners_len = 0
    indexes_len = 0
    ix = 0
    #Amir : these following lines are pre made vectors used later in the for loop
    v1 = np_array([width/2,width/2], np_float32) # can't Q due to njit.
    v2 = np_array([width/2,-(width/2)], np_float32)
    upTexCoord = np_array([0.5,0], np_float32)
    downTexCoord = np_array([0.5,1], np_float32)
    leftUpTexCoord = np_array([0,0], np_float32)
    leftDownTexCoord = np_array([0,1], np_float32)
    RightUpTexCoord = np_array([1,0], np_float32)
    RightDownTexCoord = np_array([1,1], np_float32)
    for i,n in enumerate(lens):
        for m in range(ix, ix+n-1):
            p = xy[m]
            q = xy[m+1]
            d = q-p
            l = (d[0]**2+d[1]**2)**0.5
            if l:
                u = d/l
                v = np_array([u[1]*width/2,-u[0]*width/2], np_float32)
                corners[corners_len] = p+v # numpy arrays can be added as vectors
                corners[corners_len+1] = q+v
                corners[corners_len+2] = q-v
                corners[corners_len+3] = p-v
                textCoords[corners_len] = downTexCoord # numpy arrays can be added as vectors
                textCoords[corners_len+1] = downTexCoord
                textCoords[corners_len+2] = upTexCoord       
                textCoords[corners_len+3] = upTexCoord
                corners_len += 4 
                corners[corners_len+1] = p-v1 #Amir: add the points as triangles
                corners[corners_len+2] = p+v2 
                corners[corners_len+3] = p+v1
                corners[corners_len] = p-v2 # numpy arrays can be added as vectors
                textCoords[corners_len] = leftUpTexCoord 
                textCoords[corners_len+1] = RightUpTexCoord
                textCoords[corners_len+2] = RightDownTexCoord
                textCoords[corners_len+3] = leftDownTexCoord
                corners_len += 4
                if m == ix+n-2: #Amir : add the last point as well
                    corners[corners_len+1] = q-v1 #Amir: add the points as triangles
                    corners[corners_len+2] = q+v2 
                    corners[corners_len+3] = q+v1
                    corners[corners_len] = q-v2 # numpy arrays can be added as vectors
                    textCoords[corners_len] = leftUpTexCoord
                    textCoords[corners_len+1] = RightUpTexCoord
                    textCoords[corners_len+2] = RightDownTexCoord
                    textCoords[corners_len+3] = leftDownTexCoord
                    corners_len += 4
        ix += n
    for n in range(corners_len//4):
        n6 = 6*n
        n4 = 4*n
        indexes[n6] = n4
        indexes[n6+1] = n4+1
        indexes[n6+2] = n4+2
        indexes[n6+3] = n4
        indexes[n6+4] = n4+2
        indexes[n6+5] = n4+3
        indexes_len += 6
    resize[0] = corners_len
    resize[1] = indexes_len
    return 0






















class ThickLines:
    def __init__(self, canvas):
        self.canvas = canvas
        program = Program(THICK_LINE_VERT_SHADER, THICK_LINE_FRAG_SHADER)
        program['u_alpha'] = 1.0
        program['u_texture1'] = canvas.point_texture #Amir
        self.programs = [[program, 'triangles',None]]#Amir : no need for point program anymore 
    def set_pos(self, x_y_lens_width):
        if x_y_lens_width is not None:
            XY = np_stack([Qf32(x_y_lens_width[d]) for d in [0,1]],axis=1)
            LENS = Qu32(x_y_lens_width[2])
            # Amir:
            LEN = XY.shape[0]
            corners = Qf32((8*LEN,2),None)
            texCoords = Qf32((8*LEN,2),None)
            indexes = Qu32(12*LEN,None)
            resize = Qu32(2,None)
            _JIT_thick_line_helper(resize, corners, texCoords, indexes,
                               XY, LENS, np_float32(x_y_lens_width[3]))
            # This is buggy and apparently useless:
            ###corners = np_resize(corners, (resize[0],2))
            ###texCoords = np_resize(texCoords, (resize[0],2))
            ###indexes = np_resize(indexes, (resize[1],))
            self.programs[0][0]['a_position'] = VertexBuffer(corners) 
            self.programs[0][0]['a_texcoord'] = VertexBuffer(texCoords) # Amir : set the texture coords for the program
            self.programs[0][2] = IndexBuffer(indexes)
            #self.programs[1][0]['a_position'] = VertexBuffer(points) Amir : point program removed
            

class BigPoints:
    def __init__(self, canvas):
        self.canvas = canvas
        prog = Program(BIG_POINT_VERT_SHADER, BIG_POINT_FRAG_SHADER)
        prog['u_alpha'] = 1.0 # throw away
        self.programs = [(prog,'points')]
    def set_pos(self, x_y):
        if x_y is not None and not is_empty(x_y[0]):
            XY = np_stack([Qf32(x_y[d]) for d in [0,1]],axis=1)
            self.programs[0][0]['a_position'] = VertexBuffer(XY)


class Box:
    def __init__(self,canvas):
        self.canvas = canvas
        prog = Program(TRIANGLE_VERT_SHADER, TRIANGLE_FRAG_SHADER)
        prog['u_alpha'] = 1.0
        TPI = IndexBuffer(Qu32((0,1,2,0,3,2)))
        self.programs = [(prog, 'triangles', TPI)]
    def set_pos(self, lrbt):
        if lrbt is not None:
            l,r,b,t = lrbt
            self.programs[0][0]['a_position'] = VertexBuffer(Qf32(((l,b),(r,b),(r,t),(l,t))))


def is_empty(obj):
     return (hasattr(obj,'len') and not len(obj)) or\
            (isinstance(obj,np_ndarray) and not np_prod(obj.shape))

class MAP_Display:
    def __init__(self, canvas, TYPE, coords=None, rgb=None, alpha=1.0, width=1):
        self.canvas = canvas
        self.type = TYPE
        self.width = width
        self.empty = True
        if TYPE=='thin_lines': self.thing = ThinLines(canvas)
        elif TYPE=='thick_lines': self.thing = ThickLines(canvas)
        elif TYPE=='points': self.thing = BigPoints(canvas)
        if coords: self.reshape(coords,width)
        if rgb or alpha is not None: self.set_rgb_a(rgb, alpha)
    def set_rgb_a(self, rgb_a=None, alpha=None):
        if alpha is not None:
            a = alpha
            if rgb_a: c = rgb_a[0:3]
        else:
            if rgb_a:
                c = rgb_a[0:3]
                if len(rgb_a) == 4: a = rgb[3]
                else: a = None
            else: a = None
        for program in self.thing.programs:
            if rgb_a is not None: program[0]['u_color'] = c
            if a is not None: program[0]['u_alpha'] = a
    def reshape(self, x_y_lens=None, width = None):
        if width is not None: self.width = width
        pan = self.canvas.pan
        scale = (self.canvas.scale/self.canvas.size[0],
                 self.canvas.scale/self.canvas.size[1]) 
        for program in self.thing.programs:
            program[0]['u_scale'] = scale
            program[0]['u_pan'] = pan
        if x_y_lens is not None: self.empty = is_empty(x_y_lens[0])
        if not self.empty:
            if x_y_lens is not None:
                if self.type == 'thin_lines': position = x_y_lens
                elif self.type == 'thick_lines': position = list(x_y_lens)+[self.width]
                elif self.type == 'points': position = x_y_lens[0:2]
                self.thing.set_pos(position)
            if self.type == 'points':
                self.thing.programs[0][0]['u_point_size'] = self.width ###self.canvas.point_size() * self.canvas.scale * 2         
    def draw(self):
        if not self.empty:
            for program in self.thing.programs:
                program[0].draw(*program[1:])

        

##    def draw(self):
##        for program in self.programs:
##            program['u_pan'] = self.canvas.pan
##            program['u_scale'] = (self.canvas.scale/self.canvas.size[0],
##                                  self.canvas.scale/self.canvas.size[1])
##        if self.type=='empty':
##            pass
##        elif self.type=='thin_line':
##             self.programs[0].draw('line_strip')
##        elif self.type=='thick_line':
##            self.programs[0].draw('triangles',self.triangle_program_index)
##            self.programs[1].draw('points')

class HUD_Display:
    def __init__(self, canvas, TYPE, coords=None, rgb=None, alpha=1.0, width = 1):
        self.canvas = canvas
        self.type = TYPE
        self.width = width
        self.empty = True
        if TYPE == 'lrbt_box': self.thing = Box(canvas)
        if TYPE == 'lrbt_frame' or TYPE == 'thick_line':self.thing = ThickLines(canvas)
        self.reshape(coords,width)
        if rgb or alpha is not None: self.set_rgb_a(rgb, alpha)
    def set_rgb_a(self, rgb_a=None, alpha=None):
        if alpha is not None:
            a = alpha
            if rgb_a: c = rgb_a[0:3]
        else:
            if rgb_a:
                c = rgb_a[0:3]
                if len(rgb_a) == 4: a = rgb_a[3]
                else: a = None
            else: a = None
        for program in self.thing.programs:
            if rgb_a is not None: program[0]['u_color'] = c
            if a is not None: program[0]['u_alpha'] = a
    def reshape(self, coords=None, width=None):
        if width is not None: self.width = width
        pan = (-self.canvas.size[0]/2, -self.canvas.size[1]/2)
        scale = (2/self.canvas.size[0], -2/self.canvas.size[1]) # mirror around y-axis
        for program in self.thing.programs:
            program[0]['u_scale'] = scale
            program[0]['u_pan'] = pan
        if coords is not None: self.empty = is_empty(coords)
        if coords is not None and not self.empty:
            if self.type == 'lrbt_box': position = coords                
            if self.type == 'thick_line' or self.type == 'lrbt_frame':
                l,r,b,t = coords
                if self.type == 'thick_line':
                    X = (l,r)
                    Y = (b,t)
                elif self.type == 'lrbt_frame':
                    X = (l,r,r,l,l)
                    Y = (b,b,t,t,b)
                position = [Qf32(D) for D in [X,Y]] + [Qu32([len(X)]),self.width]
            self.thing.set_pos(position)
    def draw(self):
        if not self.empty:
            for program in self.thing.programs: program[0].draw(*program[1:])  

def press_callback(data):
    print('BUTTON PRESS',data)

def click_callback(data):
    print('BUTTON CLICK',data)




class Button:
    def __init__(self, canvas, x=0, y=0, w=0, h=0, background_rgb_a = None,
                 pressed_background_rgb_a = None, frame_rgb = None, frame_width = 1,
                 data = None, visible = False):
        Qsave(self, locals(),"canvas data visible background_rgb_a pressed_background_rgb_a frame_rgb frame_width")
        Qnew(self, 'frame^N on_press on_click pressed^F displays{} textboxes[]')
        if self.frame_rgb: self.displays['frame'] = HUD_Display(canvas, 'lrbt_frame', rgb = frame_rgb , width=frame_width)
        self.displays['background'] = HUD_Display(canvas, 'lrbt_box')
        self.displays['background'].set_rgb_a(background_rgb_a)
        self.reshape(x_y=(x,y),w_h=(w,h))
        ###self.depress()
    def lrbt(self):
        W,H = self.canvas.size
        if self.x < 0: lr = [W+self.x-self.w, W+self.x]
        else: lr = [self.x, self.x+self.w]
        if self.y < 0: bt = [H+self.y, H+self.y-self.h] # note y-axis reversal
        else: bt = [self.y+self.h, self.y] # note y-axis reversal
        return lr+bt
    def inside(self, xy):
        if not self.visible: return False
        l,r,b,t = self.lrbt()
        x,y = xy
        if x>=l and x<=r and y<=b and y>=t: # note y-axis reversal
            return True
        return False
    def draw(self):
        if not self.visible: return None
        if (self.background_rgb_a and self.pressed==False) or\
            (self.pressed_background_rgb_a and self.pressed):
                self.displays['background'].draw()
        LRBT = self.lrbt()
        for T in self.textboxes[::-1]: T.draw(LRBT)
        if self.frame_rgb: self.displays['frame'].draw()
##    def press(self):
##        self.pressed = True
##        if self.pressed_background_rgb_a: self.displays['background'].set_rgb_a(self.pressed_background_rgb_a)
##        if self.on_press: self.on_press(self.data)
##    def depress(self):
##        self.pressed = False
##        if self.background_rgb_a: self.displays['background'].set_rgb_a(self.background_rgb_a)
##    def click(self):
##        self.pressed = False
##        if self.on_click: self.on_click(self.data)
    def reshape(self,x_y=None,w_h=None,frame_width=None):
        if frame_width is not None: self.frame_width = frame_width
        if x_y is not None: self.x,self.y = x_y
        if w_h is not None: self.w,self.h = w_h
        for _,D in self.displays.items(): D.reshape(self.lrbt(),self.frame_width)
            


class GeometryOverlay:
    def __init__(self, map_key, geo_type = 'thin_contour', rgb_a = [1]*4, keyboard = ' ', visible=True):
        Qsave(self, locals(), "map_key geo_type rgb_a visible")
        self.keyboard = keyboard.upper()
        self.canvas = None
        self.empty = True
    def link(self, canvas, MAP):
        self.canvas = canvas
        self.map = MAP
        self.geometry = MAP.get(self.map_key)
        if not self.geometry:
            print('Geometry "'+self.map_key+'" not found in MAP.')
            return False
        shape = self.geometry.shape
        fail = True
        if 'points' in self.geo_type and shape==2:
            fail = False
            self.display = MAP_Display(canvas,'points',[self.geometry.data[d].array() for d in 'xy'],
                                       self.rgb_a[0:3],
                                       1.0, # BigPoints doesn't implement alpha (yet)
                                       width = 20)### width fixed to 20 for now
            print('GeometryOverlay.link Fail?',fail)
        elif self.geo_type in 'thin_contour thin_line'.split():
            fail = False
            if shape == 2:
                X,Y = [[self.geometry.data[d].array()] for d in 'xy']
            elif shape == 3:
                X,Y = [self.geometry.data[d].list_of_arrays() for d in 'xy']
            elif shape == 4:
                X,Y = [sum([ll.list_of_arrays() for ll in self.geometry.data[d]],[]) for d in 'xy']
            else:
                fail = True
            if not fail:
                if 'contour' in self.geo_type:
                    X,Y = [[np_concatenate([A,[A[0]]]) for A in D] for D in (X,Y)]
                LENS = [A.shape[0] for A in X]
                X,Y = [np_concatenate(D) for D in (X,Y)]
                self.display = MAP_Display(canvas, 'thin_lines',(X,Y,LENS),
                                           self.rgb_a[0:3],
                                           self.rgb_a[3] if len(self.rgb_a)>3 else 1.0)
        self.empty = fail
        if fail:
            print('GeometryOverlay shape '+self.geo_type+' '+str(shape)+' not implemented!')
            return False
        else:
            #print('GeometryOverlay '+self.map_key+' ('+self.geo_type+' '+str(shape)+') linked to canvas ['+self.keyboard+'].')
            return True
    def draw(self):
        if not self.empty and self.visible: self.display.draw()





##CL_HEATMAP_GAUSSIAN_CODE = ("HEATMAP_GAUSSIAN","""
##__kernel void A(
##const uint M,
##const uint t,
##const uint H,
##const uint W,
##const float half_s2,
##__global const uint *X,
##__global const uint *Y,
##__global const float *V,
##__global uint *C
##){
##uint xy = get_global_id(0);
##int x = xy%W + t;
##int y = xy/W + t;
##__global uint *c = C+(H-y+3*t+1)+(x-t)*H;  //note YX and Y-flip
##int t2 = t*t;
##for(uint m=0; m<M; m++){
##    int r2 = (X[m]-x)*(X[m]-x)+(Y[m]-y)*(Y[m]-y);
##    if(r2<t2) atomic_add(c, (uint)(V[m]*1000000.0*exp(-r2/half_s2)));
##}}""")
class HeatMap:
    def __init__(self, canvas, x, y, values, COMPUTE, kernel = [10,40], visible = False):
        Qsave(self, locals(), "*")
    def global_maximum(self):
        sc = self.canvas.scale * 4
        s, t = self.kernel
        s = s * sc
        m2s = self.canvas.m2screen
        x,y = self.x,self.y
        X,Y = [Qi32(z) for z in list(zip(*[m2s((x[n],y[n])) for n in Qr(x)]))]
        X -= min(X)
        Y -= min(Y)
        C = Qf32((max(Y)+1,max(X)+1), 0)
        for n in Qr(V): C[Y[n],X[n]] = V[n]
        C = gaussian_filter(C,s,truncate=4.0)
        return C.max()
    def draw(self, maximum=None):
        sc = self.canvas.scale * 4
        s, t = self.kernel
        s = s * sc
        t = int(t*sc)+1
        W,H = self.canvas.size
        m2s = self.canvas.m2screen
        x,y = self.x,self.y
        X,Y = [Qi32(z) for z in list(zip(*[m2s((x[n],y[n])) for n in Qr(x)]))]
        I = (X>-t)*(X<W-1+t)*(Y>-t)*(Y<H-1+t)
        if I.size == 0: return None
        X = Qu32(X[I]+t)
        Y = Qu32(Y[I]+t)
        V = self.values[I]
        C = Qf32((H+2*t,W+2*t), 0) # note YX
        _Y = -Y+H+2*t
        for n in Qr(V): C[_Y[n],X[n]] = V[n] # note YX and Y-flip
        C = gaussian_filter(C,s,truncate=4.0)[t:H+t,t:W+t]
        P = Program(HEATMAP_VERT_SHADER, MAGMA_SHADER+HEATMAP_FRAG_SHADER)
        P["u_normvalue"] = C/(maximum if maximum else C.max())
        P["u_normvalue"].interpolation = 'linear'
        P['a_position'] = Qf32([(-1, 1),(-1, -1), (1, 1),(1, -1)])
        P['a_texcoord'] = Qf32([(0, 1),(0, 0), (1, 1),(1, 0)]) 
        P.draw('triangle_strip')
        self.program = P





class RoadDisplay:
    def __init__(self, canvas, name, data, layers, road_width_m, listoflists, lines=None,
                 seg_len_m=None, node_x=None, node_y=None, road_width_override_factor={}):
        t0 = time_time()
        Qsave(self, locals(),'*')
        self.keys = [str(l.button) for l in layers]
        self.key_map = {k:n for n,k in enumerate(self.keys)}
        for p in Qmx('visible legend_name value rgb'):
            setattr(self,p,[getattr(l,p) for l in layers])
        self.N = len(self.value)
        if listoflists:
            print('listoflists not implemented yet in  class RoadDisplay ! ')
            return None
        rx = node_x[lines.data]
        ry = node_y[lines.data]
        rl = lines.lens()
        ro = lines.offsets
        #rslm = seg_len_m.data # not used
        rd = data
        rh = np_arange(rd.size,dtype='uint32')
        self.map_displays = {}
        self.present = [False] * self.N
        for n,layer in enumerate(layers):
            rgb = layer.rgb
            hsv = rgb_to_hsv(*rgb)
            rgb2 = hsv_to_rgb(hsv[0],hsv[1]/2,min(1.0,hsv[2]+0.25))
            rv = layer.value
            ri = rh[rd[rh]==rv]
            if ri.size: # don't draw empty lists of road values
                self.present[n] = True
                lens = rl[ri]
                #xy = Qf32(lol_to_tuple(tuple(tuple((rx[k],ry[k]) for k in Qr(ro[i],ro[i+1])) for i in ri)))
                # This line is nasty and takes about 1/3 of the time (1-2seconds): TODO: vectorize? 
                x,y = [Qf32(lol_to_tuple(tuple(tuple(r[ro[i]:ro[i+1]]) for i in ri))) for r in [rx,ry]]
                rw = road_width_m
                rwof = road_width_override_factor.get(rv)
                if rwof is not None: rw *= rwof
                for P in [('thinroad','thin_lines',rgb,1.0),
                          ('thickroad','thick_lines',rgb,rw),
                          ('thickroadinset','thick_lines',rgb2,rw*0.5)]:
                    print(P[0],n)
                    self.map_displays[P[0]+str(n)] = MAP_Display(canvas, P[1], (x,y,lens), P[2], width=P[3])
                    #print(P[0],"%.2f seconds."%(time_time()-t0))

###############



class Canvas(app_Canvas):
    def set_window(self, Window): self.Window = Window     

    def normalize(self, xy):
        return (xy[0]/(self.size[0]/2)-1, xy[1]/(self.size[1]/2)-1)

    def denormalize(self,xy):
        return ((xy[0]+1)*(self.size[0]/2), (xy[1]+1)*(self.size[1]/2))

    def m2screen(self, xy):
        N = ((xy[0]+self.pan[0])*self.scale/self.size[0],
             -(xy[1]+self.pan[1])*self.scale/self.size[1])
        return self.denormalize(N)

    def screen2m(self, xy):
        N = self.normalize(xy)
        return (N[0]/self.scale*self.size[0]-self.pan[0],
                -N[1]/self.scale*self.size[1]-self.pan[1])

    def screen_lrbt_m(self):
        l,t = self.screen2m((0,0))
        r,b = self.screen2m(self.size)
        return (l,r,b,t)

    ### THESE two functions should become def make_layout() for all buttons:
    def legend_locate(self):
        if self.legend_visible == 1: return (-20, 20)
        else: return (20, 20)
    def prop_box_locate(self):
        if self.legend_visible == 1: return (20, 20)
        else: return (-20, 20)

    def get_menu(self):
        MENU =  [(('MENU',''),),
                 (('VIEW',''),('GRID','G'),('TOPOGRAPHY','T'),('POPULATION','P'),('WORKPLACES','J'),('SLOPE','Y'),('LTS','K')),
                 (('ANALYSIS',''),('SINGLE_ROUTE','U'),('ISOCHRONE','I'),('ALL_ROUTES','O')),
                 (('CONFIG.',''),('VISUAL_SCHEME','F3'))]
        T = self.text
        LANG_MENU = list(tuple((n,T[n].upper(),b) for n,b in COLUMN) for COLUMN in MENU) # Can also substitute keys here, if needed.
        LANG = self.GLOBAL['LANGUAGE']
        li = LANG['LANGS'].index(LANG['LANG'])
        lis = list(Qr(li+1,len(LANG['LANGS'])))+list(Qr(li))
        LANG_MENU[3] = (LANG_MENU[3][0],)+\
                       tuple(('LANG_'+str(i),LANG['LANG_NAMES'][li].upper(),'F2') for i,li in enumerate(lis))+\
                       LANG_MENU[3][1:]
        return LANG_MENU
    
    def make_buttons(self):
        self.button_order = []
        self.buttons = {}
        for column in self.get_menu():
            for n,_,_ in column:
                self.button_order.append('MENU:'+n)
                self.buttons['MENU:'+n] = Button(self,background_rgb_a = self.colourscheme['glass'])
        ###self.buttons['MENU:MENU'].visible = True
        self.button_order += ['prop_box','legend', 'location']################'legend', - second.
        self.buttons['prop_box'] = Button(self,*self.prop_box_locate(), 0,0, self.colourscheme['glass'],
                                          None, self.colourscheme['line'], 3,
                                          'prop_box', bool(self.highlight)) 
        self.buttons['legend'] = Button(self,*self.legend_locate(), 0,0, self.colourscheme['glass'],
                                        None, self.colourscheme['line'], 3,
                                        'legend',bool(self.legend_visible))
##        for _,button in self.buttons.items():
##            button.on_press = press_callback
##            button.on_click = click_callback
        self.buttons['location'] = Button(self, *(-20,-20,1,1),data='location')


    def make_menu(self):
        rgb_a_map = {'0':self.colourscheme['text_key'],'1':self.colourscheme['text_value']}
        text_height = 20; space = 15; x = 300
        text_width = self.typewriter.get_text_width(text_height)
        for c in self.get_menu():
            width = 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            for n,mi in enumerate(c):
                a = mi[1]
                b = ('['+mi[2]+']')*bool(n)
                typeset = self.typewriter.typeset_dictionary({a:b},1+text_width*(len(a)+len(b)+0.5*bool(n)),
                                                             text_height,0.5*bool(n),[],text_height)
                TB = TextBox(self.typewriter,typeset,rgb_a_map,inset_lrbt=(7,7,9,7),rel_pos=(0.0,0.0))
                width = max(width,TB.width)
                self.buttons['MENU:'+mi[0]].textboxes = [TB]
                self.buttons['MENU:'+mi[0]].reshape(x_y = (x, 20 + (text_height+16)*n + space*(n>0)),
                                                    w_h = (width+14, text_height+16))
            for n,mi in enumerate(c):
                if n: self.buttons['MENU:'+mi[0]].reshape(w_h = (width+14, text_height+16))
                else: self.buttons['MENU:'+mi[0]].reshape(w_h = (self.buttons['MENU:'+mi[0]].textboxes[0].width+14, text_height+16))
            x += width + space + 14

    def make_HUD(self):
        #Q(self,'make_legend! make_location! make_menu!')  
        self.make_legend()
        self.make_location()
        self.make_menu()
        # make highlight ### TO_DO

    def make_legend(self):
        if not self.road_displays: return None
        RD = self.road_displays[0]
        text_height = 20
        legend_text = []
        for n in Qr(RD.N):
            if RD.visible[n] and RD.present[n]:
                line = ''
                if self.printing_screen or self.legend_style==1: line += '- '
                else: line += '['+str(RD.keys[n])+'] '
                g = self.text.get(RD.legend_name[n])
                if g == None: g = RD.legend_name[n]
                line += public_name(g)
                legend_text.append(line)
        title_text = public_name(RD.name)
        max_len = len(title_text)
        if legend_text: max_len = max([max_len]+[len(l) for l in legend_text])
        rgb_a_map = {'0':(0,0,0)}###self.colourscheme['text_key']
        count = len(rgb_a_map)
        for n in Qr(RD.N):
            if RD.visible[n] and RD.present[n]:
                rgb_a_map[chr(48+count)] = RD.rgb[n]
                count += 1
        typeset = self.typewriter.typeset_rainbow_list([title_text]+legend_text,text_height=text_height)
        TB = TextBox(self.typewriter,typeset,rgb_a_map,inset_lrbt=(9,9,11,9),rel_pos=(0.0,0.0))
        self.buttons['legend'].textboxes = [TB]
        self.buttons['legend'].reshape(w_h = (TB.width + 18, TB.height + 20))

    def get_location(self):
        if self.last_mouse_pos is None: return None
        a,b = self.last_mouse_pos
        if a<0 or b<0 or a>=self.size[0] or b>=self.size[1]: # This does not happen :) How to detect exit from canvas???
            return None
        xy = self.screen2m((a,b))
        P = Geometry(xy, form='xy')
        P(self.centre)
        ll = P.lonlat()
        TOPOG = self.map.get('topography')
        return {'lonlat':ll, 'xy':xy,
                'h_t':TOPOG.get_altitude_tile(*ll) if TOPOG else None}
    def make_location(self):
        LOC = self.get_location()
        if not LOC:
            pass
            #self.buttons['location'].visible = False
        elif self.buttons['location'].visible:
            text_height = 20
            rgb_a_map = {str(n):self.colourscheme['line'] for n in Qr(10)} # glass
            texts = []
            HL = self.hold_location
            if 'Q' in self.keys_pressed and HL:
                O = GeometryOverlay('hold_mouse_line', 'thin_line', (1,1,0), 'Q' , False)
                self.overlays.append(O)
                self.map['hold_mouse_line'] = Geometry([[LOC['xy'],HL['xy']]]) #mount
                O.link(self, self.map)
                if HL['h_t'] is None or LOC['h_t'] is None: dh = None
                else: dh = LOC['h_t'][0] - HL['h_t'][0]
                d = dist(LOC['xy'],HL['xy'])
                texts += ['S ' + ('?' if dh is None else '%.1f%%'%(100.0*dh/d)) + \
                          ' D %.1fm'%d + \
                          ' dH ' + ('?' if dh is None else '%.1fm'%dh)]
            texts += [self.caption, "%d %d"%(self.size), "%.5f"%(self.scale),"  %.2f %.2f"%(self.pan)]
            texts += [('%.1fm t%d'%LOC['h_t'] if LOC['h_t'] else 'alti?'), ' '.join(self.text_funcs['lonlat'](*LOC['lonlat']))]
            typeset = self.typewriter.typeset_rainbow_list(texts,text_height=text_height)
            TB = TextBox(self.typewriter,typeset,rgb_a_map,inset_lrbt=(2,2,4,2),rel_pos=(0.0,0.0))
            self.buttons['location'].textboxes = [TB]
            self.buttons['location'].reshape(w_h = (TB.width + 4, TB.height + 6))
            
        
    def get_button(self, xy):
        x,y = xy
        for n, B in enumerate(self.button_order):
            if self.buttons[B].inside(xy): return n
        return None

    def point_size(self):
        return max(16/self.scale, self.point_width)

    def highlight_width(self):
        return max(self.road_width_m * 2, 12/self.scale)
    def highlight_reshape(self):
        if self.highlight:
            if self.highlight[3] == 'TR':
                x_y = self.highlight[1]
                self.map_displays['highlight'].reshape(list(x_y)+[[len(x_y[0])]],self.highlight_width())
            elif self.highlight[3] == 'point':
                x_y = Point(self.highlight[1]).buffer(self.point_size()).exterior.coords
                self.map_displays['highlight'].reshape(list(x_y)+[[len(x_y[0])]],self.point_size()/4)
            P1 = self.m2screen(self.highlight[2])
            l,r,b,t = self.buttons['prop_box'].lrbt()
            P2 = ((l+r)/2,(b+t)/2)
            D = LineString([P1,P2]).difference(sPolygon(((l,b),(r,b),(r,t),(l,t))))
            if self.highlight[3] == 'point':
                print('TO_FIX!')
                D = D.difference(sPolygon([self.m2screen(p) for p in x_y]))###TO_FIX
            if D.geom_type == 'LineString' and len(D.coords):
                lbrt = lol_to_tuple(D.coords)
                self.hud_displays['highlight_connector'].reshape((lbrt[0],lbrt[2],lbrt[1],lbrt[3]))
            else: self.hud_displays['highlight_connector'].empty = True
    def highlight_draw(self):
        if self.highlight:
            self.map_displays['highlight'].draw()
            self.hud_displays['highlight_connector'].draw()
            self.buttons['prop_box'].visible = True
    def highlight_remove(self):
        if self.highlight:
            self.highlight = False
            self.buttons['prop_box'].visible = False
            self.please_redraw_hud = True
    def select_highlight(self, pos, toggle=False):
        if pos is None or self.get_button(pos) is not None: return None
        point = self.screen2m(pos)
##        PKs = [str(n) for n in Qr(self.layers)
##               if self.layers[n].visible and self.layers[n].type == 'point']
        if 0: # visibility is difficult to resolve right now, so bypass it:
            RVs = [self.layers[n].value for n in Qr(self.layers)
                   if self.layers[n].visible and self.layers[n].type == 'road']
        else:
            RVs = list(Qr(self.road_displays[0].N)) if self.road_displays else None
        if not RVs:# and not PKs:
            #print('NOTHING VISIBLE')
            self.highlight_remove()
        else: # TO_DO
            P = None###self.hashgrid.find_one_point(point,PKs, self.point_size())
            if P is not None:
                dot = self.hashgrid.geometry[P[0]][P[1]]    
                xy = dot.coords[0]
                self.highlight = (int(P[0]), xy, xy, 'point')
                if 'point' in info:
                    info_dict = self.info['point'][self.layer_value[int(P[0])]][P[1]]
                else:
                    info_dict = {'NO INFO':''}
                dict_keys = None
            else:
                R = self.hashgrid.find_one_line(point,'TRs',100,None) #RVs would go here instead of 'None'; visibility is difficult to resolve right now, so bypass it
                if toggle and self.highlight and R==self.highlight[0]:
                    self.highlight_remove()
                    return None
                if R is not None:
                    line = self.hashgrid.geometry['TRs'][R]
                    x_y = [Qf32(p[d] for p in line.coords) for d in [0,1]]
                    proj = tuple(line.interpolate(line.project(Point(point))).coords[0])
                    self.highlight = (int(R), x_y, proj, 'TR', None) # None <- self.hashgrid.data['roads'][R])
                    if 1:
                        TRs = self.NWK.TRs
                        I = TRs.osm_road[R]
                        info_dict = self.map.roads.props[I]
                        info_dict = self.roads_decoder(info_dict)
                        if 1: # internal properties:
                            #for n in 
                            for p,arr in self.map.roads.property.items():
                                ###if p in 'LTS cycleInfra oneway clampSpeed_kph'.split():
                                if len(p)<4 or p[:4] not in Qmx('fwd_ bwd_ equ_'):
                                    info_dict['*'+p] = str(arr[I])
                        info_dict['TR_int'] = R
                        info_dict['TR_len_m'] = TRs.len_m[R]
                        if 'num_thread' in TRs: info_dict['TR_thread'] = TRs['num_thread'][R]
                        ####ENDS = self.map.roads.lines[int(R)] # cast from numpy value!
                        ####info_dict['!INTERNAL'] = str((ENDS[0],ENDS[-1]))
                        keys = set(info_dict.keys())
                        first = Qmx('name TR_int is_upgradeable is_serivce') + WAY_TYPES
                        last = Qmx('osm_id TR_len_m')
                        middle = list(keys - set(first) - set(last))
                        middle.sort()
                        dict_keys = first + middle + last
                        dict_keys = [k for k in dict_keys if k in info_dict]
            if P is not None or R is not None: 
                text_height = 22.5
                text_width = self.typewriter.get_text_width(text_height)
                max_width = 20.5*text_width
                max_height = text_height*30
                indent = 0.5
                rgb_a_map = {'0':self.colourscheme['text_key'],'1':self.colourscheme['text_value']}
                typeset = self.typewriter.typeset_dictionary(info_dict,max_width,max_height,
                                                             indent,dict_keys,text_height)
                TB = TextBox(self.typewriter,typeset,rgb_a_map,inset_lrbt=(7,7,9,7),rel_pos=(0.0,0.0))
                self.buttons['prop_box'].textboxes = [TB]
                self.buttons['prop_box'].reshape(w_h = (TB.width + 14, TB.height + 16))
                self.highlight_reshape()
                self.please_redraw_hud = True
            else:
                #print('NO HIT!')
                self.highlight_remove()

    def map_reshape(self):
        if self.map_active:
            if self.move_highlight and self.last_mouse_pos is not None:
                self.select_highlight(self.last_mouse_pos)
            self.make_location()
        for _,D in self.map_displays.items(): D.reshape()
        for RD in self.road_displays:
            for _,D in RD.map_displays.items(): D.reshape()
        if self.overlays:
            for O in self.overlays:
                if not O.empty: O.display.reshape()
        if self.highlight: self.highlight_reshape()
        ##########for S in self.studies: S.reshape()
        self.please_redraw_map = True
    
    def shift(self, delta):
        new_pan = (self.pan[0]+delta[0]/self.scale*self.size[0],
                   self.pan[1]+delta[1]/self.scale*self.size[1])
        if new_pan != self.pan:
            self.pan = new_pan
            self.map_reshape()
            
    def zoom(self, delta):
        new_scale = min(self.scale_range[1], max(self.scale_range[0],self.scale*1.15**delta))
        if new_scale != self.scale:
            self.scale = new_scale
            for _,B in self.buttons.items(): B.reshape()
            self.map_reshape()

    def centre_map(self):
        if 'view' in self.map.optional:
            V = self.map.optional['view']
            self.scale = V[2];
            self.pan = (V[3],V[4])
        else:
            self.scale = 0.2;
            self.pan = (0.0,0.0)
                                    
    def on_mouse_release(self, event):
        if self.mouse_press == (-1,): pass
        else: # TO_DO: Make buttons work.
            B = self.get_button(event.pos)
            if (B,) == tuple(self.mouse_press):
                pass#self.buttons[self.button_order[self.mouse_press[0]]].click()
            elif len(tuple(self.mouse_press)) == 1:
                pass#self.buttons[self.button_order[tuple(self.mouse_press)[0]]].depress()
            self.mouse_press = (-1,)
    
    def on_mouse_press(self, event):
        B = self.get_button(event.pos)
        if B is not None:
            self.mouse_press = (B,)
            ###self.buttons[self.button_order[B]].press()
        else:
            self.mouse_press = tuple(event.pos)
        if self.map_active:
            if event.button == 2: self.select_highlight(event.pos, toggle=True)
        
    def on_resize(self, event):
        set_viewport(0, 0, *event.physical_size)
        for _,B in self.buttons.items(): B.reshape()
        if self.intro_active: pass
        else: self.map_reshape()

    def on_mouse_move(self, event):
        if self.map_active:
            self.last_mouse_pos = event.pos
            if event.is_dragging:
                if len(self.mouse_press)==2:
                    x, y = self.normalize(event.pos)
                    x1, y1 = self.normalize(event.last_event.pos)
                    self.shift((x - x1, -y + y1))
            else:
                if self.move_highlight: self.select_highlight(event.pos)
                if len(self.mouse_press)==1:
                    if self.mouse_press[0]>=0: 
                        ###self.buttons[self.button_order[B]].depress()
                        self.mouse_press = (-1,)
                self.make_location()
                self.please_redraw_hud = True     
        
    def on_mouse_wheel(self, event):
        if self.map_active: self.zoom(np_sign(event.delta[1]))

    def export_name(self):
        GEOGRAPHY = self.GLOBAL['GEOGRAPHY']
        image_name = GEOGRAPHY['CITY']
        #if 0: image_name += time_strftime(" %Y %m %d    %H %M %S")
        #else: image_name += ' '+self.caption
        path = 'outputs/'+GEOGRAPHY['CITY']+'/'
        if not os_path_exists(path): os_mkdir(path)
        return path+image_name

    def write_report(self):
        if 1:
            root = self.export_name()
            odt_f = root+' maps.odt'
            KEYS = list(self.map.roads.property.keys())
            KEYS += list(self.NWK.TRs.extra.keys())
            if self.STUDY: KEYS += list(self.STUDY['*Seg'].keys())
            images = [PUBLIC_NAMES[KEY] if KEY in PUBLIC_NAMES else KEY for KEY in KEYS]
        if 1:
            ODT = auto_odt.OdtDocument()
            image = images[0]
            fig = 1
            for image in images:
                im_f = root+' '+image+'.svg'
                if os_path_isfile(im_f):
                    print(im_f)
                    ODT.addImageToDocument(im_f, 'This is '+image+'.', fig, 0.1)
                    fig += 1
                    ODT.addPageBreakToDocument()
            ODT.addParagraphToDocument("Done and donner!")
            ODT.save(odt_f)
            
    def __old_export(self, crop=False):
        pass
##        SVG_SCALE = 1/60
##        print('Making Shapefile & SVG...')
##        GLOBAL = self.GLOBAL
##        TRs = self.NWK.TRs
##        lines = TRs.lines #TRs.seg_len_m,
##        roads = self.map.roads
##        x = roads.node_x[lines.data]
##        y = roads.node_y[lines.data]
##        off = lines.offsets
##        osm_road = TRs.osm_road
##        if crop:
##            lrbt = self.screen_lrbt_m()
##            box = sPolygon(make_box(lrbt = lrbt))
##        if 1:
##            WH = tuple(SVG_SCALE*(lrbt[c[0]]-lrbt[c[1]]) for c in [(1,0),(3,2)])
##            DIM = "%.4f %.4f"%WH
##            #SVG_HEAD = '<?xml version="1.0" encoding="utf-8" ?><svg baseProfile="full" version="1.1" ' +\
##            #'width="%.4f" height="%.4f" '%WH + 'enable-background="new 0 0 %s"'%DIM + 'viewBox="0 0 %s"'%DIM + '><defs />'
##            SVG_HEAD = '<?xml version="1.0" encoding="UTF-8"?>'
##            SVG_HEAD += "<!DOCTYPE svg  PUBLIC \'-//W3C//DTD SVG 1.1//EN\'  "
##            SVG_HEAD += "'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'>"
##            SVG_HEAD += '<svg enable-background="new 0 0 %s" version="1.1" '%DIM
##            SVG_HEAD += 'viewBox="0 0 %s" xml:space="preserve" xmlns="http://www.w3.org/2000/svg">'%DIM
##            shp = {'polylines':[]}
##            X = ((x-lrbt[0])*SVG_SCALE)
##            Y = WH[1] - ((y-lrbt[2])*SVG_SCALE)
##            draw = []
##            svg_lines = {}
##            for n in Qr(lines):
##                L = [(x[i],y[i]) for i in Qr(off[n],off[n+1])]
##                shp['polylines'].append([L]) # writer needs G4, separate roads.
##                if not crop or box.intersects(LineString(L)):
##                    # svg += <line x1="0" y1="0" x2="200" y2="200" style="stroke:rgb(66,0,0);stroke-width:2" />
##                    #for i in Qr(off[n],off[n+1]-1): # must be -1 so as to access [i+1]
##                        #svg += '<line x1="%d" y1="%d" x2="%d" y2="%d"'%(X[i],Y[i],X[i+1],Y[i+1])
##                        #svg += '<path d="M%d %d %d %d"/>'%(X[i],Y[i],X[i+1],Y[i+1])
##                    svg_lines[n] = '<polyline points="'+' '.join('%.4f,%.4f'%(X[i],Y[i]) for i in Qr(off[n],off[n+1]))+'"/>'
##                    draw.append(n)
##            if 0:
##                print('Shapefile')
##                field_names = 'LTS hasHouses oneway cycleLane cycleTrack'.split()#list(roads.property.keys())
##                records = {}
##                rows = TRs.osm_road###[Qu32(draw)]  
##                for field in field_names:
##                    records[field] = roads.property[field][rows]
##                if 'STUDY' in GLOBAL and '*Seg' in GLOBAL['STUDY']:
##                    for SEG in ['Seg','*Seg']:
##                        for segs, data in GLOBAL['STUDY'][SEG].items():
##                            fn = {'Seg':'','*Seg':'L '}[SEG]+PUBLIC_NAMES[segs]
##                            field_names.append(fn)
##                            records[fn] = data#[Qu32(draw)]  
##                shp['field_names'] = field_names
##                shp['records'] = records
##                try: write_shp_from_dict(shp,self.export_name()+'.shp')
##                except: print("Failed making .SHP")
##            if 1:
##                def rgb_svg(rgb):
##                    return 'rgb(%d,%d,%d)'%tuple([min(255,int(f*255)) for f in rgb[0:3]])
##                GRAPHS = [(KEY,layer,prop[KEY][osm_road]) \
##                          for KEY,layer in self.LAYERS.items() if KEY in roads.property]
##                GRAPHS += [(KEY,layers[KEY],data) for KEY,data in TRs.extra.items()]
##                if self.STUDY:
##                    GRAPHS += [(KEY,self.LAYERS['HEAT8'],data) \
##                               for KEY,data in self.STUDY['*Seg'].items()]
##                for GRAPH in GRAPHS:
##                    KEY,layer,data = GRAPH
##                    PUBLIC = PUBLIC_NAMES[KEY] if KEY in PUBLIC_NAMES else KEY
##                    print(KEY, PUBLIC)
##                    RGB = {l.value:rgb_svg(l.rgb) for l in layer}
##                    svg = SVG_HEAD
##                    val_dict = {l.value:n for n,l in enumerate(layer)}
##                    svg_layers = {l.value:'<g fill="none" stroke="'+RGB[l.value]+'"'+\
##                                  ' stroke-linecap="round" stroke-width='+\
##                                  '"%.6f">'%(SVG_SCALE*self.road_width_m) for l in layer}
##                    for n in draw: svg_layers[data[n]] += svg_lines[n]
##                    svg += ' '.join(svg_layers[l.value]+'</g>' for l in layer)
##                    if 1: # legend
##                        visible = set(data[draw])
##                        legend = [(rgb_svg(self.colourscheme['text_key']),PUBLIC)]
##                        for l in layer:
##                            if l.value in visible:
##                                legend.append((RGB[l.value],l.legend_name))
##                        width = WH[0]/SVG_SCALE
##                        leg_width = max(len(t[1]) for t in legend)*200
##                        box = [width-200-leg_width,200,leg_width, 450+300*len(legend)]
##                        svg += '<rect x="%.4f" y="%.4f" width="%.4f" height="%.4f"'%tuple(v*SVG_SCALE for v in box)
##                        svg += ' style="fill:%s;'%rgb_svg(self.colourscheme['glass'])
##                        svg += ' stroke:%s;'%rgb_svg(self.colourscheme['line'])
##                        svg += ' stroke-width:%.4f;'%(50*SVG_SCALE)
##                        svg += ' fill-opacity:%.2f;'%0.8#(self.colourscheme['glass'][3] if len(self.colourscheme['glass'])>3 else 1.0)
##                        svg += ' stroke-opacity:1.0"/>'
##                        for n, R in enumerate(legend):
##                            svg += '<text x="%.4f" y="%.4f" font-size="%.4fem"'%tuple(v*SVG_SCALE for v in [100+box[0], 300*(n+2), 17])
##                            if n==0: svg += ' text-decoration="underline"'
##                            svg += ' fill="%s">%s</text>'%R
##                    svg += '</svg>'
##                    im_name = self.export_name()+' '+PUBLIC
##                    with open(im_name+'.svg','w') as f: f.write(svg)
######                    subprocess_run(["C:/Program Files/Inkscape/bin/inkscape.exe", im_name+".svg", "-E",
######                                    im_name+".eps", "--export-ignore-filters", "--export-ps-level=3"],shell=True)                         
##        print('DONE!')
##        self.write_report()
##        print('ODT Ready.')

    def on_draw(self, event = None):
        if self.intro_active: return None
        if self.printing_screen:
            es = self.export_state
            #print(es.task, es.state, es.state_X, es.state_Y)
            if es.state == 'drawing':
                d = (es.tiles-1)/2
                c = es.centre
                self.pan = (c[0] + es.dxm*(es.state_X-d),
                            c[1] + es.dym*(es.state_Y-d))
                self.map_reshape()
                t = self.export_state.task
                clear(color=self.colourscheme['background'])
                if 'topography' in self.map and 'topography' in t: 
                    self.map_displays['topography'].draw()
                for RD in self.road_displays:
                    if RD.name in t:
                        for n in Qr(RD.N):
                            K = 'thickroad'+str(n)
                            if K in RD.map_displays: RD.map_displays[K].draw()
                        break # only draw one road datum
                if 'pop_dens' in self.map and 'pop_dens' in t: 
                    self.map_displays['pop_dens'].draw(alpha = 0.8)
                if self.workplaces and 'workplaces' in t:
                    # only draw first workplace, only one implemented for now:
                    if es.state_X == 0 and es.state_Y == 0:
                        es.workplaces_global_maximum[0] = self.workplaces[0].global_maximum()
                    self.workplaces[0].draw(es.workplaces_global_maximum[0])
                if 'cells' in self.map and 'cells' in t: self.map_displays['cells'].draw(alpha=0.8)
        else:
            thresh = [2/self.road_width_m, 6/self.road_width_m]
    ##        if self.printing_screen:
    ##            thresh = [v/self.print_screen_mag_factor for v in thresh]
            self.draw_count += 1
            if self.please_redraw_map or self.please_redraw_hud:
                clear(color=self.colourscheme['background'])
                if 'topography' in self.map and self.topography_view:
                    self.map_displays['topography'].draw()
                if self.road_displays:
                    RD = self.road_displays[0]
                    for n in Qr(RD.N):
                        if RD.visible[n]:
                            K = ('thin' if self.scale < thresh[0] else 'thick')+'road'+str(n)
                            if K in RD.map_displays: RD.map_displays[K].draw()
                            if self.scale > thresh[1]:
                                K = 'thickroadinset'+str(n)
                                if K in RD.map_displays: RD.map_displays[K].draw()
    ##                for PK in self.point_keys:
    ##                    if self.layers[PK].visible:
    ##                        self.map_displays['points'+str(PK)].draw()
                if 'pop_dens' in self.map and self.pop_dens_view:
                    self.map_displays['pop_dens'].draw(alpha = 0.9 - self.pop_dens_view/5.0)
                ############if self.visible_study:
                 #################   self.studies[self.visible_study-1].draw()
                for W in self.workplaces: # Draw workplace heatmaps
                    if W.visible: W.draw()
                if 'cells' in self.map and self.cells_state: self.map_displays['cells'].draw(alpha=0.8)
                if self.overlays: # Draw Overlays
                    for O in self.overlays:
                        if O.visible and not O.empty: O.draw()
                if self.printing_screen: # draw legend only
                    if self.caption not in ['Pop Density', 'Topography']:
                        if 'legend' in self.buttons: self.buttons['legend'].draw()
                else: # Draw all HUD
                    if self.highlight: self.highlight_draw()
                    for B in self.button_order[::-1]:
                        if B in self.buttons: self.buttons[B].draw()
        # Sync Fence:
        self.please_redraw_map = False
        self.please_redraw_hud = False
        self.glFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)


    def export_image(self, array):
        H, W, _ = array.shape # Note: image is YX
        img = PIL_Image.fromarray(array)
        img1 = PIL_ImageDraw.Draw(img)
        w = 3-1; b = 5*(w+1); a = int((w+1)*1.4+0.5)
        c, k = [tuple(int(n*255) for n in u) for u in (URBINTEL_RGB,URBINTEL_BACK_RGB)]
        # Note: image is YX
        if 1:
            d = b-a+1
            img1.polygon([(0,H),(0,H-d),(d,H)], fill = k)
            img1.polygon([(W,0),(W-d,0),(W,d)], fill = k)
            for coords in ([0, 0, W-b+w, w],[0,0,w,H-b+w],[b-w,H-w,W,H],[W-w,b-w,W,H]):
                img1.rectangle(coords, fill = c)
            img1.polygon([(W-b,0),(W-b+a,0),(W,b-a),(W,b)], fill = c)
            img1.polygon([(0,H-b),(0,H-b+a),(b-a,H),(b,H)], fill = c)
        if 0:
            for coords in ([0, 0, W, w],[0,0,w,H],[0,H-w,W,H], [W-w,0,W,H]):
                img1.rectangle(coords, fill = c)
            img1.polygon([(0,H),(0,H-b),(b,H)], fill = c)
            img1.polygon([(W,0),(W-b,0),(W,b)], fill = c)
        return np_array(img)
    
    def Tick(self,event):
        if self.printing_screen and not self.is_drawing:
            es = self.export_state
            s = es.state
            if s == 'start':
                self.resizable = False
                v = self.map.get('view')
                es.old_pos = (self.scale, *self.pan)
                if v:
                    self.scale = v[2]
                    self.pan = v[3:5]
                es.centre = self.pan
                es.dxp, es.dyp = self.size
                p0 = self.screen2m((0,0))
                p1 = self.screen2m(self.size)
                es.tiles = int(es.max_png_dim/max(es.dxp,es.dyp))
                self.scale *= es.tiles
                es.dxm = (p1[0]-p0[0])/es.tiles
                es.dym = (p1[1]-p0[1])/es.tiles
                es.array = np_empty((es.tiles*es.dyp, es.tiles*es.dxp, 3),dtype='uint8')
                # sic: array is Y-X; _screenshot() makes uint8                
            elif s == 'drawing':
                ss =_screenshot((0,0,es.dxp,es.dyp))[:,:,0:3]
                X = es.state_X
                Y = es.state_Y
                T = es.tiles
                es.array[(es.dyp*(T-Y-1)):(es.dyp*(T-Y)),
                         (es.dxp*(T-X-1)):(es.dxp*(T-X)),:] = ss 
                if X == T-1 and Y == T-1:
                    imsave(self.export_name()+' '+' '.join(public_name(t) for t in es.task)+'.png',
                           self.export_image(es.array))
            elif s == 'end':
                o = es.old_pos
                self.scale = o[0]
                self.pan = o[1:3]
                self.map_reshape()
                self.resizable = True
            es.step()
            self.is_drawing = True
            self.update()
            return None
        self.ticks +=1
        # Timers:
        tt = time_time()
        dt = tt - self.last_time
        self.last_time = tt
        for k in self.countdown:
            if self.countdown[k]>0: self.countdown[k] -= dt
        for k in self.countup: self.countup[k] += dt
        # Movement keys:
        if self.map_active and not self.printing_screen:
            dx = (2000/self.key_hold_fq)/self.size[0]
            dy = (2000/self.key_hold_fq)/self.size[1]
            diag = 0.5**0.5
            while self.countdown['key_hold']<=0 and self.keys_pressed:
                self.countdown['key_hold'] += 1/self.key_hold_fq #generates an event every 1/self.key_hold_fq sec.
                z = 0
                sh = Qf32(2, 0)
                for key in self.keys_pressed:
                    if key == 'Z': z += 1
                    elif key == 'X': z-= 1
                    elif key in ['W', 'Up']: sh+=(0,-1)
                    elif key in ['S', 'Down']: sh+=(0,1)
                    elif key in ['A', 'Left']: sh+=(1,0)
                    elif key in ['D', 'Right']: sh+=(-1,0)
                    elif key == 'Home': sh+=(diag,-diag)
                    elif key == 'PageUp': sh+=(-diag,-diag)
                    elif key == 'End': sh+=(diag,diag)
                    elif key == 'PageDown': sh+=(-diag,diag)
                if z: self.zoom(np_sign(z)*20/self.key_hold_fq)
                if sh[0] or sh[1]: self.shift(sh/(sh[0]**2+sh[1]**2)**0.5*[dx,dy])
        # Drawing Sync:
        if self.is_drawing and self.glFence is not None:#==None happens once.
            sync = glGetSync(self.glFence, GL_SYNC_STATUS)[0]
            if sync == GL_UNSIGNALED: pass
            elif sync == GL_SIGNALED: 
                self.draw_s.append(self.countup['draw'])
                self.countup['draw'] = 0.0
                self.is_drawing = False
                glDeleteSync(self.glFence)
                self.glFence = None
            else: print('Unusual GL Fence Signal:',sync) # should be ok to pass here as is_drawing remains True
        while self.countdown['1second']<=0:
            self.countdown['1second'] += 1.0
            self.seconds += 1
            if self.seconds%10 == 0:
                print('SEC =', self.seconds,'FPS =', int(1/max(self.draw_s)+0.5) if self.draw_s else 0)
            self.draw_s = []
        if not self.is_drawing and (self.please_redraw_map or self.please_redraw_hud):
            self.is_drawing = True
            self.update()
    def on_key_release(self, event):
        if not event.key: return None
        key = event.key.name
        while key in self.keys_pressed: # 'while' is a partial fix to key release outside app.
            self.keys_pressed.pop(self.keys_pressed.index(key))
    def on_key_press(self, event):
        if self.printing_screen: return None # ignore keys during screen printing
        if not event.key: return None
        key = event.key.name
        shift = False; ctrl = False; alt = False
        for m in event.modifiers:
            n = m.name
            if n == 'Shift': shift = True
            if n == 'Control': ctrl = True
            if n == 'Alt': alt = True
        self.keys_pressed.append(key)
        if not self.map_active: return None
        if self.overlays:
            for O in self.overlays:
                #print(O.keyboard)
                if O.keyboard and key == O.keyboard:
                    O.visible = not O.visible
                    self.please_redraw_map = True
        roads_visibility_changed = False
        RD = self.road_displays[0] if self.road_displays else None
        if key == 'C':
            self.centre_map()
            self.map_reshape()
        elif key == 'Q':
            self.hold_location = self.get_location()
        elif key in '1234567890':
            if RD: 
                n = RD.key_map.get(key)
                if n is not None:
                    RD.visible[n] = not RD.visible[n]
                    roads_visibility_changed = True
        elif key in ',<.>':
            if self.road_displays:
                self.road_displays = circular(self.road_displays, 1 - 2*(key in ',<')) # cycle road display
                roads_visibility_changed = True
                self.caption = public_name(self.road_displays[0].name)
                self.please_redraw_hud = True
        elif key == '`':
            if RD:
                all_visible = True
                for v in RD.visible:
                    if not v: all_visible = False
                RD.visible = [not all_visible]*len(RD.visible)
                roads_visibility_changed = True
        elif key == 'H':
            self.move_highlight = not self.move_highlight
            if self.move_highlight and self.last_mouse_pos is not None:
                self.select_highlight(self.last_mouse_pos)
        elif key == 'J':
            N = len(self.workplaces)
            if N:
                self.visible_workplace = (self.visible_workplace+1)%(N+1)
                for W in self.workplaces: W.visible = False
                if self.visible_workplace:
                    self.workplaces[self.visible_workplace-1].visible = True
                    self.caption = 'Workplaces'
                self.please_redraw_map = True
        elif key == 'L':
            if shift:
                self.legend_style = (self.legend_style+1)%2
                print('self.legend_style',self.legend_style)
                self.buttons['legend'].reshape(x_y=self.legend_locate())
            else:
                self.legend_visible = (self.legend_visible+1)%3
                self.buttons['legend'].reshape(x_y=self.legend_locate())
                self.buttons['prop_box'].reshape(x_y=self.prop_box_locate())
                self.highlight_reshape()
            self.buttons['legend'].visible = bool(self.legend_visible)
            
            self.please_redraw_hud = True
        elif key == 'M':
            self.menu_visible = not self.menu_visible
            for n,b in self.buttons.items():
                if n.startswith('MENU:'): b.visible = self.menu_visible
            self.buttons['MENU:MENU'].visible = True
            self.please_redraw_hud = True
        elif key == 'P':
            if 'pop_dens' in self.map:
                self.pop_dens_view = (self.pop_dens_view+1)%5
                self.please_redraw_map = True
                if self.pop_dens_view: self.caption = 'Pop Density'
        elif key == 'T':
            if 'topography' in self.map:
                self.topography_view = not bool(self.topography_view)
                self.please_redraw_map = True
                if self.topography_view: self.caption = 'Topography'
        elif key == 'Y':
            self.cells_state = int(not self.cells_state)
            self.please_redraw_map = True
            if self.cells_state: self.caption = 'Access'
        elif key == 'G':
            #self.export()
            if shift: self.export_state = ExportState(self)
            else: # draw current view to file
                ss =_screenshot((0,0,*self.size))[:,:,0:3]                
                imsave(self.export_name()+' '+public_name(self.caption)+' '+\
                    time_strftime("%d_%m_%y %H_%M_%S")+'.png', self.export_image(ss))
        elif key == '\\':
            if self.highlight:
                tr = self.highlight[0]
                W = self.map.roads[tr]['osm_id']
                webbrowser_open('openstreetmap.org/way/%d'%W)
        elif key in '/?':
            self.buttons['location'].visible = not self.buttons['location'].visible
            self.please_redraw_hud = True
        elif key == 'F2':
            LA = self.LANGUAGE
            li = LA['LANGS'].index(LA['LANG'])
            li = (li+1)%len(LA['LANGS'])
            LA['LANG'] = LA['LANGS'][li]
            self.text = LA['TEXT'][LA['LANG']]
            self.text_funcs = LA['FUNCS'][LA['LANG']]
            self.make_HUD()
            self.please_redraw_hud = True
        elif key == 'F3':
            VI = self.VISUAL
            VI['SCHEME'] = (VI['SCHEME']+1)%len(VI['SCHEMES'])
            self.colourscheme = VI['COLOURS'][VI['SCHEMES'][VI['SCHEME']]]
            # Still have to inject colourscheme everywhere!
            self.please_redraw_hud = True
        elif 0:#key == 'F4' or key == 'Tab': # imporved by Saurabh
            pass
##            print(self.size)
##            print('Making map image...',end='')
##            X_SIZE = self.size[0]*self.print_screen_mag_factor
##            Y_SIZE = self.size[1]*self.print_screen_mag_factor
##            #backup = [self.scale, self.pan]
##            ##self.scale *= MAG_FACTOR
##            self.map_reshape()
##            aa_texture = Texture2D(shape=(Y_SIZE,X_SIZE,3)) # sic: Y,X
##            high_res_fbo = FrameBuffer(aa_texture)
##            #####aa_out_fbo = FrameBuffer(RenderBuffer(shape=(self.size[1],self.size[0],3))) # AntiAliasing output FBO of size same as window but this can be of any size
##            self.please_redraw_map = True
##            self.please_redraw_hud = True
##            self.printing_screen = True
##            self.make_legend() # legend has to be redone to change entries (no keys: [1])
##            with high_res_fbo:
##                glViewport(0,0,X_SIZE,Y_SIZE) # saurabh - glviewport decides clipping planes bounds for render pass that also has to be updated with new framebuffer size
##                self.on_draw()
##                image = _screenshot((0,0,X_SIZE,Y_SIZE))
##            Q(image)
##            print(image.size)
##            image = image[:,:,0:3] # drop alpha channel(4th)
##            PILimage = PIL_Image.fromarray(image)
##            PILimage = PILimage.resize((X_SIZE//self.print_screen_ds_factor,
##                                        Y_SIZE//self.print_screen_ds_factor),
##                                       self.VISUAL['print_screen_aa_filter'])
##            image = np_array(PILimage)
##            Q(image)
##            print(image.size)
##            for ext in 'png jpg'.split():
##                imsave(self.export_name()+' '+self.caption+'.'+ext,image)
##            #####AntiAliasing(aa_texture,aa_out_fbo,'outputs/'+GEOGRAPHY['CITY']+'/'+'AA '+image_name,shape=(X_SIZE,Y_SIZE)) #saurabh  
##            self.printing_screen = False
##            print('done!')
##            #self.scale, self.pan = backup
##            glViewport(0,0,self.size[0],self.size[1])# saurabh - resetting default framebuffer size as window size
##            self.map_reshape()
        if roads_visibility_changed:
            #print('roads_visibility_changed')
            self.make_legend()
##            if self.highlight:
##                for layer in self.layers:
##                    if layer.value == self.highlight[4] and not layer.visible:
##                        self.highlight_remove()
            self.please_redraw_map = True
            self.please_redraw_hud = True
            print('CAPTION:',self.caption)



##        print(gloo.gl.glTexParameteri(gloo.gl.GL_ALIASED_POINT_SIZE_RANGE))
##        #gloo. ...  print(gl.glGetBufferParameter(gl.GL_ALIASED_POINT_SIZE_RANGE))
##        print(gl.glGetFramebufferAttachmentParameter(gl.GL_ALIASED_POINT_SIZE_RANGE))
##        print(gl.glGetParameter(gl.GL_ALIASED_POINT_SIZE_RANGE))
##        print(gl.glGetProgramParameter(gl.GL_ALIASED_POINT_SIZE_RANGE))
##        print(gl.glGetRenderbufferParameter(gl.GL_ALIASED_POINT_SIZE_RANGE))
##        print(gl.glGetShaderParameter(gl.GL_ALIASED_POINT_SIZE_RANGE))
##        print(gl.glGetTexParameter(gl.GL_ALIASED_POINT_SIZE_RANGE))
##        print(gl.glTexParameterf(gl.GL_ALIASED_POINT_SIZE_RANGE))
    def __init__(self, INTERACTIVE, VISIBLE_DATA, COMPUTE, size=(700,600)):
        Qsave(self, INTERACTIVE) # moves all configuration from one object to other.
        self.COMPUTE = COMPUTE
        MAP = self.map
        LAYERS = self.LAYERS
        TRs = self.NWK.TRs
        STUDY = self.STUDY
        SLOPE = self.SLOPE
        #Amir : setting config=dict(samples = 4) will make multisampling use 4 samples
        v = MAP.get('view')
        if v:
            size = (int(v[0]),int(v[1]))
            self.scale = v[2]
            self.pan = (v[3],v[4])
        self.scale_range = (0.001, 25)
        app_Canvas.__init__(self, title='', keys='interactive',size=size,vsync=True,config=dict(samples = 4)) # samples = 2,4,8. 2 being the fastest , 8 being with the highest quality
        #self.window.set_size(400, 400)
        set_viewport(0, 0, *self.physical_size)
        set_state(clear_color=self.colourscheme['background'],
            blend=True,blend_func=('src_alpha', 'one_minus_src_alpha')) # Must happen before any programs initalized.        
        glDisable(GL_CULL_FACE)
        if 1:# MSAA:
            # Note: this requiered .whl installation of PyOpenGL 3.1.6 to work:
            glutInit()
            glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_MULTISAMPLE)
            glEnable(GL_MULTISAMPLE)
        self.timer = app_Timer(interval=0.01, connect = self.Tick, iterations=-1, start=True, app=self.app)
        self.last_time = time_time()
        self.centre_map()
        self.centre = MAP.centre
        for obj in Qmx('pop_dens topography cells'):
            O = MAP.get(obj)
            if O:
                O.set_canvas(self)
                self.map_displays[obj] = O
        print('Building roads')
        common = (self.road_width_m, None, self.TRs_visible_lines,
                  None, MAP.roads.node_x, MAP.roads.node_y)
        IN = self.TRs_visible_ix
        for name in list(MAP.roads.property)+list(TRs.extra):
            if name in LAYERS and name in VISIBLE_DATA:
                print('TR Property:', name)
                D = TRs[name]
                data = np_array(Qfilter(D,IN), dtype = D.dtype)
                self.road_displays.append(QcF(RoadDisplay, [self, name, data, LAYERS[name], *common]))
        if STUDY and '*Seg' in STUDY:
            for name, D in STUDY['*Seg'].items():
                if name in VISIBLE_DATA:
                    print('Seg study:', name)
                    data = np_array(Qfilter(D,IN), dtype = D.dtype)
                    rwof = {n:1.5 for n in range(5,8)}
                    self.road_displays.append(QcF(RoadDisplay, [self, name, data, LAYERS['HEAT8'], *common,rwof]))
        if SLOPE: # abs(uint8 array) -> draws absolute slope
            for name, slope in SLOPE.items():
                if name in VISIBLE_DATA:
                    print('Slope:', name)
                    self.road_displays.append(RoadDisplay(self, name, slope.draw, LAYERS['SLOPE'], self.road_width_m, None,
                        slope.lines, None, slope.node_x, slope.node_y))
        if self.road_displays: self.caption = self.road_displays[0].name
        #workplaces
        KL = [k for k in MAP.optional if k.startswith('workplaces')]
        KL.sort()
        for K in KL:
            W = MAP[K]
            if 'count' in W:
                self.workplaces.append(HeatMap(self,
                    W.data['x'].array(),W.data['y'].array(),
                    W['count'],self.COMPUTE))
        self.typewriter = Typewriter(self, ASCII_FONT)        
        self.please_redraw_map = True
        self.please_redraw_hud = True
        for O in self.overlays: O.link(self, MAP)
        # highlight:
        self.map_displays['highlight'] = MAP_Display(self,
            'thick_lines', rgb = self.colourscheme['line'])
        self.hud_displays['highlight_connector'] = HUD_Display(self,
            'thick_line', rgb = self.colourscheme['line'],width=3)
        # buttons:
        self.make_buttons()
        self.make_HUD()
        print('Showing Canvas...')
        self.show()   

        
class CanvasSetup:
    def __init__(self, GLOBAL):
        Qnew(self,"""legend_visible#1
        legend_style#0 draw_count ticks seconds cells_state
        buttons{} map_displays hud_displays
        pop_dens_view^F topography_view visible_workplace highlight move_highlight menu_visible
        printing_screen^F intro_active suppress_hud is_drawing
        map_active^T please_redraw_map please_redraw_hud
        last_mouse_pos^N glFence hold_location export_state
        workplaces[] draw_s keys_pressed road_displays""")
        if 1: #Q-this:
            #locals().update(GLOBAL)###doesnt work?
            MAP = GLOBAL['MAP']
            LANGUAGE = GLOBAL['LANGUAGE']
            VISUAL = GLOBAL['VISUAL']
            COMPUTE = GLOBAL['COMPUTE']
            NWK = GLOBAL['NWK']
            ALGORITHM = GLOBAL['ALGORITHM']
            LAYERS = GLOBAL['LAYERS']
            STUDY = GLOBAL.get('STUDY')#Q-?
            SLOPE = GLOBAL.get('SLOPE')#Q-?
            OVERLAYS = GLOBAL.get('OVERLAYS')#Q-?
            TRs = NWK.TRs #Q-NW>TRs
            self.overlays = OVERLAYS 
            self.VISUAL = VISUAL
            self.LAYERS = LAYERS
            self.STUDY = STUDY
            self.SLOPE = SLOPE
            self.GLOBAL = GLOBAL
            self.LANGUAGE = LANGUAGE
            self.map = MAP
            self.NWK = NWK
            # Q-end
            self.road_width_m = GLOBAL['PHYSICAL']['road_width_m']
            self.text = LANGUAGE['TEXT'][LANGUAGE['LANG']]
            self.text_funcs = LANGUAGE['FUNCS'][LANGUAGE['LANG']]
        self.print_screen_ds_factor = VISUAL['print_screen_downsample']
        self.print_screen_mag_factor = VISUAL['print_screen_resize'] * self.print_screen_ds_factor
        self.colourscheme = VISUAL['COLOURS'][VISUAL['SCHEMES'][VISUAL['SCHEME']]]
        self.lrbt_m = GLOBAL['GEOGRAPHY']['LRBT_m_VIEW']
        print(self.lrbt_m)###
        self.hashgrid = HashGrid(COMPUTE, self.lrbt_m, ALGORITHM['MAP_HASH_GRID_m'])
        #self.mesogrid = HashGrid(COMPUTE, self.lrbt_m,ALGORITHM['MAP_MESO_GRID_m'])
        #self.macrogrid = HashGrid(COMPUTE,self.lrbt_m,ALGORITHM['MAP_MACRO_GRID_m'])
        self.mouse_press = (-1,)
        self.caption = 'generic map'
        #
        lines = TRs.lines
        rx = MAP.roads.node_x[lines.data]
        ry = MAP.roads.node_y[lines.data]
        ro = lines.offsets
        rslm = TRs.seg_len_m.data
        self.hashgrid.hash_lines('TRs', rx, ry, ro, rslm)
        boundary = MAP.get('buffered_boundary') #Q>'map muni?'
        if 0: #boundary: #this is slow, removes TRs outside boundary.
            print('Selecting TRs in municipal_boundary:')
            bound = MultiPolygon([sPolygon(P) for P in boundary.xy()])
            bound = bound.buffer(-GLOBAL['PHYSICAL']['city_dilate_m'])
            #frame = Frame(MAP['buffered_boundary'].get_lrbt('m')) #Q>'map muni get_lrbt!`m'
            #bound = frame.sPolygon # Qclassone(Frame,[params],'sPolygon')
            print('Finding intersections:')
            IN = Qbool(bound.intersects(LS) for LS in TRs.get_LineStrings())
            self.TRs_visible_ix = IN
            print('Done:',sum(IN),len(IN))
            self.TRs_visible_lines = LoL(Qfilter(TRs.lines.tolol(), IN), TRs.lines.dtype)
            #!rslm = LoL(Qfilter(TRs.seg_len_m.tolol(), IN), TRs.seg_len_m.dtype).data
            print('Done filtering 1 LoL')
        else:
            self.TRs_visible_ix = Qbool(len(TRs),1)
            self.TRs_visible_lines = TRs.lines
        self.roads_decoder = MAP.roads.props_decoder
        self.point_texture = Texture2D(imageio_imread(POINT_TEXTURE_FILE)) #Amir : loading point texture
        self.point_width = 10
        #self.osm_roads = [(k,v) for k,v in MAP.roads.property.items()]
        # Tick:
        self.key_hold_fq = VISUAL['key_hold_fq'] # implies fps when pressing motion keys - change for large maps?
        self.countdown = {'key_hold':0.0,'1second':1.0}
        self.countup = {'alive':0.0,'draw':0.0}

def run(GLOBAL, VISIBLE_DATA, VARS=''): # ignore VARS
    GLOBAL['INTERACTIVE'] = CanvasSetup(GLOBAL)
    canvas = Canvas(GLOBAL['INTERACTIVE'], VISIBLE_DATA, GLOBAL['COMPUTE'])
    canvas.set_window(canvas.app.backend_module._Window)
    canvas.app.run()

class ExportState:
    def __init__(self, canvas):
        self.state = 'off'
        self.state_N = 0
        self.state_X = 0
        self.state_Y = 0
        self.workplaces_global_maximum = {}
        self.tasks = canvas.GLOBAL['TO_DRAW']
        self.task = self.tasks[0]
        self.canvas = canvas
        canvas.printing_screen = True
        self.max_png_dim = canvas.GLOBAL['VISUAL']['max_png_dim']
    def step(self):
        s = self.state
        if s == 'off': self.state = 'start'
        elif s == 'start': self.state = 'drawing'
        elif s == 'drawing':
            self.state_X += 1
            if self.state_X == self.tiles:
                self.state_X = 0
                self.state_Y += 1
            if self.state_Y == self.tiles:
                self.state_Y = 0 
                self.state_N += 1                
                if self.state_N == len(self.tasks): self.state = 'end'
                else: self.task = self.tasks[self.state_N]
        elif s == 'end': self.state = 'stop'
        elif s == 'stop':
            self.canvas.printing_screen = False
            self.canvas.please_redraw_hud = True
            self.canvas.please_redraw_map = True

if __name__ == '__main__':
    from MAIN import MAIN
    MAIN(['PE','Charlottetown'])



