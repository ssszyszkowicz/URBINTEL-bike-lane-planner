from utils import *
from qcommands import *
from svgwrite import Drawing as svgwrite_Drawing, rgb as svgwrite_rgb
from fpdf import FPDF as fpdf_FPDF
from fileformats import load_data
#import osmmap, osmfilter


def draw_svg_path(dwg, lines, path, scale, xa, xf, ya, yf):
    if len(path) > 1:
        for n in Qr(len(path)-1):
            x1 = xf+(path[n][0]-xa)*scale
            y1 = yf+(ya-path[n][1])*scale
            x2 = xf+(path[n+1][0]-xa)*scale
            y2 = yf+(ya-path[n+1][1])*scale
            lines.add(dwg.line(start=(x1, y1), end=(x2, y2)))
def draw_svg_layers(data):
    page_w = 297
    page_h = 210
    page_unit = 0.001
    frame_lrbt = [25, page_w-25, page_h-25, 25]
    frame_w = frame_lrbt[1]-frame_lrbt[0]
    frame_h = frame_lrbt[2]-frame_lrbt[3]
    frame_aspect = frame_h/frame_w
    map_lrbt =[data['lrbt_view'][n]+[-1000,1000,-1000,1000][n] for n in Qr(4)]
    map_w = map_lrbt[1]-map_lrbt[0]
    map_h = map_lrbt[3]-map_lrbt[2]
    map_aspect = map_h/map_w
    if map_aspect < frame_aspect: scale = frame_w/map_w
    else: scale = frame_h/map_h
    xf = frame_lrbt[0]
    yf = frame_lrbt[3]
    xa = map_lrbt[0]
    ya = map_lrbt[3]
    road_x = data['road_x']
    road_y = data['road_y']
    road_l = data['roads']
    road_d = data['road_data']
    layers = data['layers']
    roads = {x.value:[] for x in layers if x.type == 'road'}
    for i,r in enumerate(road_l):
        rdi = road_d[i]
        if rdi in roads:
            roads[rdi].append(r)
    #pdf.set_line_width(data['road_width']*page_unit*scale)
    for ln,L in enumerate(layers):
        dwg = svgwrite_Drawing('svg/'+str(ln)+'.svg', size=(str(page_w)+'mm',str(page_h)+'mm'))
        dwg.viewbox(0, 0, page_w, page_h)
        lines = dwg.add(dwg.g(id=L.legend_name.replace(' ','_'), stroke=svgwrite_rgb(*[int(v*255) for v in L.rgb]),
                              stroke_width = 0.01))#data['road_width']*page_unit*scale))
        #pdf.set_draw_color(*[int(v*255) for v in L.rgb])
        if L.type == 'road':
            for r,road in enumerate(roads[L.value]):
                if r%1000==0: print(r)
                path = [(road_x[n],road_y[n]) for n in road]
                draw_svg_path(dwg, lines, path, scale, xa, xf, ya, yf)
        l,r,b,t = map_lrbt
        draw_svg_path(dwg, lines, [(l,b),(l,t),(r,t),(r,b),(l,b)], scale, xa, xf, ya, yf)
        dwg.save()




def draw_pdf_polyline(pdf, path, scale, xa, xf, ya, yf, style='-'):
    if len(path) > 1:
        for n in Qr(len(path)-1):
            x1 = xf+(path[n][0]-xa)*scale
            y1 = yf+(ya-path[n][1])*scale
            x2 = xf+(path[n+1][0]-xa)*scale
            y2 = yf+(ya-path[n+1][1])*scale
##        if style == '.':
##            pdf.dashed_line(x1,y1,x2,y2, scale*5, scale*5)#scale*5 for dash lengths
##            print('.')
##        else:
        pdf.line(x1,y1,x2,y2)
def draw_pdf(data, out_file='map.pdf'):
    page_w = 297
    page_h = 210
    page_unit = 0.001
    frame_lrbt = [25, page_w-25, page_h-25, 25]
    frame_w = frame_lrbt[1]-frame_lrbt[0]
    frame_h = frame_lrbt[2]-frame_lrbt[3]
    frame_aspect = frame_h/frame_w
    map_lrbt =[data['lrbt_view'][n]+[-1000,1000,-1000,1000][n] for n in Qr(4)]
    map_w = map_lrbt[1]-map_lrbt[0]
    map_h = map_lrbt[3]-map_lrbt[2]
    map_aspect = map_h/map_w
    if map_aspect < frame_aspect: scale = frame_w/map_w
    else: scale = frame_h/map_h
    xf = frame_lrbt[0]
    yf = frame_lrbt[3]
    xa = map_lrbt[0]
    ya = map_lrbt[3]
    road_x = data['road_x']
    road_y = data['road_y']
    road_l = data['roads']
    road_d = data['road_data']
    layers = data['layers'][::-1]
    roads = {x.value:[] for x in layers if x.type == 'road'}
    for i,r in enumerate(road_l):
        rdi = road_d[i]
        if rdi in roads:
            roads[rdi].append(r)
    pdf = fpdf_FPDF('L','mm','A4')
    pdf.add_page()
    pdf.set_line_width(data['road_width']*page_unit*scale)
    for l,L in enumerate(layers):
        pdf.set_draw_color(*[int(v*255) for v in L.rgb])
        if L.type == 'road':
            for r,road in enumerate(roads[L.value]):
                if r%1000==0: print(r)
                path = [(road_x[n],road_y[n]) for n in road]
                draw_pdf_polyline(pdf, path, scale, xa, xf, ya, yf)
        if L.type == 'road.':
            for r,road in enumerate(roads[L.value]):
                pass#...
    if 1: # frame
        pdf.set_draw_color(0,0,0)
        l,r,b,t = map_lrbt
        draw_pdf_polyline(pdf, [(l,b),(l,t),(r,t),(r,b),(l,b)], scale, xa, xf, ya, yf)
    pdf.output(out_file, 'F')








if __name__ == '__main__':

    #OSM_DAT = load_data('outputs/BC/BC Kelowna.dat','pickle')
    #OSM_DAT = load_data('outputs/ON/ON Ottawa.dat','pickle')
    #OSM_DAT = load_data('outputs/ON/ON Toronto.dat','pickle')
    OSM_DAT = load_data('outputs/ON/Hamilton Ward 2.dat','pickle')
    MAP = osmmap.OSM_Map(OSM_DAT)
    
    #drawmap.draw(MAP, DRAW_MAP ='js', Roads = osmfilter.drawBO_LTS)


    layers = []
    layers.append(Layer('road', 1, (0.0,0.5,1.0)   , 'OSM LTS 1', 0))
    layers.append(Layer('road', 2, (0.0,0.75,0.0)  , 'OSM LTS 2', 0))
    layers.append(Layer('road', 3, (1,0.625,0.0)   , 'OSM LTS 3', 0))
    layers.append(Layer('road', 4, (1.0,0.0,1/16)   , 'OSM LTS 4', 0))
    layers.append(Layer('road', 0, (0.76,0.76,0.96), 'Dismount', 0))
    #layers.append(Layer('road', '?', (0.5,0,0.5),    'Unknown', 9))    
    drawmap.draw(MAP, DRAW_MAP ='svg_layers', Roads = osmfilter.LTS_BikeOttawa, Layers=layers)
