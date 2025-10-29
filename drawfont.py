from utils import *
from qcommands import *
from vispy.gloo import Program, FrameBuffer, RenderBuffer, clear, set_viewport, set_state
from vispy.app import Canvas as app_Canvas


TEXT_VERT_SHADER = """
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;
varying vec4 v_rgba;
uniform vec4 u_rgba;

void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
    v_rgba = u_rgba;
}
"""

TEXT_FRAG_SHADER = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;
varying vec4 v_rgba;

void main()
{
    float v = texture2D(u_texture, v_texcoord)[0];
    gl_FragColor = vec4(v_rgba[0], v_rgba[1], v_rgba[2], v_rgba[3]*(1.0-v));
}
"""



ASCII_FONT_BINARY = """
256 240 0 773 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 74 6 10 6 26 6 10 6 1546 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 26 6 10 6 10 6 10 6 10 6 1100 2 254 2 12 3 191 2 12 2 2 2 11 2 1 2 11 4 10 2 1 2 13 3 14 2 15 2 12 2 97 2 27 4 11 2 2 2 11 2 1 2 10 2 2 2 9 2 1 2 1 1 10 2 1 2 13 2 14 2 14 2 96 2 27 4 11 2 2 2 10 7 9 2 14 3 1 2 10 2 1 2 13 2 14 2 14 2 13 2 1 2 12 2 63 2 28 4 28 2 1 2 11 2 16 2 12 3 29 2 16 2 13 3 13 2 63 2 29 2 29 2 1 2 12 2 14 2 12 2 31 2 16 2 11 7 9 6 26 6 28 2 30 2 29 2 1 2 13 2 12 2 13 2 1 4 26 2 16 2 13 3 13 2 62 2 60 7 13 2 10 2 1 3 10 2 2 2 27 2 16 2 12 2 1 2 12 2 61 2 31 2 29 2 1 2 10 2 2 2 10 1 1 2 1 2 9 2 2 2 27 2 16 2 45 3 29 3 12 2 31 2 29 2 1 2 11 4 13 2 1 2 10 3 1 2 27 2 14 2 46 3 29 3 11 2 80 2 15 3 44 2 14 2 47 2 43 2 80 2 63 2 12 2 47 2 1086 4 13 2 12 4 12 4 12 2 13 6 12 3 11 6 11 4 12 4 47 2 26 2 15 4 12 2 2 2 11 3 11 2 2 2 10 2 2 2 11 2 13 2 16 2 16 2 10 2 2 2 10 2 2 2 45 2 28 2 13 2 2 2 11 2 1 3 9 5 11 2 2 2 10 2 2 2 11 2 1 2 10 2 15 2 16 2 11 2 2 2 10 2 2 2 12 3 13 3 13 2 30 2 12 2 2 2 11 2 1 3 12 2 15 2 14 2 11 2 1 2 10 2 14 5 14 2 11 3 1 2 10 2 2 2 12 3 13 3 12 2 13 6 13 2 14 2 12 2 2 2 12 2 14 2 13 3 12 2 1 2 10 5 11 2 2 2 12 2 13 4 11 2 2 2 42 2 34 2 12 2 13 3 1 2 12 2 13 2 16 2 10 2 2 2 14 2 10 2 2 2 12 2 12 2 1 3 11 5 43 2 13 6 13 2 13 2 13 3 1 2 12 2 12 2 13 2 2 2 10 7 13 2 10 2 2 2 11 2 13 2 2 2 13 2 45 2 30 2 29 2 2 2 12 2 11 2 14 2 2 2 14 2 13 2 11 2 2 2 11 2 13 2 2 2 12 2 14 3 13 3 14 2 28 2 15 2 14 4 13 2 11 6 11 4 15 2 10 4 13 4 12 2 14 4 12 3 14 3 13 3 15 2 26 2 16 2 191 2 253 2 1100 6 12 2 12 5 12 4 11 4 12 6 10 6 11 4 11 2 2 2 11 4 15 2 10 2 2 2 10 2 14 2 3 2 9 2 3 2 10 4 10 2 4 2 10 4 11 2 2 2 10 2 2 2 10 2 1 2 11 2 14 2 14 2 2 2 10 2 2 2 12 2 16 2 10 2 2 2 10 2 14 2 3 2 9 2 3 2 9 2 2 2 9 2 4 2 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 14 2 14 2 2 2 10 2 2 2 12 2 16 2 10 2 1 2 11 2 14 3 1 3 9 3 2 2 9 2 2 2 9 2 2 4 9 2 2 2 10 2 2 2 10 2 14 2 2 2 10 2 14 2 14 2 14 2 2 2 12 2 16 2 10 2 1 2 11 2 14 2 1 1 1 2 9 4 1 2 9 2 2 2 9 2 1 2 1 2 9 2 2 2 10 5 11 2 14 2 2 2 10 5 11 5 11 2 14 6 12 2 16 2 10 4 12 2 14 2 1 1 1 2 9 2 1 4 9 2 2 2 9 2 1 2 1 2 9 6 10 2 2 2 10 2 14 2 2 2 10 2 14 2 14 2 1 3 10 2 2 2 12 2 16 2 10 2 1 2 11 2 14 2 1 1 1 2 9 2 2 3 9 2 2 2 9 2 2 4 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 14 2 14 2 2 2 10 2 2 2 12 2 12 2 2 2 10 2 1 2 11 2 14 2 3 2 9 2 3 2 9 2 2 2 9 2 15 2 2 2 10 2 2 2 10 2 2 2 10 2 1 2 11 2 14 2 14 2 2 2 10 2 2 2 12 2 12 2 2 2 10 2 2 2 10 2 14 2 3 2 9 2 3 2 9 2 2 2 10 7 9 2 2 2 10 5 12 4 11 4 12 6 10 2 15 5 10 2 2 2 11 4 12 4 11 2 2 2 10 6 10 2 3 2 9 2 3 2 10 4 1261 2 253 4 27 5 12 4 11 5 12 4 11 6 10 2 2 2 10 2 2 2 10 2 3 2 9 2 2 2 10 2 2 2 10 6 11 4 11 2 15 4 11 2 2 2 26 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 12 2 2 2 10 2 2 2 10 2 3 2 9 2 2 2 10 2 2 2 14 2 11 2 13 2 17 2 43 2 2 2 10 2 2 2 10 2 2 2 10 2 16 2 12 2 2 2 10 2 2 2 10 2 3 2 10 2 1 1 11 2 2 2 14 2 11 2 14 2 16 2 43 2 2 2 10 2 2 2 10 2 2 2 11 2 15 2 12 2 2 2 10 2 2 2 10 2 1 1 1 2 11 2 12 2 2 2 13 2 12 2 14 2 16 2 43 5 11 2 2 2 10 5 13 2 14 2 12 2 2 2 10 2 2 2 10 2 1 1 1 2 11 2 13 4 13 2 13 2 15 2 15 2 43 2 14 2 2 2 10 2 1 2 14 2 13 2 12 2 2 2 10 2 2 2 10 2 1 1 1 2 10 1 1 2 13 2 13 2 14 2 15 2 15 2 43 2 14 2 2 2 10 2 2 2 14 2 12 2 12 2 2 2 10 2 2 2 11 2 1 2 10 2 2 2 12 2 12 2 15 2 16 2 14 2 43 2 14 2 2 2 10 2 2 2 10 2 2 2 12 2 12 2 2 2 11 4 12 2 1 2 10 2 2 2 12 2 12 2 15 2 16 2 14 2 43 2 15 4 11 2 2 2 11 4 13 2 13 4 13 2 13 2 1 2 10 2 2 2 12 2 12 6 11 2 17 2 13 2 62 2 156 2 17 2 13 2 63 2 155 2 32 2 26 8 186 4 28 4 300 3 254 2 142 2 15 2 94 2 27 2 34 2 28 4 26 2 16 2 15 2 11 2 14 4 92 2 34 2 27 2 29 2 46 2 16 2 77 4 11 5 12 4 12 5 11 4 12 2 14 5 10 5 11 4 13 4 11 2 2 2 12 2 12 6 10 5 12 4 31 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 11 2 13 2 2 2 10 2 2 2 12 2 15 2 11 2 2 2 12 2 12 2 1 1 1 2 9 2 2 2 10 2 2 2 30 2 10 2 2 2 10 2 14 2 2 2 10 2 2 2 10 6 10 2 2 2 10 2 2 2 12 2 15 2 11 2 1 2 13 2 12 2 1 1 1 2 9 2 2 2 10 2 2 2 27 5 10 2 2 2 10 2 14 2 2 2 10 6 11 2 13 2 2 2 10 2 2 2 12 2 15 2 11 4 14 2 12 2 1 1 1 2 9 2 2 2 10 2 2 2 26 2 2 2 10 2 2 2 10 2 14 2 2 2 10 2 15 2 13 2 2 2 10 2 2 2 12 2 15 2 11 2 1 2 13 2 12 2 1 1 1 2 9 2 2 2 10 2 2 2 26 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 15 2 13 2 2 2 10 2 2 2 12 2 15 2 11 2 2 2 12 2 12 2 1 1 1 2 9 2 2 2 10 2 2 2 27 5 10 5 12 4 12 5 11 4 12 2 14 5 10 2 2 2 10 6 13 2 11 2 2 2 10 6 10 2 3 2 9 2 2 2 11 4 127 2 45 2 207 2 45 2 203 5 43 4 925 2 112 2 13 2 13 2 13 3 3 1 9 6 75 2 111 2 14 2 14 2 11 2 1 2 1 2 9 6 10 5 12 5 10 2 2 2 11 5 10 6 10 2 2 2 10 2 2 2 10 2 3 2 9 2 2 2 10 2 2 2 10 6 12 2 14 2 14 2 11 1 3 3 10 6 10 2 2 2 10 2 2 2 10 2 1 3 10 2 15 2 13 2 2 2 10 2 2 2 10 2 1 1 1 2 9 2 2 2 10 2 2 2 14 2 12 2 14 2 14 2 28 6 10 2 2 2 10 2 2 2 10 3 13 2 15 2 13 2 2 2 10 2 2 2 10 2 1 1 1 2 10 4 11 2 2 2 13 2 12 2 15 2 15 2 27 6 10 2 2 2 10 2 2 2 10 2 15 4 12 2 13 2 2 2 10 2 2 2 10 2 1 1 1 2 11 2 12 2 2 2 12 2 12 2 16 2 16 2 26 6 10 2 2 2 10 2 2 2 10 2 18 2 11 2 13 2 2 2 10 2 2 2 10 2 1 1 1 2 10 4 11 2 2 2 11 2 14 2 15 2 15 2 27 6 10 2 2 2 10 2 2 2 10 2 18 2 11 2 13 2 2 2 11 4 12 2 1 2 10 2 2 2 10 2 2 2 10 2 16 2 14 2 14 2 28 6 10 5 12 5 10 2 14 5 13 4 11 5 12 2 13 2 1 2 10 2 2 2 11 4 11 6 12 2 14 2 14 2 28 6 10 2 18 2 125 2 29 2 14 2 14 2 44 2 18 2 124 2 31 2 13 2 13 2 45 2 18 2 121 4 47 2 830 3 11 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 11 2 2 1 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 11 1 14 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 5 11 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 11 1 14 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 5 11 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 11 1 14 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 11 2 2 1 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 12 3 11 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 1546 6 13 2 13 3 11 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 12 2 14 3 11 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 12 3 14 2 11 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 12 3 13 2 12 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 42 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 42 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 42 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 42 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 42 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 10 6 1017 8 393 2 2 2 156 2 13 4 27 2 2 2 12 2 13 4 11 2 2 2 10 6 11 4 59 6 60 2 12 2 2 2 10 2 2 2 10 2 2 2 12 2 12 2 2 2 25 2 4 2 13 2 57 2 4 2 58 4 11 2 15 4 11 2 2 2 12 2 12 2 29 1 2 2 2 1 10 5 57 1 1 3 2 1 43 2 12 2 2 2 10 2 14 2 2 2 11 4 13 2 13 3 27 1 1 1 2 1 1 1 9 2 2 2 57 1 1 1 2 1 1 1 43 2 12 2 13 6 11 2 2 2 10 6 12 2 13 4 26 1 1 1 4 1 10 5 11 2 2 2 9 6 10 6 9 1 1 1 2 1 1 1 57 2 14 2 15 4 13 2 28 2 2 2 25 1 1 1 2 1 1 1 25 2 2 2 14 2 25 1 1 3 2 1 43 2 12 2 2 2 10 2 14 2 2 2 10 6 26 2 2 2 25 1 2 2 2 1 9 6 9 2 2 2 15 2 25 1 1 1 2 1 1 1 43 2 13 4 10 2 33 2 14 2 13 4 26 2 4 2 25 2 2 2 41 2 4 2 42 4 13 2 11 7 28 2 14 2 14 3 27 6 27 2 2 2 41 6 43 4 13 2 62 2 16 2 155 4 77 2 12 2 2 2 156 2 78 2 13 4 461 3 188 4 61 2 124 2 14 2 13 3 30 2 2 2 27 3 13 3 13 2 31 5 43 2 13 4 26 3 13 3 15 2 29 2 2 2 29 2 14 2 44 5 43 3 12 2 2 2 26 2 3 2 9 2 3 2 9 2 3 2 26 4 13 2 14 2 14 2 28 2 2 2 10 6 44 2 12 2 2 2 26 2 2 2 10 2 2 2 11 2 1 2 44 2 13 2 16 2 27 2 2 2 10 6 12 3 29 2 12 2 2 2 26 2 1 2 11 2 1 2 10 3 1 2 13 2 28 6 11 4 12 3 28 2 2 2 10 6 12 3 29 2 13 4 10 2 2 2 13 2 14 2 14 2 14 2 30 2 60 2 2 2 11 5 74 2 2 2 11 2 1 3 10 5 11 2 1 3 43 2 60 2 2 2 12 4 58 6 11 2 2 2 9 2 1 4 9 2 3 2 9 2 1 4 11 2 92 2 2 2 14 2 74 2 2 2 9 2 1 2 1 2 8 2 3 2 9 2 1 2 1 2 11 2 28 6 58 4 1 2 13 2 73 2 2 2 13 5 12 2 13 5 10 2 93 2 18 2 28 2 65 2 12 4 14 2 9 2 2 2 90 2 18 2 29 2 107 2 2 2 89 2 19 2 27 3 109 4 12 2 16 2 12 4 12 3 1 2 9 2 2 2 11 4 44 2 16 2 12 4 11 2 2 2 11 2 16 2 12 4 11 2 2 2 12 2 14 2 12 2 2 2 10 2 1 3 10 2 2 2 10 2 2 2 44 2 14 2 12 2 2 2 10 2 2 2 12 2 14 2 12 2 2 2 10 2 2 2 91 4 173 2 14 2 14 2 14 2 14 2 14 2 15 4 10 4 11 6 10 6 10 6 10 6 11 4 12 4 12 4 12 4 12 4 12 4 12 4 12 4 12 4 12 4 13 3 11 2 2 2 10 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 11 4 11 2 2 2 10 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 1 2 11 2 14 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 1 4 9 2 14 5 11 5 11 5 11 5 13 2 14 2 14 2 14 2 12 6 10 6 10 6 10 6 10 6 10 6 10 5 11 2 14 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 1 2 11 2 2 2 10 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 1 2 11 2 2 2 10 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 1 4 10 4 11 6 10 6 10 6 10 6 11 4 12 4 12 4 12 4 125 2 255 2 252 3 157 3 1 2 10 2 16 2 12 4 12 3 1 2 9 2 2 2 43 2 16 2 12 4 11 2 2 2 13 2 59 2 1 3 12 2 14 2 12 2 2 2 10 2 1 3 10 2 2 2 44 2 14 2 12 2 2 2 10 2 2 2 12 2 300 4 12 2 3 2 10 4 12 4 12 4 12 4 12 4 28 5 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 15 3 12 2 1 2 11 2 3 2 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 1 4 1 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 14 2 1 2 11 2 2 2 10 3 2 2 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 1 3 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 5 11 2 1 2 11 2 2 2 10 4 1 2 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 11 4 11 2 1 3 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 1 2 10 4 1 2 10 2 1 4 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 12 6 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 11 4 11 2 2 2 10 2 2 2 10 2 2 2 10 2 2 3 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 11 4 11 3 1 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 3 2 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 3 1 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 12 5 11 2 2 2 10 2 1 2 11 2 3 2 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 1 4 1 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 12 2 14 2 2 2 10 4 12 2 3 2 10 4 12 4 12 4 12 4 12 4 27 5 12 4 12 4 12 4 12 4 13 2 12 2 14 2 1 2 1035 3 16 3 12 2 12 3 3 1 26 4 43 3 16 3 12 2 28 3 16 3 12 2 29 2 16 2 12 4 10 2 1 2 1 2 9 2 2 2 10 2 2 2 43 2 16 2 12 4 11 2 2 2 11 2 16 2 12 4 11 2 2 2 12 2 14 2 12 2 2 2 9 1 3 3 10 2 2 2 11 4 45 2 14 2 12 2 2 2 10 2 2 2 12 2 14 2 12 2 2 2 10 2 2 2 267 4 12 4 12 4 12 4 12 4 12 4 11 2 1 3 11 4 12 4 12 4 12 4 12 4 11 4 12 4 12 4 12 4 16 2 14 2 14 2 14 2 14 2 14 2 12 2 1 2 9 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 14 2 14 2 12 2 1 2 9 2 14 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 14 2 14 2 14 2 13 5 11 5 11 5 11 5 11 5 11 5 10 7 9 2 14 6 10 6 10 6 10 6 12 2 14 2 14 2 14 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 9 2 1 2 12 2 14 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 9 2 1 2 12 2 2 2 10 2 14 2 14 2 14 2 16 2 14 2 14 2 14 2 13 5 11 5 11 5 11 5 11 5 11 5 10 3 1 3 10 4 12 4 12 4 12 4 12 4 11 6 10 6 10 6 10 6 124 2 255 2 252 3 412 3 3 1 9 3 16 3 12 2 12 3 3 1 57 3 16 3 12 2 31 3 57 2 1 2 1 2 10 2 16 2 12 4 10 2 1 2 1 2 9 2 2 2 43 2 16 2 12 4 11 2 2 2 13 2 44 2 1 2 9 1 3 3 12 2 14 2 12 2 2 2 9 1 3 3 10 2 2 2 44 2 14 2 12 2 2 2 10 2 2 2 12 2 12 2 32 2 110 2 108 2 30 2 1 2 11 5 12 4 12 4 12 4 12 4 12 4 13 2 13 5 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 32 4 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 26 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 5 27 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 6 10 2 1 3 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 26 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 26 6 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 26 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 12 3 1 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 26 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 12 2 12 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 10 2 2 2 27 4 11 2 2 2 11 4 12 4 12 4 12 4 12 4 27 5 12 5 11 5 11 5 11 5 11 4 11 5 238 2 11 2 240 2 12 2 237 4 13 2 25
"""


                                     
def read_binary(binary):
    V = [int(v) for v in binary.split()]
    X = V.pop(0)
    Y = V.pop(0)
    A = Qu8(X*Y,None)
    b = False
    v = V.pop(0)
    for n in Qr(X*Y):
        if not v:
            b = not b
            v = V.pop(0)
        A[n] = int(b)
        v -= 1
    return A.reshape((Y,X))

def write_binary(array):
    X = array.shape[1]
    Y = array.shape[0]
    S = [X,Y]
    A = array.reshape(X*Y)
    b = False
    v = 0 
    for n in Qr(X*Y):
        if b != bool(A[n]):
            S.append(v)
            b = not b
            v = 0
        v += 1
    S.append(v)
    print(' '.join(str(s) for s in S))
    return S
            



ASCII_FONT = {'image':'ascii_font.bmp',
        'binary':ASCII_FONT_BINARY,
        'rows':16,
        'columns':16,
        'box_w':16,
        'box_h':15,
        'char_l':4,
        'char_r':12,
        'char_t':0,
        'char_b':15,
        'default_char':0}


##def resource_path(relative_path): ## BUILD
##    if hasattr(sys, '_MEIPASS'):
##        return os.path.join(sys._MEIPASS, relative_path)
##    return os.path.join(os.path.abspath("."), relative_path)
##...resource_path(P['image']))  ## BUILD








# Assumes constant width characters for now.
class Typewriter:
    def __init__(self,canvas,P):
        self.canvas = canvas
        if 'binary' in P:
            A = read_binary(P['binary'])
        elif 'image' in P:
            A = Qu8(PIL_Image.open(P['image']))
        M = {}
        for r in Qr(P['rows']):
            for c in Qr(P['columns']):
                n = r+P['rows']*c
                x = P['box_h']*c
                y = P['box_w']*r
                M[n] = A[(x+P['char_t']):(x+P['char_b']),
                         (y+P['char_l']):(y+P['char_r'])]
        self.glyphs = M
        self.glyph_width = P['char_r']-P['char_l']
        self.glyph_height = P['char_b']-P['char_t']
        self.default_char = P['default_char']
        self.max_char = P['rows']*P['columns']-1
        self.program = Program(TEXT_VERT_SHADER, TEXT_FRAG_SHADER)
    def __getitem__(self,char):
        o = ord(char)
        if o>self.max_char or o<0: o = self.default_char
        return self.glyphs[o]
    def get_text_width(self, h):
        return h * self.glyph_width/self.glyph_height
    def draw_char(self, char, rgba, x=0, y=0, h=0):
        G = self[char]
        if not h: h = G.shape[0]
        w = h * G.shape[1]/G.shape[0]
        W, H = self.canvas.size
        l = 2*x/W-1
        r = l+2*(w)/W 
        b = 2*(H-y)/H-1
        t = b-2*(h)/H 
        P = self.program
        P["a_position"] = [(l, t), (l, b), (r, t), (r, b)]
        P["a_texcoord"] = [(0, 1),(0, 0), (1, 1),(1, 0)]
        P["u_texture"] = Qf32(G)
        P["u_texture"].interpolation = 'linear'
        P["u_rgba"] = Qf32(rgba)
        P.draw('triangle_strip')
        return w,h
    def draw_typeset(self, chars, colors, x, y, rgb_a_map):
        for k,v in rgb_a_map.items():
            if len(v) == 3: rgb_a_map[k] = tuple(v)+(1.0,)
        for n,ch in enumerate(chars):
            if ch[0] != ' ':
                self.draw_char(ch[0], rgb_a_map[colors[n]], x+ch[1], y+ch[2], ch[3])
    def typeset_rainbow_list(self, text_list, max_width=10000, max_height=10000, text_height = None):
        if text_height == None: h = self.glyph_height
        else: h = text_height
        w = self.get_text_width(text_height)
        H = max_height
        W = max_width
        R = int(H/h)
        C = int(W/w)
        if R<len(text_list):
            TL = text_list[0:(R-1)]
            MORE = len(text_list)-R+1
            TL.append('+'+str(MORE)+' more...')
        else:
            TL = text_list
            MORE = 0
        for n in Qr(len(TL)): TL[n] = TL[n][0:C]
        colors = [chr(48+n)*len(TL) for n,TL in enumerate(TL)]
        if MORE: colors[-1] = chr(48)*len(colors[-1])
        colors = ''.join(colors)
        height = h*len(TL)
        width = w*max(len(t) for t in TL)
        chars = []
        for n,t in enumerate(TL):
            for m, ch in enumerate(t):
                chars.append((ch,m*w,n*h,h))
        return {'height':height,'width':width,'chars':chars,'colors':colors}
    def typeset_dictionary(self, DICT, max_width=10000, max_height=10000, indent = 0, dict_keys=None, text_height = None):
        if text_height == None: h = self.glyph_height
        else: h = text_height
        w = self.get_text_width(text_height)
        H = max_height
        W = max_width
        R = int(H/h)
        C = int((W-indent*w)/w)
        if C<1 or R<1: return None
        if not DICT or (hasattr(dict_keys,'len') and not len(dict_keys)):
            DICT = {'NO INFO':'-'}
            dict_keys = ['NO INFO']
        chars = []
        colors = []
        if not dict_keys:
            dict_keys = list(DICT.keys())
            dict_keys.sort()
        items = [(str(dk),str(DICT[dk])) for dk in dict_keys if dk in DICT]
        blocks = []
        for k,v in items:
            block = []
            lk = len(k)
            lv = len(v)
            if lk+lv<=C:#fits on one line
                b = ' '*(C-lk-lv)
                blocks.append([(k+b+v,'0'*lk+b+'1'*lv,lk)]) # add 3rd element lk for indenting on one line.
                continue
            nk = max(1,1+(lk-1)//C)
            nv = max(1,1+(lv-1)//C)
            nkv = max(1,1+(lk+lv-1)//C)
            if nkv == nk+nv:#no gain from tight packing
                for i in Qr(nk):
                    s = k[i*C:min(lk,(i+1)*C)]
                    block.append((s,'0'*len(s)))
                if nv == 1: # justify right
                    b = ' '*(C-lv)
                    block.append((b+v,b+'1'*lv))
                else: # justify left
                    for i in Qr(nv):
                        s = v[i*C:min(lv,(i+1)*C)]
                        block.append((s,'1'*len(s)))
            else:# gain from tight packing
                for i in Qr(nk-1):
                    s = k[i*C:(i+1)*C]
                    block.append((s,'0'*len(s)))
                s1 = k[(nk-1)*C:lk]
                l1 = len(s1)
                l2 = lv-(nv-1)*C
                b = ' '*(C-l1-l2)
                block.append((s1+b+v[:l2],'0'*l1+b+'1'*l2))
                v = v[l2:]
                lv = len(v)
                if lv:
                    for i in Qr(lv//C):
                        s = v[i*C:(i+1)*C]
                        block.append((s,'1'*C))                 
            blocks.append(block)
            if sum(len(b) for b in blocks)>R:
                break
        if len(items)-len(blocks): # vertical overflow:
            blocks.pop() # remove overflowing block
            if sum(len(b) for b in blocks)==R: blocks.pop() # if exact fit, remove one more block to make room for "+N more..."
            MORE = len(items)-len(blocks)
            STR = '+'+str(MORE)+' more...'
            STR = STR[:min(len(STR),C)]
            blocks.append([(STR,'0'*len(STR))])
        height = h*sum(len(b) for b in blocks)
        width = w*(indent+C)#w*(indent+max(len(b[0][0]) for b in blocks)) # to reimplement
        colors = ''.join([''.join([i[1] for i in b]) for b in blocks])
        chars = []
        x = 0
        y = 0
        for block in blocks:
            for i,I in enumerate(block):
                x = 0 + bool(i)*indent*w
                for n,ch in enumerate(I[0]):
                    if len(I) == 3 and I[2] == n: x += w*indent
                    chars.append((ch,x,y,h))
                    x += w
                y += h
        return {'height':height,'width':width,'chars':chars,'colors':colors}

class TextBox:
    def __init__(self,typewriter,typeset,rgb_a_map,inset_lrbt=(0,0,0,0),rel_pos=(0.0,0.0)):
        Qsave(self, typeset, "chars colors width height")
##        self.chars = typeset['chars']
##        self.colors = typeset['colors']
##        self.width = typeset['width']
##        self.height = typeset['height']
        Qsave(self, locals(), "typewriter rgb_a_map rel_pos inset_lrbt")
##        self.typewriter = typewriter
##        self.rgb_a_map = rgb_a_map
##        self.rel_pos = rel_pos
##        self.inset_lrbt = inset_lrbt
    def draw(self,lrbt_container):
        l,r,b,t = [lrbt_container[n]+self.inset_lrbt[n] for n in Qr(4)]
        l -= 1 #pixel correction
        t -= 1 #pixel correction
        w = self.width
        h = self.height
        px, py = self.rel_pos
        x = l + (r-l-w)*px
        y = t + (b-t-h)*py
        self.typewriter.draw_typeset(self.chars,self.colors,x,y,self.rgb_a_map)
    

        
if __name__ == '__main__':
    A = Qu8(PIL_Image.open(ASCII_FONT['image']))
    #S = write_binary(A)
    B = read_binary(ASCII_FONT_BINARY)
    print(sum(sum(A==B)),B.shape[0]*B.shape[1])
