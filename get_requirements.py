initial = ['Cython', 'Descartes', 'Fiona', 'Polygon', 'PyOpenGL', 'Rtree', 'Shapely', 'fpdf', 'fuzzywuzzy', 'geopandas', 'hickle', 'imageio', 'matplotlib', 'networkx', 'numba', 'numpy', 'odfpy', 'osmium', 'osmnx', 'pandas', 'pyglet', 'pykml', 'pyopencl', 'pyproj', 'pyshp', 'requests', 'scipy', 'seaborn', 'siphash24', 'svgwrite', 'tifffile', 'tripy', 'vispy', 'xmltodict', 'zipfile2']
initial = set(i.lower() for i in initial)

import os
files = [f for f in os.listdir(os.getcwd()) if f.endswith('.py')]
py = [os.path.basename(f)[:-3].lower() for f in files]


missing = set()
for FN in files:
    with open(FN, 'r', encoding="utf-8") as file:
        for line in file:
            s = line.strip()
            if s.startswith('from '):
                s = s[5:]
                s = s.split()[0]
                s = s.split('.')[0].lower()
                if s not in initial and s not in py: missing.add(s)

print(missing)
