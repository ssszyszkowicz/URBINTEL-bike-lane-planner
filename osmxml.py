### Design an ultra-fast parser to read .osm.pbf - later
# The file format is detailed on the OSM Wiki
# Get someone to do the C-code?
### Design:
# OpenCL: parse each frame with one worker.
# Start by parsing ways:
#   -
# Then nodes:
#   - We already know the desired number of nodes
#   - We need a HashLoL with short lists of node_osm_id's for each %10**7 (or so)
#   - The above HashLoL shares the shape with a HashLoL for lats, and one for lons







#from imposm.parser import OSMParser

from utils import *
from qcommands import *
from fileformats import *
from geometry2 import buffer_polygon, bbox, boundary_sPolygons, lrbt as geometry2_lrbt
from earth import _deg2m
#from google.protobuf.internal.decoder import _DecodeVarint32
from xml.dom.minidom import parse, parseString
from lxml.etree import iterparse as lxml_etree_iterparse
from osmium import SimpleHandler as osmium_SimpleHandler
from osmium.io import Reader as osmium_io_Reader
from osmium.osm import osm_entity_bits as o_osm_osm_entity_bits




#################### PBF-Compatible CODE ###########################

def osm_props_str2dict(s):
    out = {}
    if not s: return out
    parts = s[1:].split(s[0])
    for n in Qr(len(parts)//2):
        k,v = parts[2*n],parts[2*n+1]
        if k == '_nodes':
            if set(v)-set('1234567890 '): return {} # invalid node list
            v = tuple(int(n) for n in v.split())
        out[k] = v
    return out


# See also algorithm in OSMAccumulator.way()
def osm_props_dict2str(d):
    s = ''
    for k,v in d.items(): s += chr(0)+str(k)+chr(0)+str(v)
    return s

class OSMData:
    def __init__(self, centre=None, osm_file=None, way_types=[[]], way_types_polygon=set(), boundary=None, node_frame=None, way_filter=None):
        self.centre = centre
        self.node_x = {}
        self.node_y = {}
        self.way_lines = {}
        self.tynames = set([t[0] for t in way_types if len(t)])
        self.node_highway_tags = {}
        self.len_m = {}
        self.seg_len_m = {}
        self.props_encoder = osm_props_dict2str
        self.props_decoder = osm_props_str2dict
        if not osm_file:
            self.error = True
            return None
        # Access header info:
        header = osmium_io_Reader(osm_file, o_osm_osm_entity_bits.NOTHING).header()
        print("Bbox:", header.box())
        print("History file:", header.has_multiple_object_versions)
        print("Replication base URL:", header.get("osmosis_replication_base_url", "<none>"))
        print("Replication sequence number:", header.get("osmosis_replication_sequence_number", "<none>"))
        print("Replication timestamp:", header.get("osmosis_replication_timestamp", "<none>"))
        B_SIZE = os_path_getsize(osm_file)
        EST_N_NODES = None
        #(Max: CA @ 817 MB_HD would allocate 1867 MB_RAM for the HugeDict) + again that much max for lon/lat
        # ON(CAN-max) @ 700 MB_HD -> alloc 1600 MB_RAM HugeDict; +1600 lon/lat; +
        if osm_file.endswith('pbf'): EST_N_NODES = B_SIZE//7
        if osm_file.endswith('osm'): EST_N_NODES = B_SIZE//150
        A = OSMAccumulator(way_types, way_types_polygon, boundary, node_frame, EST_N_NODES, way_filter) # Could use population info here!
        A.apply_file(osm_file) ### TO_DO: add safety-checking here.
        self.error = False #hopefully
        print('OSM parsing done.')
        for tyname in self.tynames:
            sub_nodes = {}
            for s in [tuple(s) for s in merge_sets([set([k]+v) for k,v in A.NODE_ABSORB[tyname].items()])]:
                for n in s[1:]: sub_nodes[n] = s[0]
            ixs = [sub_nodes[n] if n in sub_nodes else n for n in A.USED_NODES[tyname].annihilate_to_set()]
            S = set(ixs)
            del A.USED_NODES[tyname]
            nodes = IndexDict(ixs,'uint32')
            if tyname == 'highway':
                for k,v in A.NODE_HIGHWAY_TAGS.items():
                    self.node_highway_tags[k] = Qu32(tuple(set(tuple(nodes[sub_nodes[ix]] if ix in sub_nodes else nodes[ix]\
                                                                        for ix in v if ix in S))))
            lines_offsets = np_concatenate((np_zeros(1,dtype='uint32'),
                np_cumsum([d.size for d in A.WAY_LINES[tyname].data],dtype='uint32')))
            lines_data = Qu32(tuple(nodes[sub_nodes[ix]] if ix in sub_nodes else nodes[ix]\
                                        for ix in A.WAY_LINES[tyname].array()))
            lol = LoL()
            self.way_lines[tyname] = lol.load(lines_data, lines_offsets)
            node_x, node_y = _deg2m(centre,
                Qf64(tuple(A.NODE_LON[ix] for ix in ixs)),
                Qf64(tuple(A.NODE_LAT[ix] for ix in ixs)))
            self.node_x[tyname] = node_x
            self.node_y[tyname] = node_y
            if tyname not in way_types_polygon:
                x = node_x[lines_data]
                y = node_y[lines_data]
                o = lines_offsets
                print('seg_len_m')
                # careful with offsets!
                self.seg_len_m[tyname] = LoLf32(tuple(tuple(((x[n]-x[n+1])**2+(y[n]-y[n+1])**2)**0.5\
                    for n in Qr(o[k],o[k+1]-1)) for k in Qr(len(o)-1)))
                print('len_m')
                S = self.seg_len_m[tyname]
                self.len_m[tyname] = Qf32(sum(S[n]) for n in Qr(S))
                #np_array(tuple(sum(d[o[n]:o[n+1]-1]) for n in Qr(self.way_lines[tyname])),'float32')
        self.way_props = A.WAY_PROPS
        #if relations, do something here...
        del A
    def import_ui(self, ui_dict):
        self.node_x = {}
        self.node_y = {}
        self.way_props = {}
        self.way_lines = {}
        self.len_m = {}
        self.seg_len_m = {}
        self.tynames = set(ui_dict['tynames'].split(chr(0)))
        self.centre = tuple(float(c) for c in ui_dict['centre'])
        self.error = False
        way_big_string = {}
        way_props_offsets = {}
        way_lines_data = {}
        way_lines_offsets = {}
        seg_len_m_data = {}
        seg_len_m_offsets = {}
        self.node_highway_tags = {}
        for k,v in ui_dict.items():
            ###print(k)
            if ':' not in k: continue
            i = k.index(':')
            var,tyname = k[:i],k[i+1:]
            if var == 'node_x': self.node_x[tyname] = v
            if var == 'node_y': self.node_y[tyname] = v
            if var == 'len_m': self.len_m[tyname] = v
            if var == 'seg_len_m': seg_len_m_data[tyname] = v
            if var == 'seg_len_m_off': seg_len_m_offsets[tyname] = v
            if var == 'way_props': way_big_string[tyname] = v
            if var == 'way_props_off': way_props_offsets[tyname] = v
            if var == 'way_lines': way_lines_data[tyname] = v
            if var == 'way_lines_off': way_lines_offsets[tyname] = v
            if var == 'node_highway_tags': self.node_highway_tags[tyname] = v
        print(self.tynames)  
        for tyname in self.tynames:
            L = []
            S = way_big_string[tyname]
            O = way_props_offsets[tyname]
            for n, o in enumerate(O):
                if n == 0: L.append(S[:o])
                else: L.append(S[O[n-1]:o])
            del way_big_string[tyname]
            del way_props_offsets[tyname]
            self.way_props[tyname] = tuple(L)
            lol = LoL()
            self.way_lines[tyname] = lol.load(way_lines_data[tyname], way_lines_offsets[tyname])
            if tyname in seg_len_m_data:
                lol = LoL()
                self.seg_len_m[tyname] = lol.load(seg_len_m_data[tyname], seg_len_m_offsets[tyname])
        return self
    def export_ui(self):
        if self.error: return False
        out = {}
        for tyname in self.tynames:
            out['node_x:'+tyname] = self.node_x[tyname]
            out['node_y:'+tyname] = self.node_y[tyname]
            out['len_m:'+tyname] = self.len_m[tyname]
            out['seg_len_m:'+tyname] = self.seg_len_m[tyname].data
            out['seg_len_m_off:'+tyname] = self.seg_len_m[tyname].offsets
            out['way_props:'+tyname] = self.way_props[tyname]
            out['way_props_off:'+tyname] = np_cumsum(tuple(len(s) for s in self.way_props[tyname]),dtype='uint32')
            out['way_lines:'+tyname] = self.way_lines[tyname].data
            out['way_lines_off:'+tyname] = self.way_lines[tyname].offsets
        for ht in self.node_highway_tags:
            out['node_highway_tags:'+ht] = self.node_highway_tags[ht]
        out['centre'] = Qf64(self.centre)
        out['tynames'] = chr(0).join(list(self.tynames))
        return out
    

def intersects(BOUNDARY, lons, lats, force_line=False):
    if not BOUNDARY: return True
    N = len(lons)
    if N == 1: Geom = Point((lons[0], lats[0]))
    else:
        if N <= 3 or force_line:
            Geom = LineString((lons[n], lats[n]) for n in Qr(N))
        else:
            Geom = sPolygon((lons[n], lats[n]) for n in Qr(N-1))
    for B in BOUNDARY:
        if Geom.intersects(B): return True
    return False


class OSMAccumulator(osmium_SimpleHandler):
    def __init__(self, way_types, way_types_polygon=set(), boundary=None, node_frame=None, EST_N_NODES = None, way_filter=None):
        osmium_SimpleHandler.__init__(self)
        self.NODE_HIGHWAY_TAGS = {}
        self.NODE_ABSORB = {}
        self.way_filter = way_filter
        self.way_types = way_types
        self.way_types_polygon = way_types_polygon
        if boundary: self.BOUNDARY = boundary_sPolygons(boundary)
        else: self.BOUNDARY = None
        self.node_frame = node_frame
        if node_frame: self.L,self.R,self.B,self.T = node_frame
        #self.RELATIONS = {}
        self.WAY_PROPS = {}
        self.WAY_LINES = {}
        self.USED_NODES = {}
        #self.USED_WAYS = {}#;self.USED_RELATIONS = {};self.WAY_N={}
        for tyname in [t[0] for t in self.way_types]:#+['$RELATION$']:
            self.NODE_ABSORB[tyname] = {}
            self.WAY_PROPS[tyname] = []
            self.WAY_LINES[tyname] = List(dtype='uint32')
            self.USED_NODES[tyname] = HugeVec('uint32', 10**6)
            #self.WAY_N[ty] = HugeDict('uint32', 10**6, 12)
        #for ty in way_types_polygon:
            #self.RELATIONS[ty] = {}
            #self.USED_WAYS[ty] = set()
            #self.USED_RELATIONS[ty] = set()
        #self.USED_WAYS['$RELATION$'] = set()
        self.NODE_LON = HugeVec('float64', 10**6)
        self.NODE_LAT = HugeVec('float64', 10**6)
        if not EST_N_NODES: EST_N_NODES = 10**7
        self.NODE_N = HugeDict('uint64', EST_N_NODES//10, 20) # Gives an expansion factor of 2, a Poisson prob(overflow)~=0.001
        self.parsed_nodes = 0
        self.parsed_ways = 0
    def node(self, node):
        if self.parsed_nodes%1000000==0: print('Parsed',self.parsed_nodes,'nodes...')
        self.parsed_nodes += 1
        lon = node.location.lon
        lat = node.location.lat
        if self.node_frame:
            if lon<self.L or lon>self.R or lat<self.B or lat>self.T: return None
        nix = len(self.NODE_N)
        if 'highway' in node.tags: # do this, otherwise for block is very slow
            for t in node.tags:
                if t.k == 'highway':
                    v = t.v
                    if v in self.NODE_HIGHWAY_TAGS: self.NODE_HIGHWAY_TAGS[v].append(nix)
                    else: self.NODE_HIGHWAY_TAGS[v] = [nix]
        self.NODE_N[node.id] = nix # this increments len(self.NODE_N)
        self.NODE_LON(lon)
        self.NODE_LAT(lat)
    def way(self, way):
        if self.parsed_ways%100000==0: print('Parsed',self.parsed_ways,'ways...')
        self.parsed_ways += 1
        PASS = False
        if self.way_filter:
            if not self.way_filter(way.tags): return 0
        for ty in self.way_types:
            for t in ty:
                if t in way.tags:
                    PASS = True
                    break
            if PASS:
                tyname = ty[0]
                #if len(self.WAYS[tyname])%10000 == 0: print('Found',len(self.WAYS[tyname]),tyname+'s...')
                ixs = tuple(self.NODE_N[nd.ref] for nd in way.nodes)
                if None in ixs: break # some nodes have not been registered.
                ixs = tuple(int(ix) for ix in ixs)
                if self.BOUNDARY: # This only works assuming all ways appear after all nodes.
                    NODE_LON = self.NODE_LON
                    NODE_LAT = self.NODE_LAT
                    lons = tuple(NODE_LON[ix] for ix in ixs)
                    lats = tuple(NODE_LAT[ix] for ix in ixs)
                    if not intersects(self.BOUNDARY, lons, lats,
                        force_line=(ixs[0]!=ixs[-1] or tyname not in self.way_types_polygon)):
                        break
                if tyname in self.way_types_polygon: ixs = ixs[:-1]
                line = []
                ixlast = -1
                x = self.NODE_LON
                y = self.NODE_LAT
                for ix in ixs:
                    if ix == ixlast: continue
                    if ixlast>-1:
                        if x[ix] == x[ixlast] and y[ix] == y[ixlast]:
                            if ixlast in self.NODE_ABSORB[tyname]: self.NODE_ABSORB[tyname][ixlast].append(ix)
                            else: self.NODE_ABSORB[tyname][ixlast] = [ix]
                            continue
                    line.append(ix)
                    ixlast = ix
                if len(line)>1:
                    for ix in ixs: self.USED_NODES[tyname](ix)
                    #W = ''
                    #for t in way.tags: W += chr(0)+t.k+chr(0)+t.v
                    W = ''.join(chr(0)+t.k+chr(0)+t.v for t in way.tags)
                    W += chr(0)+'osm_id'+chr(0)+str(way.id) #  
                    # TO_DO: add OSM_ID as porperty if needed for relations
                    self.WAY_PROPS[tyname].append(W)
                    self.WAY_LINES[tyname].extend(line)
                    #self.WAY_N[ty][way.id] = len(self.WAY_N[ty])
                break # consider types in order, no double-categories
##        if not PASS and len(nodes)>3 and nodes[0]==nodes[-1]: # Capture potential rings for relation topologies
##            if intersects(self.BOUNDARY, self.NODE_N,
##                          self.NODE_LON, self.NODE_LAT,
##                          way.nodes, force_line=False):
##                self.WAYS['$RELATION$'].append('{"nodes":'+str(nodes)+'}')
##                self.WAY_N['$RELATION$'][way.id] = len(self.WAY_N['$RELATION$'])
##    def relation(self, relation):
##        pass










































    
##    data = open(FILE, "rb").read()  
##    decoder = _DecodeVarint32
##
##    # Flush header:
##    next_pos,pos = 0,0
##    while pos+next_pos < len(data):
##        pos += next_pos
##        next_pos, pos = decoder(data, pos)
##        print(next_pos,pos)
##    # Parse:
    
    







# Cities can be downloaded at: http://metro.teczno.com/
# Idea: write a script with Beautiful_Soup to download, extract and setup cities automatically

# NODES:
# <node id="914780145" lat="48.8848239" lon="2.3401970" (ETC...) />

# BUILDINGS:
##<way id="77807647" user="didier2020" uid="300459" visible="true" version="2" changeset="9198478" timestamp="2011-09-03T08:04:16Z">
##  <nd ref="914780145"/>
##  <nd ref="914791543"/>
##  <nd ref="914792643"/>
##  <nd ref="914798611"/>
##  <nd ref="914780145"/>
##  <tag k="building" v="yes"/>
##  <tag k="source" v="cadastre-dgi-fr source : Direction Générale des Impôts - Cadastre. Mise à jour : 2010"/>
## </way>

## MULTIPOLYGON BUILDINGS:
##  <relation id="2449758" version="1" timestamp="2012-10-03T11:55:30Z" uid="456113" user="eric_G" changeset="13346349">
##    <member type="way" ref="53602683" role="outer"/>
##    <member type="way" ref="53602680" role="inner"/>
##    [...]
##    <member type="way" ref="53602689" role="inner"/>
##    <tag k="building" v="yes"/>
##    <tag k="type" v="multipolygon"/>
##  </relation>

## idea for WATER:
## - extract waterways as polygons - they are normally simple polygons, with islands created by cutting up the waterway into simple polygons
## - bridges should also be extracted as non-water areas over polygons

## NOTE: if you want to do courtyards properly, note that there are also 'relations' that define 'multipolygons' with holes.
## See: Paris:NotreDame cut folder for example


def string_to_dict(s):
    d = {}
    for i in s.split():
        kv = i.split(':')
        if len(kv) == 2:
            k,v = kv
            if k in d: d[k].append(v)
            else: d[k] = [v]
        elif len(kv) == 1: d[kv[0]] = True
        else: return False
    return d
            
def match_type(tags, pattern):
    for k in tags.keys() & pattern.keys():
        if pattern[k] is True: return True
        if tags[k] in pattern[k]: return True 
    return False


# ??? 'transportation', 'works',  'food_and_drink', 'construction', 'garages', 'industrial'
building_types_reject = {'no', 'roof', 'bunker', 'bandstand'}
building_types_accept = {'building:part','yes', 'train_station', 'shop', 'office', 'tower', 'church', 'school',
                         'apartments', 'public', 'chapel', 'civic','castle','station', 'cathedral',
                         'university','station','commercial','residential', 'house', 'convent',
                         'hotel', 'college', 'dormitory', 'retail', 'theatre', 'cinema', 'museum',
                         'hospital', 'courthouse', 'fire_station', 'synagogue','library'}

water_types = string_to_dict("waterway natural:water landuse:basin landuse:reservoir")
green_types = string_to_dict("""leisure:garden leisure:golf_course leisure:miniature_golf leisure:park leisure:pitch
leisure:track landuse:cemetery landuse:forest landuse:grass landuse:meadow landuse:orchard
landuse:recreation_ground landuse:village_green	landuse:vineyard
natural:wood natural:scrub natural:grassland""")



#http://boscoh.com/programming/reading-xml-serially.html
def get_all_streets(osmf):
    print('Processing '+osmf+' to find all streets (\'highways\')...')
    all_streets = []
    all_nodes = set()
    for event, elem in lxml_etree_iterparse(osmf, events=('start', 'end')):
        if event == 'end':
            if elem.tag == 'way':
                way = {}
                for e in elem.getchildren():
                    keys = e.keys()
                    if 'v' in keys and 'k' in keys:
                        items = dict(e.items())
                        way[items['k']] = items['v']
                #print(way)
                if 'highway' in way:
                    names = [way[k] for k in way if len(k)>3 and k[:4]=='name']
                    if names:
                        street = {'highway':way['highway']}
                        street['names'] = names
                        nodes = [int(e.items()[0][1]) for e in elem.getchildren() if 'ref' in e.keys()]
                        street['nodes'] = nodes
                        all_nodes.update(nodes)
                        all_streets.append(street)
                        #print(names)
                elem.clear()
    print('Found %d streets with a total of %d unique nodes.'%(len(all_streets),len(all_nodes)))
    print('Finding node coordinates...')
    node_map = {}
    for event, elem in lxml_etree_iterparse(osmf, events=('start', 'end')):
        if event == 'end':
            if elem.tag == 'node':
                node = dict(elem.items())
                ID = int(node['id'])
                if ID in all_nodes:
                    node_map[ID] = (node['lon'],node['lat'])
    #print('%d nodes not found.'%(len(all_nodes - node_map.keys())))
    print('Mapping nodes coordinates to streets...')
    for n,st in enumerate(all_streets):
        all_streets[n]['nodes'] = [node_map[nID] for nID in st['nodes']]
    return all_streets



def dump(osmf):
    with open(osmf,'r') as f:
        while True:
            print(f.readline())
            input()

def get_bounds(osmf):
    chars = 250
    with open(osmf,'r') as f:
        s = f.read(chars)
    li = re_findall(re_compile('<bounds.*/>'), s)
    if len(li) == 1:
        xml = parseString(li[0])
        bounds_xml = xml.getElementsByTagName('bounds')[0]
        out = {}
        for x in ['minlon', 'maxlon', 'minlat', 'maxlat']:
            out[x] = float(bounds_xml.attributes.getNamedItem(x).nodeValue)
        return out
    else:
        print('Could not find <bounds> field in first '+str(chars)+' bytes of <'+file+'>)')
        return False

def extract(osmf, items = set('streets green water buildings heights building_types'.split())):
    xml = parse(osmf)
    out = {}
    # get bounding box and centre:
    bounds_xml = xml.getElementsByTagName('bounds')[0]
    lrbt_deg = (float(bounds_xml.attributes.getNamedItem('minlon').nodeValue),
                float(bounds_xml.attributes.getNamedItem('maxlon').nodeValue),
                float(bounds_xml.attributes.getNamedItem('minlat').nodeValue),
                float(bounds_xml.attributes.getNamedItem('maxlat').nodeValue))
    centre_deg = (lrbt_deg[0]+lrbt_deg[1])/2,(lrbt_deg[2]+lrbt_deg[3])/2  
    out["centre"] = centre_deg
    out["lrbt_deg"] = lrbt_deg
    xy_bounds = deg2m(centre_deg,((lrbt_deg[1],lrbt_deg[3]),))[0]
    out["lrbt_m"] = (-xy_bounds[0],xy_bounds[0],-xy_bounds[1],xy_bounds[1])

    #get all nodes:
    all_nodes = {}
    for nd in xml.getElementsByTagName('node'):
        ID = nd.attributes.getNamedItem('id')
        lat = nd.attributes.getNamedItem('lat')
        lon = nd.attributes.getNamedItem('lon')
        # visible = nd.attributes.getNamedItem('visible') # not needed: check for visibility at 'way' level only
        if ID != None and lat != None and lon != None:
            all_nodes[int(ID.nodeValue)] = (float(lon.nodeValue),float(lat.nodeValue))

    # get all visible ways:
    all_ways = []
    for way in xml.getElementsByTagName('way'):
        visible = way.attributes.getNamedItem('visible')
        if visible and not visible.nodeValue:
            pass
        else:
            nodes = []
            tags = {}
            for child in way.childNodes:
                ch_name = child.nodeName
                if ch_name == 'nd':
                    nodes.append(int(child.attributes.getNamedItem('ref').nodeValue))
                if ch_name == 'tag':
                    tags[child.attributes.getNamedItem('k').nodeValue] = child.attributes.getNamedItem('v').nodeValue
            all_ways.append({'n':nodes,'t':tags,'#':int(way.attributes.getNamedItem('id').nodeValue)})

    # Process all multipolygon relations to find additional buildings and waterways
    if "buildings" in items: 
        all_bdg_relations = []
        for rel in xml.getElementsByTagName('relation'):
            visible = rel.attributes.getNamedItem('visible')
            if visible and not visible.nodeValue:
                pass
            else:
                tags = {}
                outer_ways = []
                inner_ways = []
                for child in rel.childNodes:
                    ch_name = child.nodeName
                    if ch_name == 'member':
                        if child.attributes.getNamedItem('type').nodeValue == 'way':
                            if child.attributes.getNamedItem('role').nodeValue == 'outer':
                                outer_ways.append(int(child.attributes.getNamedItem('ref').nodeValue))
                            elif child.attributes.getNamedItem('role').nodeValue == 'inner':
                                inner_ways.append(int(child.attributes.getNamedItem('ref').nodeValue))
                    if ch_name == 'tag':
                        tags[child.attributes.getNamedItem('k').nodeValue] = child.attributes.getNamedItem('v').nodeValue
                if 'type' in tags and tags['type'] == 'multipolygon':
                    if 'building' in tags or 'waterway' in tags:
                        new_way = {'t':tags}
                        for w in all_ways:
                            if '#' in w:
                                for ow in outer_ways:
                                    if w['#'] == ow:
                                        new_way['n'] = w['n']
                                        all_ways.append(new_way)
                                #for iw in innerways: # not implemented
                                    
    # Process all buildings, green areas, and bodies of water:
    if {"buildings", "water","green"} & items:  
        building_types_rejected = set()
        building_types_unknown = set()
        building_min_level = []
        building_types_out = {}
        buildings = []
        heights = []
        water = []
        water_not = []
        green = []
        for way in all_ways:
            b_type = None
            if 'building' in way['t']: b_type = way['t']['building']
            elif 'building:part' in way['t']: b_type = 'building:part'
            elif match_type(way['t'], water_types):
                nodes = way['n']
                if len(nodes) > 3 and nodes[0] == nodes[-1]:
                    water.append(deg2m(centre_deg,(all_nodes[x] for x in nodes[0:-1])))
            elif match_type(way['t'], green_types):
                nodes = way['n']
                if len(nodes) > 3 and nodes[0] == nodes[-1]:
                    green.append(deg2m(centre_deg,(all_nodes[x] for x in nodes[0:-1])))
            if b_type: 
                if b_type == "bridge": # bridges don't count as buildings, but as not-water
                    nodes = way['n']
                    if len(nodes) > 3 and nodes[0] == nodes[-1]:
                        water_not.append(deg2m(centre_deg,(all_nodes[x] for x in nodes[0:-1])))
                elif b_type in building_types_reject: # rejected building types
                    building_types_rejected.add(b_type)
                # this needs to have better logic: what to do with building parts that are not on ground?
                elif 'building:min_level' in way['t'] and b_type != 'building:part':
                    building_min_level.append(way['t']['building:min_level'])
                else:
                    if b_type not in building_types_accept: # log unknown building types, but accept them
                        building_types_unknown.add(b_type)
                    nodes = way['n']
                    if len(nodes) > 3 and nodes[0] == nodes[-1]:
                        Nb = len(buildings)
                        try: heights.append(float(way['t']['height'].replace('m','')))
                        except: heights.append('?')
                        buildings.append(deg2m(centre_deg,(all_nodes[x] for x in nodes[0:-1])))
                        if b_type in building_types_out: building_types_out[b_type].append(Nb)
                        else: building_types_out[b_type] = [Nb]

        out["buildings"] = tuple(buildings)
        if 'building_types' in items: out["building_types"] = building_types_out
        if 'heights' in items: out["building_heights"] = heights
        if building_types_rejected: print('Building types rejected:   ', building_types_rejected)
        if building_types_unknown: print('Unknown building types:   ', building_types_unknown)
        if building_min_level: print('Buildings with min_level rejected, at levels:   ',building_min_level)
        if "water" in items:
            out['water'] = water
            out['water_not'] = water_not
        if "green" in items:
            out['green'] = green

    # Process all streets and their intersections
    if items & {"streets", "intersections"}:   
        streets = []
        single_nds = set()
        double_nds = set()
        intersection_nds = set()
        for way in all_ways:
            if 'highway' in way['t']:
                # can filter for type of road with: way['t']['highway'] # see: http://wiki.openstreetmap.org/wiki/Highways
                nodes = way['n']
                if "streets" in items:
                    streets.append(deg2m(centre_deg,(all_nodes[x] for x in nodes)))
                if "intersections" in items:    
                    for nd in nodes:
                        if nd in single_nds:
                            if nd in double_nds: intersection_nds.add(nd)
                            else: double_nds.add(nd)
                        else: single_nds.add(nd)
        if "streets" in items: out["streets"] = tuple(streets)
        if "intersections" in items: out["intersections"] = tuple(deg2m(centre_deg,(all_nodes[x] for x in intersection_nds)))
        
    return out






def get_all_way_keys(osmf, filtall = []):
    print('Processing '+osmf+' to find all \'way\' keys...')
    out = {}
    counter = 0
    for event, elem in lxml_etree_iterparse(osmf, events=('start', 'end')):
        if event == 'end':
            if elem.tag == 'way':
                if counter == 0: print('starting...')
                counter += 1
                if counter%10000 == 0: print(counter)
                way = {}
                for e in elem.getchildren():
                    keys = e.keys()
                    if 'v' in keys and 'k' in keys:
                        items = dict(e.items())
                        way[items['k']] = items['v']
                if all([f in way for f in filtall]):
                    for k in way.keys():
                        if k not in out: out[k] = set()
                        out[k].add(way[k])
    return out



















##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################



##import xml.sax
##class XMLHandler(xml.sax.ContentHandler):
##    def __init__(self):
##        xml.sax.ContentHandler.__init__(self)
##        self.NODE_LON = HugeVec('float64', 10**7)
##        self.NODE_LAT = HugeVec('float64', 10**7)
##        self.NODE_ID = HugeDict('uint32', 10**7, 12)
##        self.node_count = 0
##    def startElement(self, name, attrs):
##        if name=='node':
##            self.NODE_ID[int(attrs.get('id'))] = self.node_count
##            self.node_count += 1
##            self.NODE_LON(float(attrs.get('lon')))
##            self.NODE_LAT(float(attrs.get('lat')))
##            if self.node_count%100000 == 0: print('Found '+str(self.node_count)+' nodes...')
##    
##def parse_osm(osmf, way_types, way_types_polygon, boundary=None):
##    xml.sax.parse(osmf, XMLHandler())



              



#http://boscoh.com/programming/reading-xml-serially.html
#import xml.etree.ElementTree as etree
def parse_osm(osmf, way_types, way_types_polygon, boundary=None):  
    BOUNDARY = boundary_sPolygons(boundary)
    RELATIONS = {};WAYS = {};USED_NODES = {};USED_WAYS = {};USED_RELATIONS = {}
    for ty in list(way_types)+['$RELATION$']:
        WAYS[ty] = {}
        USED_NODES[ty] = set()
    for ty in way_types_polygon:
        RELATIONS[ty] = {}
        USED_WAYS[ty] = set()
        USED_RELATIONS[ty] = set()
    USED_WAYS['$RELATION$'] = set()
    print('Processing '+osmf+'...')
    ###NODE_KEYS = set()
    NODE_LON = HugeVec('float64', 10**7)
    NODE_LAT = HugeVec('float64', 10**7)
    NODE_N = HugeDict('uint32', 10**7, 12)
    node_count = 0
    # magic from imposm.parser.xml.uitl.py code:
    context = lxml_etree_iterparse(osmf, events=('start', 'end'))
    _, root = context.__next__()
    for event, elem in context:
        if event == 'start': continue # goes to next for step
        if elem.tag == 'node':
            NODE_N[int(elem.get('id'))] = node_count
            node_count += 1
            NODE_LON(float(elem.get('lon')))
            NODE_LAT(float(elem.get('lat')))
            if node_count%100000 == 0: print('Found '+str(node_count)+' nodes...')
        elif elem.tag == 'way':
            way = {}
            for e in elem.getchildren():
                items = dict(e.items())
                if 'v' in items and 'k' in items:
                    way[items['k']] = items['v']
            nodes = [int(e.items()[0][1]) for e in elem.getchildren() if 'ref' in e.keys()]
            PASS = False
            for ty in way_types:
                if ty in way:
                    if BOUNDARY:
                        if not intersects(BOUNDARY, NODE_N, NODE_LON, NODE_LAT, nodes, force_line=(ty not in way_types_polygon)):
                            PASS = True
                            break
                    d = {'nodes':nodes}
                    d.update(way)
                    WAYS[ty][int(elem.get('id'))] = d
                    if len(WAYS[ty])%10000 == 0: print('Found '+str(len(WAYS[ty]))+' '+ty+'s...')
                    PASS = True
                    break # consider types in order, no double-categories
            if not PASS and len(nodes)>3 and nodes[0]==nodes[-1]: # Capture potential rings for relation topologies
                if intersects(BOUNDARY, NODE_N, NODE_LON, NODE_LAT, nodes, force_line=False):                
                    WAYS['$RELATION$'][int(elem.get('id'))] = {'nodes':nodes}
        elif elem.tag == 'relation':
            relation = {}
            outer = set()
            inner = set()
            for e in elem.getchildren():
                items = dict(e.items())
                if 'v' in items and 'k' in items:
                    relation[items['k']] = items['v']
                if 'type' in items and 'ref' in items and 'role' in items:
                    if items['type'] == 'way':
                        if items['role'] == 'outer': outer.add(int(items['ref']))
                        if items['role'] == 'inner': inner.add(int(items['ref']))
            for ty in way_types_polygon:
                if ty in relation and outer:
                    relation['outer'] = tuple(outer)
                    relation['inner'] = tuple(inner)
                    RELATIONS[ty][int(elem.get('id'))] = relation
                    break
        elif elem.tag == 'bounds':
            bounds = dict(elem.items())
            lrbt_deg = [float(bounds[s]) for s in 'minlon maxlon minlat maxlat'.split(' ')]
        #elem.clear() # This breaks ways!
        root.clear() # This is good!
        #while elem.getprevious() is not None: # These properties don't exist!
        #    del elem.getparent()[0]
#######################################################
    for ty,RS in RELATIONS.items():
        for ID,R in RS.items():
            INCLUDE = False
            for O in R['outer']:
                way = WAYS['$RELATION$'].get(O)
                if way is None: way = WAYS[ty].get(O)
                if way is None:
                    break # Bad relation.
                if intersects(BOUNDARY, NODE_N, NODE_LON, NODE_LAT, way['nodes']):
                    INCLUDE = True
                    break
            if INCLUDE:
                ways = set()
                WR = WAYS['$RELATION$']
                CC = list(R['outer'])+list(R['inner'])
                for C in CC:
                    if C in WAYS[ty]: WR[C] = WAYS[ty].pop(C)
                VALID_WAYS = True
                for C in CC:
                    if C not in WR:
                        VALID_WAYS = False
                        break
                if VALID_WAYS:
                    USED_WAYS['$RELATION$'] |= set(CC)
                    USED_RELATIONS[ty].add(ID)
    for ty in RELATIONS:
        R = RELATIONS[ty]
        RELATIONS[ty] = {k:R[k] for k in USED_RELATIONS[ty]}
    for ty in WAYS:
        if ty in way_types_polygon:
            for ID,way in WAYS[ty].items():
                if intersects(BOUNDARY, NODE_N, NODE_LON, NODE_LAT, way['nodes']):
                    USED_WAYS[ty].add(ID)
    for ty in WAYS:
        if ty in list(way_types_polygon)+['$RELATION$']:
            W = WAYS[ty]
            WAYS[ty] = {k:W[k] for k in USED_WAYS[ty]}
    for ty,WS in WAYS.items():
        UN = USED_NODES[ty]
        for _,way in WS.items():
            UN |= set(way['nodes'])
    NODES = {ty:{i:(NODE_LON[NODE_N[i]],NODE_LAT[NODE_N[i]]) for i in USED_NODES[ty]}
             for ty in USED_NODES}
    for ty in USED_NODES: print(ty,':',len(USED_NODES[ty]),'nodes')
################################################################  
    if BOUNDARY: lrbt_deg = geometry2_lrbt(sum([[(C.bounds[0],C.bounds[1]),(C.bounds[2],C.bounds[3])] for C in BOUNDARY],[]))
    else: lrbt_deg = geometry2_lrbt(sum([[ll for _,ll in NODES[ty].items()] for ty in NODES],[])) 
    return {'lrbt_deg':lrbt_deg,'relations':RELATIONS, 'ways':WAYS, 'nodes':NODES}




def parse_osm_ways(osmf, way_types):  
    WAYS = {}
    for ty in way_types:
        WAYS[ty] = {}
    print('Processing '+osmf+' for ways...')
    # magic from imposm.parser.xml.uitl.py code:
    context = lxml_etree_iterparse(osmf, events=('start', 'end'))
    _, root = context.__next__()
    for event, elem in context:
        if event == 'start': continue # goes to next for step
        elif elem.tag == 'way':
            way = {}
            for e in elem.getchildren():
                items = dict(e.items())
                if 'v' in items and 'k' in items:
                    way[items['k']] = items['v']
            PASS = False
            for ty in way_types:
                if ty in way:
                    WAYS[ty][int(elem.get('id'))] = way
                    if len(WAYS[ty])%10000 == 0: print('Found '+str(len(WAYS[ty]))+' '+ty+'s...')
                    PASS = True
                    break # consider types in order, no double-categories
        root.clear()
    return {'ways':WAYS}



def osmosis_polygon(location, paths, boundary):
    frame = bbox(boundary)
    try: 
        bounds=get_bounds(paths['city_master_osm'])
        print('Grid longitudes:   %.4f - %.4f'%(frame[0][0],frame[1][0]))
        print('Master longitudes: %.4f - %.4f'%(bounds['minlon'],bounds['maxlon'])) 
        print('Grid latitudes:   %.4f - %.4f'%(frame[0][1],frame[1][1]))
        print('Master latitudes: %.4f - %.4f'%(bounds['minlat'],bounds['maxlat']))
        if frame[0][0]<bounds['minlon'] or frame[0][1]<bounds['minlat'] or\
           frame[1][1]>bounds['maxlat'] or frame[1][0]>bounds['maxlon']:
            print('Out of bounds')
            return False
    except:
        print('Could not read bounds - cutting blindly...')

    osmosis_run = paths['osmosis_path']+'osmosis.bat'

    if not os_path_isfile(osmosis_run):
        print('Cannot find OSMOSIS in ')
        print(paths['osmosis_path'])
        return False
    else:
        print('OSMOSIS found.')
    CITY_FILES = paths['output']+'_'.join(location)
    if os_path_isfile(CITY_FILES+'.osm'):
        print('OSM file already exists.')
        return True
    POLY = ['_'.join(location)]+['1']+[' '.join(str(c) for c in t) for t in boundary]+['END']*2
    with open(CITY_FILES+'.poly','w') as f:
        for i in POLY: f.write('%s\n'%i)
    CALL = [osmosis_run,'--read-xml', 'file="'+paths['city_master_osm']+'"',
            '--bounding-polygon','file="'+CITY_FILES+'.poly"','completeWays=yes',
            '--write-xml','file="'+CITY_FILES+'.osm"']
    osmosis_return = subprocess_call(' '.join(CALL),shell=(os_name == 'nt'))
    if not os_path_isfile(CITY_FILES+'.osm'):
        print('OSMOSIS ERROR! ',osmosis_return)
        print('Make sure you set the JAVACMD to a valid Java executable file.')
        return False
    return True

if 0:
    ROOT_PATH = 'C:/Project/'  
    PATHS = {'source':ROOT_PATH+'/Canada/ON.osm',
            'output':ROOT_PATH+'/Canada/ON/',
            'osmosis_path':ROOT_PATH+"Osmosis/bin/"}
    BOUNDARY = load_data('data/PeelRegion shape+9km.shp')['polygons'][0]
    osmosis_polygon('Canada ON PeelRegion'.split(), PATHS, BOUNDARY)

def Canada_Top100():
    ROOT_PATH = 'C:/Project/'
    Country = 'Canada'
    ALL_COUNTRY = load_data(ROOT_PATH+Country+'/'+Country+' Boundaries.jsn')
    for PLACE in ALL_COUNTRY:
        if PLACE['Prov'] in 'NL NB NS'.split():
            Location = [Country, PLACE['Prov'], PLACE['City']]
            PATHS = {'source':ROOT_PATH+'/'.join(Location[:-1])+'.osm',
                 'output':ROOT_PATH+'/'.join(Location[:-1])+'/',
                 'osmosis_path':ROOT_PATH+"Osmosis/bin/"}
            print(Location)
            if not os_path_exists(PATHS['output']): os_makedirs(PATHS['output'])
            Polygon = [(float(p[0]),float(p[1])) for p in PLACE['polygonpoints']]
            Centre = (float(PLACE['lon']),float(PLACE['lat']))
            Polygon = m2deg(Centre,buffer_polygon(deg2m(Centre, Polygon),10000))
            osmosis_polygon(Location, PATHS, Polygon)
    return ALL_COUNTRY











