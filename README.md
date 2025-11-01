URBINTEL is an automated bike lane planning software, written in Python and some embedded OpenCL. 


The simulator runs from MAIN.py. The main function call is on the last line of that file. 
The simulator is set up to run for the city of Charlottetown in the province of Prince Edward Island (code PE).
To run a different city in Canada, download the corresponding province .osm.pbf file, place it in its folder, and edit the call to MAIN() with first parameter like so, for example: ['CA', 'ON', 'Ottawa']. The folder structure will be explained shortly.

Once the simulation is finished (will take several hours), a GUI with maps will appear. You can navigate the maps using the AWSDZX<> keys. 

In order to produce a HTML map, you will need to run website_js.py once the simulation has finished. This will produce the ChPE_data.js file (that you already have). Open Maplibre.html, which reads from that file. "Priority" shows the most needed bike lanes (protected) by the city according to the algorithm. 

This code was tested using Python 3.13 - hopefully I've listed all the requirements, but if any are missing, I apologize - see the error messages and install the missing packages manually. 



---


We use the 2016 Canadian census, since we found the 2021 census to contain missing population data for some "Dissemination Areas" (DAs).

1. Requires census geometry of DAs: lda_000b16a_e files:

https://www12.statcan.gc.ca/census-recensement/alternative_alternatif.cfm?l=eng&dispext=zip&t=lda_000b16a_e.zip&k=%20%20%20%2090414&loc=http://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/files-fichiers/2016/lda_000b16a_e.zip


2. Requires population counts for each DA (2016 census), as .csv:

https://hub.arcgis.com/datasets/esrica-tsg::canadian-population-dwelling-counts-by-dissemination-area-2016/explore


3. Requires the province OSM extract, properly named as #province#-latest, as .osm.pbf:

https://download.geofabrik.de/north-america/canada.html

Make a folder/file structure as so:
C:/data/Canada/DAs: all DA information (steps 1 and 2) unzipped here.
C:/data/Canada/topography : empty folder for topography downloads. These are done automatically by the simulator for Canada (NRCan data).
C:/data/PE: .osm.pbf file here (step 3) - simulator will store all intermediary results here for all cities from Prince Edward Island.


---


- MAIN.py 
Entry point of the simulator. Runs a sequence of functions that make up the program, each starting with "main_".

- qcommands.py
An attempt at expanding Python & numpy with some useful macros, all starting with "Q". 

- interactive.py
At the end of the simulation, the program displays a GUI written in OpenGL. Navigate the map with AWSDZX, change map layer with <>, turn road types on/off with numbers.

- UrbNet.py 
Contains low-level routing: Network, TNodes, TRoads objects make the topological network, the Graph object extracts a graph that can be routed using Dijkstra. Idea: rewriting Dijkstra using OpenCL would greatly accelerate the simulation, even when Dijkstra is used just without predecessors.
"Anchors" are where the residential and work locations are attached to the Network.  

- conflation.py
Allows importing bike lane Shapefiles to add updated bike lanes to the existing OSM data.  

- gGlobals.py 
Contains global configuration settings, colour palettes, etc. as well as OpenCL material. The entire simulator runs at the high level over a global dictionary of variables passed to the functions starting with "main_".

- earth.py
Contains functions to convert projections. We project the local earth onto a flat trapezoid. Geometry objects are handled by the Geometry object, that can store both lon-lat and trapezoid x-y (meter) coordinates conveniently.






