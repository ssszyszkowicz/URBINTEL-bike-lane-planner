URBINTEL is an automated bike lane planning software, written in Python and some embedded OpenCL. 


The simulator runs from MAIN.py.
Not all the requirements are listed - you will have to find them yourself, sorry :(
The simulator is set up to run for the city of Charlottetown in the province of Prince Edward Island (code PE).

Once the simulation is finished (will take several hours), a GUI with maps will appear. You can navigate the maps using the three <>? keys. 

In order to produce a HTML map, you will need to run website_js.py once the simulation has finished. This will produce the ChPE_data.js file that you already have. Open Maplibre.html, which reads from that file. "Priority" shows the most needed bike lanes (protected) by the city according to the algorithm. 

---


We use the 2016 Canadian census, since we found the 2021 census to contain missing population data for some "Dissemination Areas" (DAs).

- Requires census geometry of DAs: lda_000b16a_e files:

https://www12.statcan.gc.ca/census-recensement/alternative_alternatif.cfm?l=eng&dispext=zip&t=lda_000b16a_e.zip&k=%20%20%20%2090414&loc=http://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/files-fichiers/2016/lda_000b16a_e.zip


- Requires population counts for each DA (2016 census), as .csv:

https://hub.arcgis.com/datasets/esrica-tsg::canadian-population-dwelling-counts-by-dissemination-area-2016/explore


- Requires the province OSM extract, properly named as #province#-latest, as .osm.pbf:

https://download.geofabrik.de/north-america/canada.html

- Make a folder/file structure as so:
C:/data/
	> Canada > DAs: all DA information unzipped here
		 > topography : empty folder for topog. downloads
	> PE: .osm.pbf file here - simulator will store all intermediary results here for all cities from Prince Edward Island.




