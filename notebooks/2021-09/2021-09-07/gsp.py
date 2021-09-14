import json
from urllib.request import urlopen
import geopandas as gpd
import plotly.graph_objects as go
import numpy as np
import itertools

WGS84_CRS = "EPSG:4326"

# get file
url = (
    "https://data.nationalgrideso.com/backend/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/resource/"
    "a3ed5711-407a-42a9-a63a-011615eea7e0/download/gsp_regions_20181031.geojson"
)

with urlopen(url) as response:
    shapes_gdp = gpd.read_file(response).to_crs(WGS84_CRS)

# set z axis
shapes_gdp["Amount"] = 0

# get dict shapes
shapes_dict = json.loads(shapes_gdp.to_json())

# plot it
fig = go.Figure()
fig.add_trace(
    go.Choroplethmapbox(geojson=shapes_dict, locations=shapes_gdp.index, z=shapes_gdp.Amount, colorscale="Viridis")
)

fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=4, mapbox_center={"lat": 55, "lon": 0})
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show(renderer="browser")
fig.write_html('gsp.html')


# find out if point is in gsp
from shapely.geometry import Point, Polygon

_pnts = [Point(3, 3), Point(8, 8), Point(0, 51.38)]
pnts = gpd.GeoDataFrame(geometry=_pnts, index=['A', 'B', 'C'])

# useful way to see if a point is in a polygon
pnts = pnts.assign(**{str(key): pnts.within(geom['geometry']) for key, geom in shapes_gdp.iterrows()})

# create lat and long array, same system as Example,
sat_x_coords = np.array(range(480000, 600000, 2000)) / 10000
sat_y_coords = np.array(range(330000, 200000, -2000)) / 10000

sat_x_y_coords = np.zeros((sat_x_coords.size, sat_y_coords.size,2))
_pnts = []
for i, j in itertools.product(range(sat_x_coords.size), range(sat_y_coords.size)):
    _pnts.append(Point(sat_x_coords[i], sat_y_coords[j]))
    pnts = gpd.GeoDataFrame(geometry=_pnts)
    sat_x_y_coords[i,j] = np.array([sat_x_coords[i], sat_y_coords[j]])

pnts = pnts.assign(**{str(key): pnts.within(geom['geometry']) for key, geom in shapes_gdp.iterrows()})











