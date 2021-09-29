# Idea is to make a GSP video using Sheffield solar data for one day
import pandas as pd

from nowcasting_dataset.data_sources.gsp.pvlive import load_pv_gsp_raw_data_from_pvlive
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso
from datetime import datetime
from nowcasting_dataset.geospatial import WGS84_CRS
import logging
import plotly.graph_objects as go
import json

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# get gsp data
start_dt = datetime.fromisoformat("2019-06-22 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2019-06-23 00:00:00.000+00:00")
# start_dt = datetime.fromisoformat("2019-01-01 12:00:00.000+00:00")
# end_dt = datetime.fromisoformat("2019-01-01 14:00:00.000+00:00")
range = pd.date_range(start=start_dt, end=end_dt, freq="5T")
gsp_df = load_pv_gsp_raw_data_from_pvlive(start=start_dt, end=end_dt)
data_df = gsp_df.pivot(index="gsp_id", columns="datetime_gmt", values="generation_mw")
max_generation = data_df.max().max()
data_df.reset_index(inplace=True)

# gsp metadata
meta_data = get_gsp_metadata_from_eso()
shape_data_raw = meta_data.to_crs(WGS84_CRS)
shape_data_raw = shape_data_raw.sort_values(by=["RegionName"])

# merge gsp data and metadata
gps_data = shape_data_raw.merge(data_df, how="left", on=["gsp_id"])
gps_data["Area"] = gps_data["geometry"].area
shapes_dict = json.loads(gps_data["geometry"].to_json())

# plot one
midday = pd.Timestamp("2019-06-22 12:00:00")


def get_trace(dt):

    # plot to check it looks right
    return go.Choroplethmapbox(
        geojson=shapes_dict,
        locations=gps_data.index,
        z=gps_data[dt],
        zmax=max_generation,
        zmin=0,
        colorscale="Viridis",
    )


def get_frame(dt):

    # plot to check it looks right
    return go.Choroplethmapbox(
        z=gps_data[dt],
    )


# plot one
fig = go.Figure()
fig.add_trace(get_trace(midday))
# fig.update_traces(showscale=False)
fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=6, mapbox_center={"lat": 55, "lon": 0})

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title="Midday")

# fig.show(renderer="browser")
fig.write_html(f"midday_fix.html")
fig.write_image(f"midday_fix.png")

# make annimation
frames = []
for col in data_df.columns[1:]:
    print(col)
    frames.append(go.Frame(data=[get_frame(col)], layout=go.Layout(title=str(col))))

fig = go.Figure(
    frames=frames,
    data=get_trace(midday),
    layout=go.Layout(
        updatemenus=[
            dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])
        ],
    ),
)
fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=6, mapbox_center={"lat": 55, "lon": 0})

fig.show(renderer="browser")
fig.write_html(f"video.html")
