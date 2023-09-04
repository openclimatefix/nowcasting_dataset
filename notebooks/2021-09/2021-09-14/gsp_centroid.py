import json

import plotly.graph_objects as go

from nowcasting_dataset.data_sources.gsp.eso import (
    get_gsp_metadata_from_eso,
)
from nowcasting_dataset.geospatial import WGS84_CRS

# load data
shape_data_raw = get_gsp_metadata_from_eso()
shape_data_raw = shape_data_raw.to_crs(WGS84_CRS)
shape_data_raw = shape_data_raw.sort_values(by=["RegionName"])
shape_data_raw["Amount"] = 0

# for index in range(0, len(shape_data_raw)):
for index in range(140, 150):
    # just select the first one
    shape_data = shape_data_raw.iloc[index : index + 1]
    shapes_dict = json.loads(shape_data["geometry"].to_json())
    lon = shape_data["centroid_lon"].iloc[0]
    lat = shape_data["centroid_lat"].iloc[0]

    gsp_lon = shape_data["gsp_lon"].iloc[0]
    gsp_lat = shape_data["gsp_lat"].iloc[0]

    region_name = shape_data["RegionName"].iloc[0]
    region_id = shape_data["RegionID"].iloc[0]
    print(region_name, index)

    # plot to check it looks right
    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=shapes_dict,
            locations=shape_data.index,
            z=shape_data["Amount"],
            colorscale="Viridis",
        )
    )
    fig.update_traces(showscale=False)
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=7, mapbox_center={"lat": lat, "lon": lon}
    )
    fig.add_trace(
        go.Scattermapbox(
            lon=[lon], lat=[lat], mode="markers", name="Centroid", marker=dict(size=[10])
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lon=[gsp_lon],
            lat=[gsp_lat],
            mode="markers",
            name="GSP Location",
            marker=dict(size=[10]),
        )
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title=region_name)

    # fig.show(renderer="browser")
    fig.write_html(f"images/{region_name}.html")
    fig.write_image(f"images/{region_name}_{index}.png")
