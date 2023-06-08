import json

import plotly.graph_objects as go

from nowcasting_dataset.data_sources.gsp.eso import (
    get_gsp_shape_from_eso,
)
from nowcasting_dataset.geospatial import WGS84_CRS

# Seem to have 2 different GSP shape files, #Hams Hall, Melksham, Iron Acton, Axminster
s = get_gsp_shape_from_eso(join_duplicates=False)
s = s.to_crs(WGS84_CRS)
duplicated_raw = s[s["RegionID"].duplicated(keep=False)]
duplicated_raw["Amount"] = range(0, len(duplicated_raw))

for i in range(0, 8, 2):

    # just select the first one
    duplicated = duplicated_raw.iloc[i : i + 2]
    shapes_dict = json.loads(duplicated["geometry"].to_json())
    region_name = duplicated["RegionName"].iloc[0]

    # plot to check it looks right
    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=shapes_dict,
            locations=duplicated.index,
            z=duplicated["Amount"],
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=6, mapbox_center={"lat": 52, "lon": 0}
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title=region_name)

    fig.show(renderer="browser")
    # fig.write_html(f"images/duplicated_{region_name}.html")
    # fig.write_image(f"images/duplicated_{region_name}.png")


# plot the un-duplicated versions

s = get_gsp_shape_from_eso(join_duplicates=True)
s = s.to_crs(WGS84_CRS)
no_duplicated = s[s["RegionID"].isin(duplicated_raw.RegionID)]
no_duplicated["Amount"] = range(0, len(no_duplicated))

for i in range(0, len(no_duplicated)):
    # just select the first one
    data = no_duplicated.iloc[i : i + 1]
    shapes_dict = json.loads(data["geometry"].to_json())
    region_name = data["RegionName"].iloc[0]

    # plot to check it looks right
    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=shapes_dict,
            locations=data.index,
            z=data["Amount"],
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=6, mapbox_center={"lat": 52, "lon": 0}
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title=region_name)

    fig.show(renderer="browser")
    # fig.write_html(f"images/duplicated_{region_name}.html")
    # fig.write_image(f"images/duplicated_{region_name}.png")
