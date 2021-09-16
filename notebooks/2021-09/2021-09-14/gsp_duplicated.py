from nowcasting_dataset.data_sources.gsp.eso import (
    get_gsp_metadata_from_eso,
    get_gsp_shape_from_eso,
)
import plotly.graph_objects as go
import json


# Seem to have 2 different GSP shape files, #Hams Hall, Melksham, Iron Acton, Axminster
s = get_gsp_shape_from_eso(join_duplicates=False)
duplicated_raw = s[s['RegionID'].duplicated(keep=False)]
duplicated_raw["Amount"] = range(0, len(duplicated_raw))

for i in range(0,8,2):

    # just select the first one
    duplicated = duplicated_raw.iloc[i: i + 2]
    shapes_dict = json.loads(duplicated["geometry"].to_json())
    region_name = duplicated['RegionName'].iloc[0]

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

    # fig.show(renderer="browser")
    fig.write_html(f"images/duplicated_{region_name}.html")
    fig.write_image(f"images/duplicated_{region_name}.png")


