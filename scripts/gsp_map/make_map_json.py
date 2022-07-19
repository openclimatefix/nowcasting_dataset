"""
script to make map for nowcasting application

"""
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_shape_from_eso

d = get_gsp_shape_from_eso(load_local_file=False)

# add gsp_id to
d["gsp_id"] = range(1, len(d) + 1)

d = d[["gsp_id", "GSPs", "geometry"]]
d = d.to_crs(4326)

d_json = d.to_json(indent=4)
with open("gsp_regions_20220314.json", "w") as f:
    f.write(d_json)
