"""
Script that takes the SRTM 30m worldwide elevation maps and creates a
single representation of the area in NetCDF and GeoTIFF formats

The SRTM files can be downloaded from NASA, and are in CRS EPSG:4326
"""
import os.path
import xarray
import glob
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.enums import Resampling
from nowcasting_dataset.geospatial import lat_lon_to_osgb
from itertools import zip_longest

# Go through, open all the files, combined by coords, then save out to NetCDF, or GeoTIFF
files = glob.glob("/run/media/jacob/data/SRTM/*.tif")
out_dir = "/run/media/jacob/data/SRTM1KM/"


upscale_factor = 0.03  # 30m to 1km

for f in files:
    with rasterio.open(f) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor),
            ),
            resampling=Resampling.bilinear,
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )
        name = f.split("/")[-1]
        out_meta = dataset.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform,
            }
        )
        with rasterio.open(os.path.join(out_dir, name), "w", **out_meta) as dest:
            dest.write(data)
files = glob.glob("/run/media/jacob/data/SRTM1KM/*.tif")
src = rasterio.open(files[0])

mosaic, out_trans = merge(files)
show(mosaic, cmap="terrain")
out_meta = src.meta.copy()
out_meta.update(
    {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    }
)
out_fp = "europe_dem_1km.tif"
with rasterio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)
out = rasterio.open(out_fp)
show(out, cmap="terrain")

data = xarray.open_rasterio(out_fp, parse_coordinates=True)
data.to_netcdf("europe_dem_1km.nc")


lats = data.coords["x"].values
lons = data.coords["y"].values

osgb_x = []
osgb_y = []

for lat, lon in zip_longest(lats, lons, fillvalue=lons[0]):
    # lat is much larger than lon so stop recording y coords after length of lons
    x, y = lat_lon_to_osgb(lat, lon)
    if len(osgb_y) < len(lons):
        osgb_y.append(y)
    osgb_x.append(x)

assert len(lats) == len(osgb_x)
assert len(lons) == len(osgb_y)
# Convert to OSGB meters for the coordinates
data.assign_coords({"x": osgb_x, "y": osgb_y})

data.to_netcdf("europe_dem_1km_meters.nc")
