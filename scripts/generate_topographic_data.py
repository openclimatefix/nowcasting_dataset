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
from rasterio.warp import calculate_default_transform, reproject, Resampling
from nowcasting_dataset.geospatial import OSGB

dst_crs = OSGB

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
        # Set the nodata values to 0, as nearly all ocean.
        data[data == -32767] = 0
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

with rasterio.open(out_fp) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

    with rasterio.open("europe_dem_1km_osgb.tif", "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

out = rasterio.open("europe_dem_1km_osgb.tif")
print(out.meta)
show(out, cmap="terrain", with_bounds=True)
out_fp = "europe_dem_1km_osgb.tif"
data = xarray.open_rasterio(out_fp, parse_coordinates=True)
data.attrs["scale_km"] = round(
    0.03 / upscale_factor, 3
)  # 30m * upscale factor to get current scale in km
data.attrs["upscale_factor_used"] = upscale_factor  # Factor used for upscaling
print(data)
data.to_netcdf("europe_dem_1km_meters_osgb.nc")
