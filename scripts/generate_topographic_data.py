"""
Script that takes the SRTM 30m worldwide elevation maps and creates a
single representation of the area in NetCDF and GeoTIFF formats

The SRTM files can be downloaded from NASA
"""
import os.path
import xarray
import glob
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.enums import Resampling


# Go through, open all the files, combined by coords, then save out to NetCDF, or Zarr
# While we want to move away from intermediate zarr files, this is one example where it might make sense?

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
