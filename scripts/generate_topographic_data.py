"""
Script that takes the SRTM 30m worldwide elevation maps and creates a
single representation of the area in NetCDF and GeoTIFF formats

The SRTM files can be downloaded from NASA, and are in CRS EPSG:4326
"""
import os.path
import glob
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling
from nowcasting_dataset.geospatial import OSGB
import rioxarray

dst_crs = OSGB

# Go through, open all the files, combined by coords, then save out to NetCDF, or GeoTIFF
files = glob.glob("/SRTM/*.tif")
out_dir = "/SRTM_TEMP/"


upscale_factor = 0.12  # 30m to 250m-ish, just making it small enough files to actually merge
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
files = glob.glob("/SRTM_TEMP/*.tif")
src = rasterio.open(files[0])

mosaic, out_trans = merge(files)
out_meta = src.meta.copy()
out_meta.update(
    {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    }
)
out_fp = "europe_dem_250m.tif"
with rasterio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)

xds = rioxarray.open_rasterio(out_fp, parse_coordinates=True)
# Reproject to exactly 1km pixels now
with rasterio.open(out_fp) as src:
    src_crs = src.crs
    xds.attrs["crs"] = src_crs

# 500 meter resolution map
xds_resampled = xds.rio.reproject(dst_crs=dst_crs, resolution=500, resampling=Resampling.bilinear)
print(xds_resampled)
print(abs(xds_resampled.coords["x"][1] - xds_resampled.coords["x"][0]))
xds_resampled.rio.to_raster("europe_dem_500m_osgb.tif")
# 1000 meter resolution map
xds_resampled = xds.rio.reproject(dst_crs=dst_crs, resolution=1000, resampling=Resampling.bilinear)
print(xds_resampled)
print(abs(xds_resampled.coords["x"][1] - xds_resampled.coords["x"][0]))
xds_resampled.rio.to_raster("europe_dem_1km_osgb.tif")
# 2000 meter resolution meter map
xds_resampled = xds.rio.reproject(dst_crs=dst_crs, resolution=2000, resampling=Resampling.bilinear)
print(xds_resampled)
print(abs(xds_resampled.coords["x"][1] - xds_resampled.coords["x"][0]))
xds_resampled.rio.to_raster("europe_dem_2km_osgb.tif")
