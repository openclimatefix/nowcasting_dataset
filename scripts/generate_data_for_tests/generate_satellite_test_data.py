""" Make satellite test data """
import glob
import os
from pathlib import Path

import numcodecs
import pandas as pd
import xarray as xr

import nowcasting_dataset
from nowcasting_dataset import consts

START = pd.Timestamp("2019-01-01T12:00")
END = pd.Timestamp("2019-01-01T18:00")
OUTPUT_PATH = Path(os.path.dirname(nowcasting_dataset.__file__)).parent / "tests" / "data"
print(OUTPUT_PATH)

# HRV Path
HRV_SAT_FILENAME = os.path.join(
    "/mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/satellite/EUMETSAT/SEVIRI_RSS/zarr/v2/hrv_*"
)

# Non-HRV path
SAT_FILENAME = os.path.join(
    "/mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/satellite/EUMETSAT/SEVIRI_RSS/zarr/v2/eumetsat_zarr_*"
)


def generate_satellite_test_data():
    """Main function to make satelllite test data"""
    output_filename = OUTPUT_PATH / "hrv_sat_data.zarr"
    print("Writing satellite tests data to", output_filename)
    zarr_paths = list(glob.glob(HRV_SAT_FILENAME))
    hrv_sat_data = xr.open_mfdataset(
        zarr_paths, chunks=None, mode="r", engine="zarr", concat_dim="time"
    )
    hrv_sat_data = hrv_sat_data.sel(variable=["HRV"], time=slice(START, END))
    print(hrv_sat_data)
    encoding = {"stacked_eumetsat_data": {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)}}
    sat_data = hrv_sat_data.chunk({"time": 1, "y": 704, "x": 548, "variable": 1})
    sat_data.to_zarr(output_filename, mode="w", consolidated=False, encoding=encoding)
    output_filename = OUTPUT_PATH / "sat_data.zarr"
    print("Writing satellite tests data to", output_filename)
    zarr_paths = list(glob.glob(SAT_FILENAME))
    sat_data = xr.open_mfdataset(
        zarr_paths, chunks=None, mode="r", engine="zarr", concat_dim="time"
    )
    sat_data = sat_data.sel(variable=["IR_016"], time=slice(START, END))
    print(sat_data)
    encoding = {"stacked_eumetsat_data": {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)}}
    sat_data = sat_data.chunk({"time": 1, "y": 704, "x": 548, "variable": 1})
    sat_data.to_zarr(output_filename, mode="w", consolidated=False, encoding=encoding)


if __name__ == "__main__":
    generate_satellite_test_data()
