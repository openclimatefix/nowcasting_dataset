""" Make satellite test data """
import glob
import os
from pathlib import Path

import numcodecs
import pandas as pd
import xarray as xr

import nowcasting_dataset

START = pd.Timestamp("2020-04-01T12:00")
END = pd.Timestamp("2020-04-01T18:00")
OUTPUT_PATH = Path(os.path.dirname(nowcasting_dataset.__file__)).parent / "tests" / "data"
print(OUTPUT_PATH)

# HRV Path
HRV_SAT_FILENAME = os.path.join(
    "/mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/ \
    satellite/EUMETSAT/SEVIRI_RSS/zarr/v2/hrv_*"
)

# Non-HRV path
SAT_FILENAME = os.path.join(
    "/mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/ \
    satellite/EUMETSAT/SEVIRI_RSS/zarr/v2/eumetsat_zarr_*"
)


def generate_satellite_test_data():
    """Main function to make satelllite test data"""
    output_filename = OUTPUT_PATH / "hrv_sat_data.zarr"
    print("Writing satellite tests data to", output_filename)
    zarr_paths = list(glob.glob(HRV_SAT_FILENAME))
    # This opens all the HRV satellite data
    hrv_sat_data = xr.open_mfdataset(
        zarr_paths, chunks=None, mode="r", engine="zarr", concat_dim="time"
    )
    hrv_sat_data = hrv_sat_data.sel(variable=["HRV"], time=slice(START, END))

    # just take a bit of the time, to keep size of file now
    hrv_sat_data = hrv_sat_data.sel(time=slice(hrv_sat_data.time[0], hrv_sat_data.time[25]))

    # Adds compression and chunking
    encoding = {"stacked_eumetsat_data": {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)}}
    hrv_sat_data = hrv_sat_data.chunk({"time": 1, "y": 704, "x": 548, "variable": 1})
    # Write the HRV data to disk
    hrv_sat_data.to_zarr(
        output_filename, mode="w", consolidated=True, encoding=encoding, compute=True
    )

    # Now do the exact same with the non-HRV data
    output_filename = OUTPUT_PATH / "sat_data.zarr"
    print("Writing satellite tests data to", output_filename)
    zarr_paths = list(glob.glob(SAT_FILENAME))
    sat_data = xr.open_mfdataset(
        zarr_paths, chunks=None, mode="r", engine="zarr", concat_dim="time"
    )
    sat_data = sat_data.sel(variable=["IR_016"], time=slice(START, END))
    encoding = {"stacked_eumetsat_data": {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)}}
    sat_data = sat_data.chunk({"time": 1, "y": 704, "x": 548, "variable": 1})
    sat_data.to_zarr(output_filename, mode="w", consolidated=True, encoding=encoding, compute=True)


if __name__ == "__main__":
    generate_satellite_test_data()
