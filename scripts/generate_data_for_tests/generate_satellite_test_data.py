#!/usr/bin/env python3

""" Make satellite test data """
import os
from pathlib import Path

import numcodecs
import pandas as pd
import xarray as xr

import nowcasting_dataset

START = pd.Timestamp("2020-04-01T12:00")
END = pd.Timestamp("2020-04-01T18:00")
OUTPUT_PATH = Path(os.path.dirname(nowcasting_dataset.__file__)).parent / "tests" / "data"
print(f"{OUTPUT_PATH=}")

# HRV Path
HRV_SAT_FILENAME = (
    "/mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
    "satellite/EUMETSAT/SEVIRI_RSS/zarr/v3/eumetsat_seviri_hrv_uk.zarr"
)

# Non-HRV path
SAT_FILENAME = (
    "/mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
    "satellite/EUMETSAT/SEVIRI_RSS/zarr/v3/eumetsat_seviri_uk.zarr"
)


def generate_satellite_test_data():
    """Main function to make satelllite test data"""
    # Create HRV data
    output_filename = OUTPUT_PATH / "hrv_sat_data.zarr"
    print("Opening", HRV_SAT_FILENAME)
    print("Writing satellite tests data to", output_filename)
    # This opens all the HRV satellite data
    hrv_sat_data = xr.open_mfdataset(
        HRV_SAT_FILENAME, chunks={}, mode="r", engine="zarr", concat_dim="time", combine="nested"
    )
    # v3 of the HRV data doesn't use variables. Instead the HRV data is in the 'data' DataArray.
    # hrv_sat_data = hrv_sat_data.sel(variable=["HRV"], time=slice(START, END))

    # just take a bit of the time, to keep size of file now
    hrv_sat_data = hrv_sat_data.sel(time=slice(START, END))

    # Adds compression and chunking
    encoding = {
        "data": {"compressor": numcodecs.get_codec(dict(id="bz2", level=5))},
        "time": {"units": "nanoseconds since 1970-01-01"},
    }
    # Write the HRV data to disk
    hrv_sat_data.to_zarr(
        output_filename, mode="w", consolidated=True, encoding=encoding, compute=True
    )

    # Now do the exact same with the non-HRV data
    output_filename = OUTPUT_PATH / "sat_data.zarr"
    print("Writing satellite tests data to", output_filename)
    sat_data = xr.open_mfdataset(
        SAT_FILENAME, chunks={}, mode="r", engine="zarr", concat_dim="time", combine="nested"
    )
    sat_data = sat_data.sel(variable=["IR_016"], time=slice(START, END))
    sat_data.to_zarr(output_filename, mode="w", consolidated=True, encoding=encoding, compute=True)


if __name__ == "__main__":
    generate_satellite_test_data()
