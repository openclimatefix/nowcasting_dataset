""" Make satellite test data """
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


def generate_satellite_test_data():
    """Main function to make satelllite test data"""
    output_filename = OUTPUT_PATH / "sat_data.zarr"
    print("Writing satellite tests data to", output_filename)
    sat_data = xr.open_zarr(consts.SAT_FILENAME, consolidated=True)
    sat_data = sat_data.sel(variable=["HRV"], time=slice(START, END))
    encoding = {"stacked_eumetsat_data": {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)}}
    sat_data = sat_data.chunk({"time": 1, "y": 704, "x": 548, "variable": 1})
    sat_data.to_zarr(output_filename, mode="w", consolidated=False, encoding=encoding)


if __name__ == "__main__":
    generate_satellite_test_data()
