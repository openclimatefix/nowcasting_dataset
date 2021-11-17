import io
import os
import time
from datetime import datetime
from pathlib import Path

import gcsfs
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset
from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWP_VARIABLE_NAMES, open_nwp

local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."
filename = "/home/jacob/nowcasting_dataset/scripts/pv_gsp.zarr"
start_dt = datetime.fromisoformat("2020-04-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2020-04-02 00:00:00.000+00:00")
gsp_power = xr.open_dataset(filename, engine="zarr")
gsp_power = gsp_power.sel(datetime_gmt=slice(start_dt, end_dt))
gsp_power = gsp_power.sel(gsp_id=slice(gsp_power.gsp_id[0], gsp_power.gsp_id[20]))

gsp_power["gsp_id"] = gsp_power.gsp_id.astype("str")

encoding = {
    var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)} for var in gsp_power.data_vars
}

gsp_power.to_zarr(f"{local_path}/tests/data/gsp/test.zarr", mode="w", encoding=encoding)
