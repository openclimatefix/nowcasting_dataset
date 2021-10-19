############
# Pull raw pv gsp data from Sheffield Solar
#
# 2021-09-01
# Peter Dudfield
#
# The data is about 1MB for a month of data
############
from datetime import datetime
import pytz
import yaml
import os
import numcodecs
import xarray as xr

from nowcasting_dataset.data_sources.gsp.pvlive import load_pv_gsp_raw_data_from_pvlive
from pathlib import Path
from nowcasting_dataset.filesystem.utils import (
    delete_all_files_in_temp_path,
    upload_and_delete_local_files,
)
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

start = datetime(2016, 1, 1, tzinfo=pytz.utc)
end = datetime(2021, 10, 1, tzinfo=pytz.utc)
gcp_path = "gs://solar-pv-nowcasting-data/PV/GSP/v2"

config = {"start": start, "end": end, "gcp_path": gcp_path}

# format local temp folder
LOCAL_TEMP_PATH = Path("~/temp/").expanduser()
delete_all_files_in_temp_path(path=LOCAL_TEMP_PATH)

# get data
data_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end)

# pivot to index as datetime_gmt, and columns as gsp_id
data_generation = data_df.pivot(index="datetime_gmt", columns="gsp_id", values="generation_mw")
data_generation.columns = [str(col) for col in data_generation.columns]
data_generation_xarray = xr.DataArray(
    data_generation, name="generation_mw", dims=["datetime_gmt", "gsp_id"]
)

data_capacity = data_df.pivot(
    index="datetime_gmt", columns="gsp_id", values="installedcapacity_mwp"
)
data_capacity.columns = [str(col) for col in data_capacity.columns]
data_capacity_xarray = xr.DataArray(
    data_capacity, name="installedcapacity_mwp", dims=["datetime_gmt", "gsp_id"]
)

data_xarray = xr.merge([data_generation_xarray, data_capacity_xarray])

# save config to file
with open(os.path.join(LOCAL_TEMP_PATH, "configuration.yaml"), "w+") as f:
    yaml.dump(config, f, allow_unicode=True)

# Make encoding
encoding = {
    var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)} for var in data_xarray.data_vars
}

# save data to file
data_xarray.to_zarr(os.path.join(LOCAL_TEMP_PATH, "pv_gsp.zarr"), mode="w", encoding=encoding)

# upload to gcp
upload_and_delete_local_files(dst_path=gcp_path, local_path=LOCAL_TEMP_PATH)
