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

from nowcasting_dataset.data_sources.pv_gsp_data_source import (
    get_pv_gsp_metadata_from_eso,
    load_pv_gsp_raw_data_from_pvlive,
)
from pathlib import Path
from nowcasting_dataset.cloud.local import delete_all_files_in_temp_path
from nowcasting_dataset.cloud.gcp import gcp_upload_and_delete_local_files
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

start = datetime(2018, 1, 1, tzinfo=pytz.utc)
end = datetime(2021, 1, 1, tzinfo=pytz.utc)
gcp_path = "gs://solar-pv-nowcasting-data/PV/GSP/v1"

config = {"start": start, "end": end, "gcp_path": gcp_path}

# format local temp folder
LOCAL_TEMP_PATH = Path("~/temp/").expanduser()
delete_all_files_in_temp_path(path=LOCAL_TEMP_PATH)

# get data
data_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end)

# pivot to index as datetime_gmt, and columns as gsp_id
data_df = data_df.pivot(index='datetime_gmt', columns='gsp_id', values='generation_mw')
data_df.columns = [str(col) for col in data_df.columns]

# change to xarray
data_xarray = data_df.to_xarray()

# save config to file
with open(os.path.join(LOCAL_TEMP_PATH, "configuration.yaml"), "w+") as f:
    yaml.dump(config, f, allow_unicode=True)

# save data to file
data_xarray.to_zarr(os.path.join(LOCAL_TEMP_PATH, "pv_gsp.zarr"), mode="w")

# upload to gcp
gcp_upload_and_delete_local_files(dst_path=gcp_path, local_path=LOCAL_TEMP_PATH)
