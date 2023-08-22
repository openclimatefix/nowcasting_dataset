"""
Pull raw pv gsp data from Sheffield Solar

2021-09-01
Peter Dudfield

The data is about 1MB for a month of data
"""
import logging
import os
from datetime import datetime
from pathlib import Path

import numcodecs
import pytz
import xarray as xr
import yaml
from pathy import Pathy

import nowcasting_dataset
from nowcasting_dataset.config import load_yaml_configuration
from nowcasting_dataset.data_sources.gsp.pvlive import load_pv_gsp_raw_data_from_pvlive
from nowcasting_dataset.filesystem.utils import (
    delete_all_files_in_temp_path,
    upload_and_delete_local_files,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

config_filename = Pathy.fluid(nowcasting_dataset.__file__).parent / "config" / "gcp.yaml"
config = load_yaml_configuration(config_filename)

start = datetime(2016, 1, 1, tzinfo=pytz.utc)
end = datetime(2021, 10, 1, tzinfo=pytz.utc)
gcp_path = config.input_data.gsp.gsp_zarr_path

config_gsp = {"start": start, "end": end, "gcp_path": gcp_path}

# format local temp folder
LOCAL_TEMP_PATH = Path("~/temp/").expanduser()
delete_all_files_in_temp_path(path=LOCAL_TEMP_PATH)


def fetch_data():
    # get data
    data_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end, normalize_data=False)

    # pivot to index as datetime_gmt, and columns as gsp_id
    data_generation_df = data_df.pivot(
        index="datetime_gmt", columns="gsp_id", values="generation_mw"
    )
    data_installedcapacity_df = data_df.pivot(
        index="datetime_gmt", columns="gsp_id", values="installedcapacity_mwp"
    )
    data_capacity_df = data_df.pivot(index="datetime_gmt", columns="gsp_id", values="capacity_mwp")
    data_updated_gmt_df = data_df.pivot(
        index="datetime_gmt", columns="gsp_id", values="updated_gmt"
    )
    data_xarray = xr.Dataset(
        data_vars={
            "generation_mw": (("datetime_gmt", "gsp_id"), data_generation_df),
            "installedcapacity_mwp": (("datetime_gmt", "gsp_id"), data_installedcapacity_df),
            "capacity_mwp": (("datetime_gmt", "gsp_id"), data_capacity_df),
            "updated_gmt": (("datetime_gmt", "gsp_id"), data_updated_gmt_df),
        },
        coords={"datetime_gmt": data_generation_df.index, "gsp_id": data_generation_df.columns},
    )

    # save config to file
    with open(os.path.join(LOCAL_TEMP_PATH, "configuration.yaml"), "w+") as f:
        yaml.dump(config_gsp, f, allow_unicode=True)

    # Make encoding
    encoding = {
        var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)}
        for var in data_xarray.data_vars
    }

    # save data to file
    data_xarray.to_zarr(os.path.join(LOCAL_TEMP_PATH, "pv_gsp.zarr"), mode="w", encoding=encoding)

    # upload to gcp
    upload_and_delete_local_files(dst_path=gcp_path, local_path=LOCAL_TEMP_PATH)

    # # code to change 'generation_mw' to 'generation_normalised'
    # data_xarray = xr.open_dataset(gcp_path + '/pv_gsp.zarr', engine="zarr")
    # data_xarray.__setitem__('gsp_id', [int(gsp_id) for gsp_id in data_xarray.gsp_id])
    # data_xarray = data_xarray.rename({"generation_mw": "generation_normalised"})
    # data_xarray.to_zarr(gcp_path + '/pv_gsp.zarr', mode="w", encoding=encoding)


if __name__ == "__main__":
    fetch_data()
