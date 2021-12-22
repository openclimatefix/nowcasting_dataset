"""Get test data."""
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
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.data_sources.nwp.nwp_data_source import open_nwp

# set up
BUCKET = Path("solar-pv-nowcasting-data")
local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."
filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")
configuration = load_yaml_configuration(filename)

start_dt = datetime.fromisoformat("2020-04-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2020-04-02 00:00:00.000+00:00")


# link to gcs
gcs = gcsfs.GCSFileSystem(access="read_only")

# this makes sure there is atleast one overlapping id
test_start_id = {"passiv": 7500, "pvoutput": 500}
pv_metadata_provider = {}
for metadata_filename in configuration.input_data.pv.pv_metadata_filenames:
    print(metadata_filename)

    pv_provider = "passiv" if "passiv" in metadata_filename.lower() else "pvoutput"
    print(pv_provider)

    # get metadata, reduce, and save to test data
    pv_metadata = pd.read_csv(metadata_filename, index_col="system_id")
    pv_metadata.dropna(subset=["longitude", "latitude"], how="any", inplace=True)
    pv_metadata = pv_metadata.iloc[
        test_start_id[pv_provider] : test_start_id[pv_provider] + 100
    ]  # just take a few sites
    pv_metadata.to_csv(f"{local_path}/tests/data/pv/{pv_provider}/UK_PV_metadata.csv")
    pv_metadata_provider[pv_provider] = pv_metadata


# get pv_data
system_ids_all = {}
for filename in configuration.input_data.pv.pv_filenames:
    print(filename)
    pv_provider = "passiv" if "passiv" in filename.lower() else "pvoutput"
    print(pv_provider)

    t = time.time()
    with gcs.open(filename, mode="rb") as file:
        file_bytes = file.read()

    with io.BytesIO(file_bytes) as file:
        pv_power = xr.open_dataset(file, engine="h5netcdf")

        # current pvoutput data only goes to 2019-08-20,
        # so we are adding a year so that the test data is not empty
        if pv_provider == "pvoutput":
            pv_power.__setitem__("datetime", pv_power.datetime + pd.Timedelta("365 days"))
        pv_power = pv_power.sel(datetime=slice(start_dt, end_dt))
        pv_power_df = pv_power.to_dataframe()

    # process data
    system_ids_xarray = [int(i) for i in pv_power.data_vars]
    system_ids = [
        str(system_id)
        for system_id in pv_metadata_provider[pv_provider].index.to_list()
        if system_id in system_ids_xarray
    ]
    system_ids_all[pv_provider] = system_ids

    # only take the system ids we need
    pv_power_df = pv_power_df[system_ids]
    pv_power_df = pv_power_df.dropna(axis="columns", how="all")
    pv_power_df = pv_power_df.clip(lower=0, upper=5e7)
    pv_power_new = pv_power_df.to_xarray()
    # Drop one with null
    # pv_power_new = pv_power_new.drop("3000")
    # print(pv_power_new.dims)
    # print(pv_power_new.coords["datetime"].values)
    # save to test data
    zarr_filename = f"{local_path}/tests/data/pv/{pv_provider}/test.zarr"
    pv_power_new.to_zarr(zarr_filename, compute=True, mode="w")
    pv_power = xr.load_dataset(zarr_filename, engine="zarr")
    pv_power.to_netcdf(zarr_filename.replace(".zarr", ".nc"), compute=True, engine="h5netcdf")

passiv = system_ids_all["passiv"]
pvoutput = system_ids_all["pvoutput"]

passiv = [int(id) for id in passiv]
pvoutput = [int(id) for id in pvoutput]

overlap = [id for id in passiv if id in pvoutput]

len(pd.unique(passiv + pvoutput))
len(passiv) + len(pvoutput)

############################
# NWP, this makes a file that is 9.5MW big
###########################

# Numerical weather predictions
NWP_BASE_PATH = (
    "/mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting/"
    "nowcasting_dataset_pipeline/NWP/UK_Met_Office/UKV/zarr/UKV_intermediate_version_2.zarr"
)

nwp_data_raw = open_nwp(zarr_path=NWP_BASE_PATH, consolidated=True)
nwp_data = nwp_data_raw.sel(variable=["t"])
nwp_data = nwp_data.sel(init_time=slice(start_dt, end_dt))
nwp_data = nwp_data.sel(variable=["t"])
nwp_data = nwp_data.sel(step=slice(nwp_data.step[0], nwp_data.step[4]))  # take 4 hours periods
# nwp_data = nwp_data.sel(x=slice(nwp_data.x[50], nwp_data.x[100]))
# nwp_data = nwp_data.sel(y=slice(nwp_data.y[50], nwp_data.y[100]))
nwp_data = xr.Dataset({"UKV": nwp_data})
nwp_data.UKV.values = nwp_data.UKV.values.astype(np.float16)

nwp_data.to_zarr(f"{local_path}/tests/data/nwp_data/test.zarr", mode="w")

####
# ### GSP data
#####
filename = "gs://solar-pv-nowcasting-data/PV/GSP/v2/pv_gsp.zarr"

gsp_power = xr.open_dataset(filename, engine="zarr")
gsp_power = gsp_power.sel(datetime_gmt=slice(start_dt, end_dt))
gsp_power = gsp_power.sel(gsp_id=slice(gsp_power.gsp_id[0], gsp_power.gsp_id[20]))

gsp_power["gsp_id"] = gsp_power.gsp_id.astype("str")

encoding = {
    var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)} for var in gsp_power.data_vars
}

gsp_power.to_zarr(f"{local_path}/tests/data/gsp/test.zarr", mode="w", encoding=encoding)

#####################
# SUN
#####################

filename = "gs://solar-pv-nowcasting-data/Sun/v1/sun.zarr/"
# filename = "./scripts/sun.zarr"

# open file
sun_xr = xr.open_dataset(filename, engine="zarr", mode="r", consolidated=True, chunks=None)

start_dt = datetime.fromisoformat("2019-04-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2019-04-02 00:00:00.000+00:00")


# just select one date
sun_xr = sun_xr.sel(time_5=slice(start_dt, end_dt))
sun_xr["locations"] = sun_xr.locations.astype("str")

# save to file
sun_xr.to_zarr(f"{local_path}/tests/data/sun/test.zarr", mode="w")
