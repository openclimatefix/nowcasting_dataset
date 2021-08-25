import time
import io
import gcsfs
import xarray as xr
from datetime import datetime
from pathlib import Path
import pandas as pd
import os
import nowcasting_dataset

# set up
BUCKET = Path("solar-pv-nowcasting-data")
local_path = os.path.dirname(nowcasting_dataset.__file__)
PV_PATH = BUCKET / "PV/PVOutput.org"
PV_METADATA_FILENAME = PV_PATH / "UK_PV_metadata.csv"

# set up variables
filename = PV_PATH / "UK_PV_timeseries_batch.nc"
metadata_filename = f"gs://{PV_METADATA_FILENAME}"
start_dt = datetime.fromisoformat("2019-01-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2019-01-02 00:00:00.000+00:00")

# link to gcs
gcs = gcsfs.GCSFileSystem(access="read_only")

# get metadata, reduce, and save to test data
pv_metadata = pd.read_csv(metadata_filename, index_col="system_id")
pv_metadata.dropna(subset=["longitude", "latitude"], how="any", inplace=True)
pv_metadata = pv_metadata.iloc[500:600]  # just take a few sites
pv_metadata.to_csv(f"{local_path}/tests/data/pv_metadata/UK_PV_metadata.csv")

# get pv_data
t = time.time()
with gcs.open(filename, mode="rb") as file:
    file_bytes = file.read()


with io.BytesIO(file_bytes) as file:
    pv_power = xr.open_dataset(file, engine="h5netcdf")
    pv_power = pv_power.sel(datetime=slice(start_dt, end_dt))
    pv_power_df = pv_power.to_dataframe()

# process data
system_ids_xarray = [int(i) for i in pv_power.data_vars]
system_ids = [str(system_id) for system_id in pv_metadata.index.to_list() if system_id in system_ids_xarray]

# only take the system ids we need
pv_power_df = pv_power_df[system_ids]
pv_power_df = pv_power_df.dropna(axis="columns", how="all")
pv_power_df = pv_power_df.clip(lower=0, upper=5e7)
pv_power_new = pv_power_df.to_xarray()

# save to test data
pv_power_new.to_netcdf(f"{local_path}/tests/data/pv_data/test.nc")
