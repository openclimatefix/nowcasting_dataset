from typing import Union
import numpy as np
import xarray as xr
from pathlib import Path


# DEFAULT PATHS
# TODO: These should be moved elsewhere!
BUCKET = Path('solar-pv-nowcasting-data')

# Satellite data
SAT_FILENAME = 'gs://' + str(
    BUCKET / 'satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr')

# Solar PV data
PV_PATH = BUCKET / 'PV/PVOutput.org'
PV_FILENAME = PV_PATH / 'UK_PV_timeseries_batch.nc'
PV_METADATA_FILENAME = PV_PATH / 'UK_PV_metadata.csv'

# Numerical weather predictions
NWP_FILENAME = 'gs://' + str(BUCKET / 'NWP/UK_Met_Office/UKV_zarr')

# Typing
Array = Union[xr.DataArray, np.ndarray]
