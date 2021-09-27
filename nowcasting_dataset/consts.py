from typing import Union
import numpy as np
import xarray as xr
from pathlib import Path


# DEFAULT PATHS
# TODO: These should be moved elsewhere!
BUCKET = Path("solar-pv-nowcasting-data")

# Satellite data
SAT_FILENAME = "gs://" + str(
    BUCKET / "satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr"
)

# Solar PV data
PV_PATH = BUCKET / "PV/PVOutput.org"
PV_FILENAME = PV_PATH / "UK_PV_timeseries_batch.nc"
PV_METADATA_FILENAME = PV_PATH / "UK_PV_metadata.csv"

# Numerical weather predictions
NWP_FILENAME = "gs://" + str(BUCKET / "NWP/UK_Met_Office/UKV_zarr")

# Typing
Array = Union[xr.DataArray, np.ndarray]
PV_SYSTEM_ID: str = "pv_system_id"
PV_SYSTEM_ROW_NUMBER = "pv_system_row_number"
PV_SYSTEM_X_COORDS = "pv_system_x_coords"
PV_SYSTEM_Y_COORDS = "pv_system_y_coords"
PV_AZIMUTH_ANGLE = "pv_azimuth_angle"
PV_ELEVATION_ANGLE = "pv_elevation_angle"
PV_YIELD = "pv_yield"
PV_DATETIME_INDEX = "pv_datetime_index"
DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE = 128
GSP_ID: str = "gsp_id"
GSP_YIELD = "gsp_yield"
GSP_X_COORDS = "gsp_x_coords"
GSP_Y_COORDS = "gsp_y_coords"
GSP_DATETIME_INDEX = "gsp_datetime_index"
DEFAULT_N_GSP_PER_EXAMPLE = 32
OBJECT_AT_CENTER = "object_at_center"
DATETIME_FEATURE_NAMES = (
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)
SATELLITE_DATA = "sat_data"
SATELLITE_Y_COORDS = "sat_y_coords"
SATELLITE_X_COORDS = "sat_x_coords"
SATELLITE_DATETIME_INDEX = "sat_datetime_index"
NWP_TARGET_TIME = "nwp_target_time"
NWP_DATA = "nwp"
NWP_X_COORDS = "nwp_x_coords"
NWP_Y_COORDS = "nwp_y_coords"
X_METERS_CENTER = "x_meters_center"
Y_METERS_CENTER = "y_meters_center"
T0_DT = "t0_dt"
