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
TOPOGRAPHIC_DATA = "topo_data"
TOPOGRAPHIC_X_COORDS = "topo_x_coords"
TOPOGRAPHIC_Y_COORDS = "topo_y_coords"
NWP_VARIABLE_NAMES = ("t", "dswrf", "prate", "r", "sde", "si10", "vis", "lcc", "mcc", "hcc")
SAT_VARIABLE_NAMES = (
    "HRV",
    "IR_016",
    "IR_039",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
    "VIS006",
    "VIS008",
    "WV_062",
    "WV_073",
)

DEFAULT_REQUIRED_KEYS = [
    NWP_DATA,
    NWP_X_COORDS,
    NWP_Y_COORDS,
    SATELLITE_DATA,
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    PV_YIELD,
    PV_SYSTEM_ID,
    PV_SYSTEM_ROW_NUMBER,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_Y_COORDS,
    X_METERS_CENTER,
    Y_METERS_CENTER,
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    GSP_DATETIME_INDEX,
    TOPOGRAPHIC_DATA,
    TOPOGRAPHIC_Y_COORDS,
    TOPOGRAPHIC_X_COORDS,
] + list(DATETIME_FEATURE_NAMES)
T0_DT = "t0_dt"
