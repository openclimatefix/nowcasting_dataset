""" Constants that can be imported when needed """
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

# Satellite data
# TODO: Issue #423: Remove this?
SAT_FILENAME = "gs://" + str(
    Path("solar-pv-nowcasting-data")
    / "satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr"
)

# Typing
Array = Union[xr.DataArray, np.ndarray]
PV_SYSTEM_ID: str = "pv_system_id"
PV_SYSTEM_ROW_NUMBER = "pv_system_row_number"
PV_SYSTEM_X_COORDS = "pv_system_x_coords"
PV_SYSTEM_Y_COORDS = "pv_system_y_coords"

SUN_AZIMUTH_ANGLE = "sun_azimuth_angle"
SUN_ELEVATION_ANGLE = "sun_elevation_angle"

PV_YIELD = "pv_yield"
PV_DATETIME_INDEX = "pv_datetime_index"
DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE = 2048
GSP_ID: str = "gsp_id"
GSP_YIELD = "gsp_yield"
GSP_X_COORDS = "gsp_x_coords"
GSP_Y_COORDS = "gsp_y_coords"
GSP_DATETIME_INDEX = "gsp_datetime_index"
N_GSPS = 338

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
X_CENTERS_OSGB = "x_centers_osgb"
Y_CENTERS_OSGB = "y_centers_osgb"
TOPOGRAPHIC_DATA = "topo_data"
TOPOGRAPHIC_X_COORDS = "topo_x_coords"
TOPOGRAPHIC_Y_COORDS = "topo_y_coords"

# "Safe" default NWP variable names:
NWP_VARIABLE_NAMES = (
    "t",
    "dswrf",
    "prate",
    "r",
    "sde",
    "si10",
    "vis",
    "lcc",
    "mcc",
    "hcc",
)

# A complete set of NWP variable names.  Not all are currently used.
FULL_NWP_VARIABLE_NAMES = (
    "cdcb",
    "lcc",
    "mcc",
    "hcc",
    "sde",
    "hcct",
    "dswrf",
    "dlwrf",
    "h",
    "t",
    "r",
    "dpt",
    "vis",
    "si10",
    "wdir10",
    "prmsl",
    "prate",
)

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
    X_CENTERS_OSGB,
    Y_CENTERS_OSGB,
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


SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME = (
    "spatial_and_temporal_locations_of_each_example.csv"
)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

PV_PROVIDERS = ["passiv", "pvoutput"]
