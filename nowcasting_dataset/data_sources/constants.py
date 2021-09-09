"""
Useful to define the names of the data items. This means they can be imported and referenced.
"""

PV_SYSTEM_ID: str = 'pv_system_id'
PV_SYSTEM_ROW_NUMBER = 'pv_system_row_number'
PV_SYSTEM_X_COORDS = 'pv_system_x_coords'
PV_SYSTEM_Y_COORDS = 'pv_system_y_coords'
PV_AZIMUTH_ANGLE = 'pv_azimuth_angle'
PV_ELEVATION_ANGLE = 'pv_elevation_angle'
PV_YIELD = 'pv_yield'
DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE = 128

GSP_SYSTEM_ID: str = "gsp_system_id"
GSP_YIELD = "gsp_yield"
GSP_SYSTEM_X_COORDS = "gsp_system_x_coords"
GSP_SYSTEM_Y_COORDS = "gsp_system_y_coords"
GSP_DATETIME_INDEX = "gsp_datetime_index"
DEFAULT_N_GSP_PER_EXAMPLE = 32

CENTROID_TYPE = "centroid_type"


DATETIME_FEATURE_NAMES = ("hour_of_day_sin", "hour_of_day_cos", "day_of_year_sin", "day_of_year_cos")