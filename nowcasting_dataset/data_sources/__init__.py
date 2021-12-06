""" Various DataSources """
from nowcasting_dataset.data_sources.data_source import DataSource  # noqa: F401
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWPDataSource

# We must import OpticalFlowDataSource *after* SatelliteDataSource,
# otherwise we get circular import errors!
from nowcasting_dataset.data_sources.optical_flow.optical_flow_data_source import (
    OpticalFlowDataSource,
)
from nowcasting_dataset.data_sources.pv.pv_data_source import PVDataSource
from nowcasting_dataset.data_sources.satellite.satellite_data_source import (
    HRVSatelliteDataSource,
    SatelliteDataSource,
)
from nowcasting_dataset.data_sources.sun.sun_data_source import SunDataSource
from nowcasting_dataset.data_sources.topographic.topographic_data_source import (
    TopographicDataSource,
)

# If you update MAP_DATA_SOURCE_NAME_TO_CLASS, then please also update ALL_DATA_SOURCE_NAMES in
# nowcasting_dataset.config.model.InputData.set_forecast_and_history_minutes
MAP_DATA_SOURCE_NAME_TO_CLASS = {
    "pv": PVDataSource,
    "satellite": SatelliteDataSource,
    "hrvsatellite": HRVSatelliteDataSource,
    "opticalflow": OpticalFlowDataSource,
    "nwp": NWPDataSource,
    "gsp": GSPDataSource,
    "topographic": TopographicDataSource,
    "sun": SunDataSource,
}
ALL_DATA_SOURCE_NAMES = tuple(MAP_DATA_SOURCE_NAME_TO_CLASS.keys())
