""" Various DataSources """
from nowcasting_dataset.data_sources.data_source import DataSource  # noqa: F401
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWPDataSource
from nowcasting_dataset.data_sources.pv.pv_data_source import PVDataSource
from nowcasting_dataset.data_sources.satellite.satellite_data_source import SatelliteDataSource
from nowcasting_dataset.data_sources.sun.sun_data_source import SunDataSource
from nowcasting_dataset.data_sources.topographic.topographic_data_source import (
    TopographicDataSource,
)

MAP_DATA_SOURCE_NAME_TO_CLASS = {
    "pv": PVDataSource,
    "satellite": SatelliteDataSource,
    "nwp": NWPDataSource,
    "gsp": GSPDataSource,
    "topographic": TopographicDataSource,
    "sun": SunDataSource,
}
ALL_DATA_SOURCE_NAMES = tuple(MAP_DATA_SOURCE_NAME_TO_CLASS.keys())
