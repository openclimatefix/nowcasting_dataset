import os
from datetime import datetime

import nowcasting_dataset
from nowcasting_dataset.data_sources.data_source_list import DataSourceList
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
import nowcasting_dataset.utils as nd_utils


def test_sample_spatial_and_temporal_locations_for_examples():
    local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_dt=datetime(2019, 1, 1),
        end_dt=datetime(2019, 1, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    data_source_list = DataSourceList([gsp])
    t0_datetimes = data_source_list.get_t0_datetimes_across_all_data_sources(freq="30T")
    locations = data_source_list.sample_spatial_and_temporal_locations_for_examples(
        t0_datetimes=t0_datetimes, n_examples=10
    )

    assert locations.columns.to_list() == ["t0_datetime_UTC", "x_center_OSGB", "y_center_OSGB"]
    assert len(locations) == 10


def test_from_config():
    config = nd_utils.get_config_with_test_paths("test.yaml")
    data_source_list = DataSourceList.from_config(config.input_data)
    assert len(data_source_list) == 6
    assert isinstance(
        data_source_list.data_source_which_defines_geospatial_locations, GSPDataSource
    )
