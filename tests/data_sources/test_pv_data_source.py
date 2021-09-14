import pandas as pd
import numpy as np

from nowcasting_dataset.data_sources.pv_data_source import PVDataSource, drop_pv_systems_which_produce_overnight, calculate_azimuth_and_elevation_all_pv_systems
from datetime import datetime
import nowcasting_dataset
import os
import logging

logger = logging.getLogger(__name__)


def test_get_example_and_batch():

    path = os.path.dirname(nowcasting_dataset.__file__)

    # Solar PV data (test data)
    PV_DATA_FILENAME = f"{path}/../tests/data/pv_data/test.nc"
    PV_METADATA_FILENAME = f"{path}/../tests/data/pv_metadata/UK_PV_metadata.csv"

    pv_data_source = PVDataSource(
        history_minutes=30,
        forecast_minutes=60,
        convert_to_numpy=True,
        image_size_pixels=64,
        meters_per_pixel=2000,
        filename=PV_DATA_FILENAME,
        metadata_filename=PV_METADATA_FILENAME,
        start_dt=datetime.fromisoformat("2019-01-01 00:00:00.000+00:00"),
        end_dt=datetime.fromisoformat("2019-01-02 00:00:00.000+00:00"),
        load_azimuth_and_elevation=False,
        load_from_gcs=False,
    )

    x_locations, y_locations = pv_data_source.get_locations_for_batch(pv_data_source.pv_power.index)

    example = pv_data_source.get_example(pv_data_source.pv_power.index[0], x_locations[0], y_locations[0])
    assert "pv_yield" in example.keys()

    batch = pv_data_source.get_batch(pv_data_source.pv_power.index[0:5], x_locations[0:10], y_locations[0:10])
    assert len(batch) == 5


def test_get_example_and_batch_azimuth():

    path = os.path.dirname(nowcasting_dataset.__file__)

    # Solar PV data (test data)
    PV_DATA_FILENAME = f"{path}/../tests/data/pv_data/test.nc"
    PV_METADATA_FILENAME = f"{path}/../tests/data/pv_metadata/UK_PV_metadata.csv"

    pv_data_source = PVDataSource(
        history_minutes=30,
        forecast_minutes=60,
        convert_to_numpy=True,
        image_size_pixels=64,
        meters_per_pixel=2000,
        filename=PV_DATA_FILENAME,
        metadata_filename=PV_METADATA_FILENAME,
        start_dt=datetime.fromisoformat("2019-01-01 00:00:00.000+00:00"),
        end_dt=datetime.fromisoformat("2019-01-02 00:00:00.000+00:00"),
        load_azimuth_and_elevation=True,
        load_from_gcs=False,
    )

    x_locations, y_locations = pv_data_source.get_locations_for_batch(pv_data_source.pv_power.index)

    example = pv_data_source.get_example(pv_data_source.pv_power.index[0], x_locations[0], y_locations[0])
    assert "pv_yield" in example.keys()

    batch = pv_data_source.get_batch(pv_data_source.pv_power.index[0:5], x_locations[0:10], y_locations[0:10])
    assert len(batch) == 5


def test_drop_pv_systems_which_produce_overnight():
    pv_power = pd.DataFrame(index=pd.date_range('2010-01-01', '2010-01-02', freq='5 min'))

    _ = drop_pv_systems_which_produce_overnight(pv_power=pv_power)


def test_calculate_azimuth_and_elevation_all_pv_systems():
    datestamps = pd.date_range('2010-01-01', '2010-01-02', freq='5 min')
    N = 2548
    pv_metadata = pd.DataFrame(index=range(0, N))

    pv_metadata['latitude'] = np.random.random(N)
    pv_metadata['longitude'] = np.random.random(N)
    pv_metadata['name'] = np.random.random(N)

    azimuth, elevation = calculate_azimuth_and_elevation_all_pv_systems(datestamps=datestamps, pv_metadata=pv_metadata)

    assert len(azimuth) == len(datestamps)
    assert len(azimuth.columns) == N

    # 49 * 2548 = 100,000 takes 26 seconds
