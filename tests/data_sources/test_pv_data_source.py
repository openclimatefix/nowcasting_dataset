import logging
import os
from datetime import datetime

import pandas as pd

import nowcasting_dataset
from nowcasting_dataset.data_sources.pv.pv_data_source import (
    PVDataSource,
    drop_pv_systems_which_produce_overnight,
)

logger = logging.getLogger(__name__)


def test_get_example_and_batch():

    path = os.path.dirname(nowcasting_dataset.__file__)

    # Solar PV data (test data)
    PV_DATA_FILENAME = f"{path}/../tests/data/pv_data/test.nc"
    PV_METADATA_FILENAME = f"{path}/../tests/data/pv_metadata/UK_PV_metadata.csv"

    pv_data_source = PVDataSource(
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
        filename=PV_DATA_FILENAME,
        metadata_filename=PV_METADATA_FILENAME,
        start_dt=datetime.fromisoformat("2019-01-01 00:00:00.000+00:00"),
        end_dt=datetime.fromisoformat("2019-01-02 00:00:00.000+00:00"),
        load_azimuth_and_elevation=False,
        load_from_gcs=False,
    )

    x_locations, y_locations = pv_data_source.get_locations(pv_data_source.pv_power.index)

    _ = pv_data_source.get_example(pv_data_source.pv_power.index[0], x_locations[0], y_locations[0])

    batch = pv_data_source.get_batch(
        pv_data_source.pv_power.index[6:11], x_locations[0:10], y_locations[0:10]
    )
    assert batch.data.shape == (5, 19, 128)


def test_drop_pv_systems_which_produce_overnight():
    pv_power = pd.DataFrame(index=pd.date_range("2010-01-01", "2010-01-02", freq="5 min"))

    _ = drop_pv_systems_which_produce_overnight(pv_power=pv_power)
