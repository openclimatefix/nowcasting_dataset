"""Test PVDataSource."""
import logging
import os
from datetime import datetime

import pandas as pd
import pytest

import nowcasting_dataset
from nowcasting_dataset.data_sources.pv.pv_data_source import (
    PVDataSource,
    drop_pv_systems_which_produce_overnight,
)
from nowcasting_dataset.time import time_periods_to_datetime_index

logger = logging.getLogger(__name__)


def test_get_example_and_batch():  # noqa: D103

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
        start_dt=datetime.fromisoformat("2020-04-01 00:00:00.000"),
        end_dt=datetime.fromisoformat("2020-04-02 00:00:00.000"),
        load_azimuth_and_elevation=False,
        load_from_gcs=False,
    )

    x_locations, y_locations = pv_data_source.get_locations(pv_data_source.pv_power.index)

    _ = pv_data_source.get_example(pv_data_source.pv_power.index[0], x_locations[0], y_locations[0])

    batch = pv_data_source.get_batch(
        pv_data_source.pv_power.index[6:16], x_locations[0:10], y_locations[0:10]
    )
    assert batch.power_mw.shape == (10, 19, 128)


def test_drop_pv_systems_which_produce_overnight():  # noqa: D103
    pv_power = pd.DataFrame(index=pd.date_range("2010-01-01", "2010-01-02", freq="5 min"))

    _ = drop_pv_systems_which_produce_overnight(pv_power=pv_power)


@pytest.mark.skip("CI does not have access to GCS")
def test_passive():
    """Test that Passive data source can be used in PVDataSource"""
    output_dir = "gs://solar-pv-nowcasting-data/PV/Passive/ocf_formatted/v0"

    filename = output_dir + "/passive.netcdf"
    filename_metadata = output_dir + "/system_metadata.csv"

    pv = PVDataSource(
        filename=filename,
        metadata_filename=filename_metadata,
        start_dt=datetime(2020, 3, 28),
        end_dt=datetime(2020, 4, 1),
        history_minutes=60,
        forecast_minutes=30,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    t0_time_periods = pv.get_contiguous_t0_time_periods()

    times = time_periods_to_datetime_index(t0_time_periods, freq="5T")
    x_locations, y_locations = pv.get_locations(times)

    i = 150
    example = pv.get_example(
        t0_dt=times[i], x_meters_center=x_locations[i], y_meters_center=y_locations[i]
    )

    assert example.data.max() > 0
