"""Test PVDataSource."""
import logging
import os
from datetime import datetime

import pandas as pd
import pytest

import nowcasting_dataset
from nowcasting_dataset.config.model import PVFiles
from nowcasting_dataset.consts import DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE
from nowcasting_dataset.data_sources.fake.batch import pv_fake
from nowcasting_dataset.data_sources.pv.pv_data_source import (
    PVDataSource,
    drop_pv_systems_which_produce_overnight,
)
from nowcasting_dataset.time import time_periods_to_datetime_index

logger = logging.getLogger(__name__)


def test_pv_normalized(configuration):
    """Test pv normalization"""

    configuration.process.batch_size = 4
    configuration.input_data.pv.history_minutes = 30
    configuration.input_data.pv.forecast_minutes = 30
    configuration.input_data.pv.n_pv_systems_per_example = 128

    pv = pv_fake(configuration=configuration)

    power_normalized = pv.power_normalized

    assert (power_normalized.values >= 0).all()
    assert (power_normalized.values <= 1).all()


def test_get_example_and_batch():  # noqa: D103

    path = os.path.dirname(nowcasting_dataset.__file__)

    # Solar PV data (test data)
    PV_DATA_FILENAME = f"{path}/../tests/data/pv/passiv/test.nc"
    PV_METADATA_FILENAME = f"{path}/../tests/data/pv/passiv/UK_PV_metadata.csv"

    pv_data_source = PVDataSource(
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
        files_groups=[
            PVFiles(
                pv_filename=PV_DATA_FILENAME,
                pv_metadata_filename=PV_METADATA_FILENAME,
                label="passiv",
            )
        ],
        start_datetime=datetime.fromisoformat("2020-04-01 00:00:00.000"),
        end_datetime=datetime.fromisoformat("2020-04-02 00:00:00.000"),
        load_azimuth_and_elevation=False,
        load_from_gcs=False,
    )

    assert pv_data_source.pv_metadata["kwp"].min() > 0

    locations = pv_data_source.get_locations(pv_data_source.pv_power.index)

    _ = pv_data_source.get_example(location=locations[6])

    # start at 6, to avoid some nans
    batch = pv_data_source.get_batch(locations=locations[6:16])
    assert batch.power_mw.shape == (10, 19, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE)
    assert str(batch.x_osgb.dtype) == "float32"
    assert str(batch.y_osgb.dtype) == "float32"
    assert str(batch.id.dtype) == "int32"
    assert str(batch.example.dtype) == "int32"
    assert str(batch.id_index.dtype) == "int32"
    assert str(batch.time_index.dtype) == "int32"


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
        files_groups=[PVFiles(pv_filename=filename, pv_metadata_filename=filename_metadata)],
        start_datetime=datetime(2020, 3, 28),
        end_datetime=datetime(2020, 4, 1),
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
        t0_datetime_utc=times[i], x_center_osgb=x_locations[i], y_center_osgb=y_locations[i]
    )

    assert example.data.max() > 0
