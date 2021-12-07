"""Test Fake Batch"""

import pytest

from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch


@pytest.fixture
def configuration():  # noqa: D103
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    return con


def test_model(configuration):  # noqa: D103

    assert configuration.input_data.opticalflow is not None

    batch = Batch.fake(configuration=configuration)

    t0_index_gsp = configuration.input_data.gsp.history_seq_length_30_minutes
    t0_index_satellite = configuration.input_data.satellite.history_seq_length_5_minutes

    t0_datetimes_utc = batch.metadata.t0_datetime_utc
    x_center_osgb = batch.metadata.x_center_osgb
    y_center_osgb = batch.metadata.y_center_osgb

    assert batch.gsp.time[0, t0_index_gsp] == t0_datetimes_utc[0]
    assert batch.satellite.time[0, t0_index_satellite] == t0_datetimes_utc[0]

    for i in range(configuration.process.batch_size):
        for data_source_name in ["satellite", "hrvsatellite", "opticalflow", "topographic", "nwp"]:
            assert x_center_osgb[i] in batch.__getattribute__(data_source_name).x[i]
            assert y_center_osgb[i] in batch.__getattribute__(data_source_name).y[i]
