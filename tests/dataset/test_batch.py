"""Test Batch."""
import os
import tempfile

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
    _ = Batch.fake(configuration=configuration)


def test_model_save_to_netcdf(configuration):  # noqa: D103
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/000000.nc")


def test_model_load_from_netcdf(configuration):  # noqa: D103
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

        batch = Batch.load_netcdf(batch_idx=0, local_netcdf_path=dirpath)

        assert batch.satellite is not None


def test_model_load_partial_from_netcdf(configuration):  # noqa: D103
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

        batch = Batch.load_netcdf(
            batch_idx=0, local_netcdf_path=dirpath, data_source_names=["pv", "hrvsatellite"]
        )

        assert batch.satellite is None
        assert batch.pv is not None
        assert batch.hrvsatellite is not None
