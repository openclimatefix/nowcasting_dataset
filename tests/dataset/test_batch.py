import pytest
import tempfile
import os
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch


@pytest.fixture
def configuration():
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    return con


def test_model(configuration):
    _ = Batch.fake(configuration=configuration)


def test_model_save_to_netcdf(configuration):
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/0.nc")


def test_model_load_from_netcdf(configuration):
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

        batch = Batch.load_netcdf(batch_idx=0, local_netcdf_path=dirpath)

        assert batch.satellite is not None
