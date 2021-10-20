import tempfile
import os

from nowcasting_dataset.dataset.batch import Batch


def test_model():

    con = Configuration()
    con.process.batch_size = 4

    _ = Batch.fake(configuration=con)


def test_model_save_to_netcdf():

    con = Configuration()
    con.process.batch_size = 4

    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=con).save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/0.nc")


def test_model_load_from_netcdf():

    con = Configuration()
    con.process.batch_size = 4

    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=con).save_netcdf(path=dirpath, batch_i=0)

        batch = Batch.load_netcdf(batch_idx=0, local_netcdf_path=dirpath)

        assert batch.satellite is not None
