from nowcasting_dataset.data_sources.gsp.gsp_model import GSPML
import numpy as np
import tempfile
import os
from pathlib import Path

from nowcasting_dataset.dataset.batch import BatchML, GSPML, Batch

# from nowcasting_dataset.dataset.validate import FakeDataset
import torch
from nowcasting_dataset.config.model import Configuration

import xarray as xr

from nowcasting_dataset.dataset.xr_utils import (
    make_xr_data_array_to_tensor,
    make_xr_data_set_to_tensor,
)


def test_model():

    _ = Batch.fake()


def test_model_save_to_netcdf():

    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake().save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/0.nc")


def test_model_load_from_netcdf():

    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake().save_netcdf(path=dirpath, batch_i=0)

        batch = Batch.load_netcdf(batch_idx=0, local_netcdf_path=dirpath)

        assert batch.satellite is not None


def test_batch_to_batch_ml():

    batch_ml = BatchML.from_batch(batch=Batch.fake())


#

#
# def test_model_from_test_data(test_data_folder):
#     x = BatchML.load_netcdf(local_netcdf_path=f"{test_data_folder}/batch", batch_idx=0)
#
#
# def test_model_from_xr_dataset_to_numpy():
#
#     f = BatchML.fake()
#
#     f_xr = f.batch_to_dict_dataset()
#     fs = BatchML.load_batch_from_dict_dataset(xr_dataset=f_xr)
#     # check they are the same
#     fs.change_type_to_numpy()
#     f.gsp.to_numpy()
#     assert f.gsp.gsp_yield.shape == fs.gsp.gsp_yield.shape
#     assert (f.gsp.gsp_yield[0].astype(np.float32) == fs.gsp.gsp_yield[0]).all()
#     assert (f.gsp.gsp_yield.astype(np.float32) == fs.gsp.gsp_yield).all()
#
#
# def test_fake_dataset():
#     train = torch.utils.data.DataLoader(FakeDataset(configuration=Configuration()), batch_size=None)
#     i = iter(train)
#     x = next(i)
#
#     x = BatchML(**x)
#     # IT WORKS
#     assert type(x.satellite.sat_data) == torch.Tensor
