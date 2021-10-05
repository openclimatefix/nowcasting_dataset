from nowcasting_dataset.dataset.model.model import Batch, batch_to_dataset
from nowcasting_dataset.dataset.model.datetime import Datetime
from nowcasting_dataset.dataset.model.gsp import GSP
from nowcasting_dataset.dataset.model.pv import PV
from nowcasting_dataset.dataset.model.nwp import NWP
from nowcasting_dataset.dataset.model.satellite import Satellite
from nowcasting_dataset.dataset.model.sun import Sun
from nowcasting_dataset.dataset.model.topographic import Topographic
import xarray as xr


def test_datetime():

    s = Datetime.fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_gsp():

    s = GSP.fake(batch_size=4, seq_length_30=13, n_gsp_per_batch=32)


def test_gsp_split():

    s = GSP.fake(batch_size=4, seq_length_30=13, n_gsp_per_batch=32)
    split = s.split()

    assert len(split) == 4
    assert type(split[0]) == GSP
    assert (split[0].gsp_yield == s.gsp_yield[0]).all()


def test_gsp_join():

    s = GSP.fake(batch_size=2, seq_length_30=13, n_gsp_per_batch=32).split()

    s: GSP = GSP.join(s)

    assert s.batch_size == 2
    assert len(s.gsp_yield.shape) == 3
    assert s.gsp_yield.shape[0] == 2
    assert s.gsp_yield.shape[1] == 13
    assert s.gsp_yield.shape[2] == 32


def test_nwp():

    s = NWP.fake(batch_size=4, seq_length_5=13, nwp_image_size_pixels=64, number_nwp_channels=8)


def test_nwp_split():

    s = NWP.fake(batch_size=4, seq_length_5=13, nwp_image_size_pixels=64, number_nwp_channels=8)
    s = s.split()


def test_pv():

    s = PV.fake(batch_size=4, seq_length_5=13, n_pv_systems_per_batch=128)


def test_satellite():

    s = Satellite.fake(
        batch_size=4, seq_length_5=13, satellite_image_size_pixels=64, number_sat_channels=7
    )

    assert s.sat_x_coords is not None


def test_sun():

    s = Sun.fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_topo():

    s = Topographic.fake(
        batch_size=4,
        satellite_image_size_pixels=64,
    )


def test_model():

    _ = Batch.fake()


def test_model_to_numpy():

    f = Batch.fake()

    f.change_type_to_numpy()

    assert type(f.gsp) == GSP


def test_model_split():

    f = Batch.fake()

    data = f.split()

    assert len(data) == f.batch_size
    assert type(data[0].gsp) == GSP


def test_model_to_xr_dataset(configuration):

    f = Batch.fake(configuration=configuration)
    f_xr = f.batch_to_dataset()

    assert type(f_xr) == xr.Dataset


def test_model_from_xr_dataset():

    f = Batch.fake()

    f_xr = batch_to_dataset(f)

    _ = Batch.load_batch_from_dataset(xr_dataset=f_xr)


def test_model_from_xr_dataset_to_numpy():

    f = Batch.fake()

    f_xr = batch_to_dataset(f)
    fs = Batch.load_batch_from_dataset(xr_dataset=f_xr)

    # check they are the same
    fs.change_type_to_numpy()
    f.gsp.to_numpy()
    assert (f.gsp.gsp_yield == fs.gsp.gsp_yield).all()
