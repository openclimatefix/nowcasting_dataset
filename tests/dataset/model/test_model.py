from nowcasting_dataset.dataset.model.model import Batch
from nowcasting_dataset.dataset.model.datetime import Datetime
from nowcasting_dataset.dataset.model.gsp import GSP
from nowcasting_dataset.dataset.model.pv import PV
from nowcasting_dataset.dataset.model.nwp import NWP
from nowcasting_dataset.dataset.model.satellite import Satellite
from nowcasting_dataset.dataset.model.sun import Sun
from nowcasting_dataset.dataset.model.topographic import Topographic


def test_datetime():

    s = Datetime.fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_gsp():

    s = GSP.fake(batch_size=4, seq_length_30=13, n_gsp_per_batch=32)


def test_nwp():

    s = NWP.fake(batch_size=4, seq_length_5=13, nwp_image_size_pixels=64, number_nwp_channels=8)


def test_pv():

    s = PV.fake(batch_size=4, seq_length_5=13, n_pv_systems_per_batch=128)


def test_satellite():

    s = Satellite.fake(
        batch_size=4, seq_length_5=13, satellite_image_size_pixels=64, number_sat_channels=7
    )


def test_sun():

    s = Sun.fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_topo():

    s = Topographic.fake(
        batch_size=4,
        seq_length_5=13,
        satellite_image_size_pixels=64,
    )


def test_model():

    _ = Batch.fake()


def test_model_to_numpy():

    _ = Batch.fake().change_type_to_numpy()
