from nowcasting_dataset.data_sources.datetime.datetime_model import Datetime
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.nwp.nwp_model import NWP
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.data_sources.topographic.topographic_model import Topographic


def test_datetime():

    s = Datetime.fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_gsp():

    s = GSP.fake(batch_size=4, seq_length_30=13, n_gsp_per_batch=32)

    assert s.data.shape == (4, 13, 32)


# def test_gsp_pad():
#
#     s = GSP.fake(batch_size=4, seq_length_30=13, n_gsp_per_batch=7)
#     # s.to_numpy()
#     # s.pad(n_gsp_per_example=32)
#
#     # need to add pad
#
#     assert s.data.shape == (4, 13, 32)


def test_nwp():

    s = NWP.fake(
        batch_size=4,
        seq_length_5=13,
        image_size_pixels=64,
        number_nwp_channels=8,
    )


def test_pv():

    s = PV.fake(batch_size=4, seq_length_5=13, n_pv_systems_per_batch=128)


# def test_nwp_pad():
#
#     s = PV.fake(batch_size=4, seq_length_5=13, n_pv_systems_per_batch=37).split()[0]
#     s.to_numpy()
#     s.pad(n_pv_systems_per_example=128)
#
#     assert s.pv_yield.shape == (13, 128)


def test_satellite():

    s = Satellite.fake(
        batch_size=4, seq_length_5=13, satellite_image_size_pixels=64, number_sat_channels=7
    )

    assert s.x is not None


def test_sun():

    s = Sun.fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_topo():

    s = Topographic.fake(
        batch_size=4,
        image_size_pixels=64,
    )
