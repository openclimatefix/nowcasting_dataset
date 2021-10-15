from nowcasting_dataset.data_sources.fake import (
    sun_fake,
    topographic_fake,
    gsp_fake,
    datetime_fake,
    nwp_fake,
    satellite_fake,
    pv_fake,
)


def test_datetime():

    s = datetime_fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_gsp():

    s = gsp_fake(batch_size=4, seq_length_30=13, n_gsp_per_batch=32)

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

    s = nwp_fake(
        batch_size=4,
        seq_length_5=13,
        image_size_pixels=64,
        number_nwp_channels=8,
    )


def test_pv():

    s = pv_fake(batch_size=4, seq_length_5=13, n_pv_systems_per_batch=128)


# def test_nwp_pad():
#
#     s = PV.fake(batch_size=4, seq_length_5=13, n_pv_systems_per_batch=37).split()[0]
#     s.to_numpy()
#     s.pad(n_pv_systems_per_example=128)
#
#     assert s.pv_yield.shape == (13, 128)


def test_satellite():

    s = satellite_fake(
        batch_size=4, seq_length_5=13, satellite_image_size_pixels=64, number_sat_channels=7
    )

    assert s.x is not None


def test_sun():

    s = sun_fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_topo():

    s = topographic_fake(
        batch_size=4,
        image_size_pixels=64,
    )
