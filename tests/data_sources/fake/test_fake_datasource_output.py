""" Tests for data_sources """
from nowcasting_dataset.data_sources.fake.batch import (
    gsp_fake,
    nwp_fake,
    pv_fake,
    satellite_fake,
    sun_fake,
    topographic_fake,
)


def test_gsp(configuration):
    """Test gsp fake"""

    configuration.process.batch_size = 4
    configuration.input_data.gsp.history_minutes = 2 * 60
    configuration.input_data.gsp.forecast_minutes = 4 * 60
    configuration.input_data.gsp.n_gsp_per_example = 32

    s = gsp_fake(configuration=configuration)

    assert s.power_mw.shape == (4, 13, 32)


def test_nwp(configuration):
    """Test nwp fake"""

    configuration.process.batch_size = 4
    configuration.input_data.nwp.history_minutes = 0
    configuration.input_data.nwp.forecast_minutes = 60
    configuration.input_data.nwp.nwp_image_size_pixels = 64
    configuration.input_data.nwp.nwp_channels = ["test_channel"] * 8

    _ = nwp_fake(configuration=configuration)


def test_pv(configuration):
    """Test pv fake"""

    configuration.process.batch_size = 4
    configuration.input_data.pv.history_minutes = 30
    configuration.input_data.pv.forecast_minutes = 30
    configuration.input_data.pv.n_pv_systems_per_example = 128

    _ = pv_fake(configuration)


def test_satellite(configuration):
    """Test satellite fake"""

    configuration.process.batch_size = 4
    configuration.input_data.satellite.history_minutes = 30
    configuration.input_data.satellite.forecast_minutes = 30
    configuration.input_data.satellite.satellite_image_size_pixels = 64
    configuration.input_data.satellite.satellite_channels = ["test_channel"] * 8

    s = satellite_fake(configuration=configuration)

    assert s.x is not None


def test_sun():
    """Test sun fake"""
    _ = sun_fake(
        batch_size=4,
        seq_length_5=13,
    )


def test_topo():
    """Test topo fake"""
    _ = topographic_fake(
        batch_size=4,
        image_size_pixels=64,
    )
