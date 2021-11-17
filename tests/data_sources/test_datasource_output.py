""" Tests for data_sources """
from nowcasting_dataset.data_sources.fake import (
    gsp_fake,
    nwp_fake,
    pv_fake,
    satellite_fake,
    sun_fake,
    topographic_fake,
)


def test_gsp():
    """Test gsp fake"""
    s = gsp_fake(batch_size=4, seq_length_30=13, n_gsp_per_batch=32)

    assert s.power_mw.shape == (4, 13, 32)


def test_nwp():
    """Test nwp fake"""
    _ = nwp_fake(
        batch_size=4,
        seq_length_5=13,
        image_size_pixels=64,
        number_nwp_channels=8,
    )


def test_pv():
    """Test pv fake"""
    _ = pv_fake(batch_size=4, seq_length_5=13, n_pv_systems_per_batch=128)


def test_satellite():
    """Test satellite fake"""
    s = satellite_fake(
        batch_size=4, seq_length_5=13, satellite_image_size_pixels=64, number_satellite_channels=7
    )

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
