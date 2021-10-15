import os
import tempfile

from nowcasting_dataset.data_sources.satellite.satellite_model import SatelliteML

from nowcasting_dataset.data_sources.fake import satellite_fake


def test_satellite_init():
    _ = satellite_fake


def test_satellite_save():

    with tempfile.TemporaryDirectory() as dirpath:
        satellite_fake().save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/0.nc")


def test_satellite_to_ml():
    sat = satellite_fake()

    _ = SatelliteML.from_xr_dataset(sat)


def test_satellite_ml_fake():
    _ = satellite_fake
