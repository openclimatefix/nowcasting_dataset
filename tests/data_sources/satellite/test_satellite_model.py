import os
import tempfile

from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite, SatelliteML

from nowcasting_dataset.dataset.xr_torch import (
    make_xr_data_array_to_tensor,
    make_xr_data_set_to_tensor,
)


def test_satellite_init():
    _ = Satellite.fake()


def test_satellite_save():

    with tempfile.TemporaryDirectory() as dirpath:
        Satellite.fake().save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/0.nc")


def test_satellite_to_ml():
    sat = Satellite.fake()

    _ = SatelliteML.from_xr_dataset(sat)


def test_satellite_ml_fake():
    _ = SatelliteML.fake()
