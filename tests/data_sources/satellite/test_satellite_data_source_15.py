"""Test SatelliteDataSource."""
import tempfile

import pytest
import zarr

from nowcasting_dataset.data_sources.fake.batch import satellite_fake
from nowcasting_dataset.data_sources.satellite.satellite_data_source import SatelliteDataSource


def test_get_15_minute_data():
    """Test to see if 15 minute data can be laoded"""
    # make fake data and save to file
    with tempfile.TemporaryDirectory() as temp_dir:
        sat_data = satellite_fake()
        sat_filename = f"{temp_dir}/sat.zarr.zip"
        sat_filename_15 = f"{temp_dir}/sat_15.zarr.zip"

        with zarr.ZipStore(sat_filename_15) as store:
            sat_data.to_zarr(store, mode="w")

        # load it into the satellite data souce
        _ = SatelliteDataSource(
            image_size_pixels_height=pytest.IMAGE_SIZE_PIXELS,
            image_size_pixels_width=pytest.IMAGE_SIZE_PIXELS,
            zarr_path=sat_filename,
            history_minutes=0,
            forecast_minutes=15,
            channels=("IR_016",),
            meters_per_pixel=6000,
            time_resolution_minutes=5,
        )
