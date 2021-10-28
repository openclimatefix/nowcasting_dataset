import tempfile

import numpy as np
import pandas as pd

from nowcasting_dataset.data_sources.sun.raw_data_load_save import (
    get_azimuth_and_elevation,
    load_from_zarr,
    save_to_zarr,
)


def test_calculate_azimuth_and_elevation():
    datestamps = pd.to_datetime(pd.date_range("2010-01-01", "2010-01-02", freq="5 min"))
    N = 100
    metadata = pd.DataFrame(index=range(0, N))

    metadata["latitude"] = np.random.random(N)
    metadata["longitude"] = np.random.random(N)
    metadata["name"] = np.random.random(N)

    azimuth, elevation = get_azimuth_and_elevation(
        datestamps=datestamps, x_centers=metadata["latitude"], y_centers=metadata["longitude"]
    )

    assert len(azimuth) == len(datestamps)
    assert len(azimuth.columns) == N

    # 49 * 100 = 4,900 takes ~1 seconds


def test_save():

    datestamps = pd.to_datetime(pd.date_range("2010-01-01", "2010-01-02", freq="5 min"))
    N = 100
    metadata = pd.DataFrame(index=range(0, N))

    metadata["latitude"] = np.random.random(N)
    metadata["longitude"] = np.random.random(N)
    metadata["name"] = np.random.random(N)

    azimuth, elevation = get_azimuth_and_elevation(
        datestamps=datestamps, x_centers=metadata["latitude"], y_centers=metadata["longitude"]
    )

    with tempfile.TemporaryDirectory() as fp:
        save_to_zarr(azimuth=azimuth, elevation=elevation, filename=fp)


def test_load(test_data_folder):

    filename = test_data_folder + "/sun/test.zarr"

    azimuth, elevation = load_from_zarr(filename=filename)

    assert type(azimuth) == pd.DataFrame
    assert type(elevation) == pd.DataFrame
