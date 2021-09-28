import datetime
import time
import io
import logging
from concurrent import futures
from typing import List, Union, Optional

import fsspec
import numcodecs
import pandas as pd
from tqdm import tqdm
import xarray as xr
import numpy as np
from pathlib import Path

from nowcasting_dataset import geospatial

logger = logging.getLogger(__name__)


def get_azimuth_and_elevation(
    datestamps: List[datetime.datetime], longitudes: List[int], latitudes: List[int]
) -> (pd.DataFrame, pd.DataFrame):
    """

    Args:
        datestamps:
        x_locations:
        y_locations:

    Returns:

    """

    logger.debug(
        f"Will be calculating for {len(datestamps)} datestamps and {len(longitudes)} locations"
    )

    # create array of index datetime, columns of system_id for both azimuth and elevation
    azimuth = []
    elevation = []

    t = time.time()
    names = []
    # loop over all metadata and fine azimuth and elevation angles,
    # not sure this is the best method to use, as currently this step takes ~2 minute for 745 pv systems,
    # and 235 datestamps (~100,000 point). But this only needs to be done once.
    with futures.ThreadPoolExecutor() as executor:

        logger.debug("Setting up jobs")

        # Submit tasks to the executor.
        future_azimuth_and_elevation_per_location = []
        for i in tqdm(range(len(longitudes))):

            name = x_y_to_name(latitudes[i], longitudes[i])
            if name not in names:

                future_azimuth_and_elevation = executor.submit(
                    geospatial.calculate_azimuth_and_elevation_angle,
                    latitude=latitudes[i],
                    longitude=longitudes[i],
                    datestamps=datestamps,
                )
                future_azimuth_and_elevation_per_location.append(
                    [future_azimuth_and_elevation, name]
                )
                names.append(name)

        logger.debug(f"Getting results")

        # Collect results from each thread.
        for i in tqdm(range(len(future_azimuth_and_elevation_per_location))):
            future_azimuth_and_elevation, name = future_azimuth_and_elevation_per_location[i]
            azimuth_and_elevation = future_azimuth_and_elevation.result()

            azimuth_per_location = azimuth_and_elevation.loc[:, "azimuth"].rename(name)
            elevation_per_location = azimuth_and_elevation.loc[:, "elevation"].rename(name)

            azimuth.append(azimuth_per_location)
            elevation.append(elevation_per_location)

    azimuth = pd.concat(azimuth, axis=1)
    elevation = pd.concat(elevation, axis=1)

    # remove timezone
    elevation.index = elevation.index.tz_localize(None)
    azimuth.index = azimuth.index.tz_localize(None)

    logger.debug(f"Calculated Azimuth and Elevation angles in {time.time() - t} seconds")

    return azimuth.round(2), elevation.round(2)


def save_to_zarr(azimuth: pd.DataFrame, elevation: pd.DataFrame, filename: Union[str, Path]):
    """

    Args:
        azimuth:
        elevation:
        filename:

    Returns:

    """

    # change pandas dataframe to xr Dataset
    elevation_xr = xr.DataArray(elevation, dims=["time_5", "locations"]).to_dataset(
        name="elevation"
    )
    azimuth_xr = xr.DataArray(azimuth, dims=["time_5", "locations"]).to_dataset(name="azimuth")

    # merge dataset
    merged_ds = xr.merge([elevation_xr, azimuth_xr])

    # Make encoding
    encoding = {
        var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)} for var in merged_ds.data_vars
    }

    # save to file
    merged_ds.to_zarr(filename, mode="w", encoding=encoding)


def load_from_zarr(
    filename: Union[str, Path],
    start_dt: Optional[datetime.datetime] = None,
    end_dt: Optional[datetime.datetime] = None,
) -> pd.DataFrame:
    """
    Load solar pv data from gcs (althought there is an option to load from loca - for testing)
    @param filename: filename of file to be loaded
    @param start_dt: the start datetime, which to trim the data to
    @param end_dt: the end datetime, which to trim the data to
    @param from_gcs: option to laod from gcs, or form local file
    @return: dataframe of pv data
    """

    logger.debug("Loading Solar PV Data from GCS")

    # It is possible to simplify the code below and do
    # xr.open_dataset(file, engine='h5netcdf')
    # in the first 'with' block, and delete the second 'with' block.
    # But that takes 1 minute to load the data, where as loading into memory
    # first and then loading from memory takes 23 seconds!
    sun = xr.open_dataset(filename, engine="zarr")

    if (start_dt is not None) and (end_dt is not None):
        sun = sun.sel(datetime_gmt=slice(start_dt, end_dt))

    elevation = pd.DataFrame(
        index=pd.to_datetime(sun.time_5), data=sun["elevation"].values, columns=sun.locations
    )
    azimuth = pd.DataFrame(
        index=pd.to_datetime(sun.time_5), data=sun["azimuth"].values, columns=sun.locations
    )

    return azimuth, elevation


def x_y_to_name(x, y):
    return f"{x},{y}"
