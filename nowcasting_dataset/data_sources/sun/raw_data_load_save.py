""" Sun Data Source """
import datetime
import logging
import time
from concurrent import futures
from pathlib import Path
from typing import List, Optional, Union

import numcodecs
import pandas as pd
import xarray as xr
from tqdm import tqdm

from nowcasting_dataset import geospatial

logger = logging.getLogger(__name__)


def get_azimuth_and_elevation(
    datestamps: List[datetime.datetime], x_centers: List[int], y_centers: List[int]
) -> (pd.DataFrame, pd.DataFrame):
    """

    Get Azimuth and elevation positions of the sun

    For a list of datestamps and a list of coordinates, get the azimuth and elevation degrees.
    Note that the degrees are rounded to 2 decimal places, as we at most need that.

    Args:
        datestamps: list of datestamps that are needed
        x_centers: list of x coordinates - ref. OSGB
        y_centers: list of y coordinates - ref. OSGB

    Returns: Tuple of dataframes for azimuth and elevation.
        The index is timestamps, and the columns are the x and y coordinates in OSGB projection

    """
    logger.debug(
        f"Will be calculating for {len(datestamps)} datestamps and {len(x_centers)} locations"
    )

    assert len(x_centers) == len(y_centers)

    # create array of index datetime, columns of system_id for both azimuth and elevation
    azimuth = []
    elevation = []

    t = time.time()
    names = []
    # loop over locations and find azimuth and elevation angles,
    with futures.ThreadPoolExecutor() as executor:

        logger.debug("Setting up jobs")

        # Submit tasks to the executor.
        future_azimuth_and_elevation_per_location = []
        for i in tqdm(range(len(x_centers))):

            name = x_y_to_name(x_centers[i], y_centers[i])
            if name not in names:

                lat, lon = geospatial.osgb_to_lat_lon(x=x_centers[i], y=y_centers[i])

                future_azimuth_and_elevation = executor.submit(
                    geospatial.calculate_azimuth_and_elevation_angle,
                    latitude=lat,
                    longitude=lon,
                    datestamps=datestamps,
                )
                future_azimuth_and_elevation_per_location.append(
                    [future_azimuth_and_elevation, name]
                )
                names.append(name)

        logger.debug("Getting results")

        # Collect results from each thread.
        for future_azimuth_and_elevation, name in tqdm(future_azimuth_and_elevation_per_location):
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


def save_to_zarr(azimuth: pd.DataFrame, elevation: pd.DataFrame, zarr_path: Union[str, Path]):
    """
    Save azimuth and elevation to zarr file

    Args:
        azimuth: data to be saved
        elevation: data to be saved
        zarr_path: the file name where it should be save, can be local of gcs

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
    merged_ds.to_zarr(zarr_path, mode="w", encoding=encoding)


def load_from_zarr(
    zarr_path: Union[str, Path],
    start_dt: Optional[datetime.datetime] = None,
    end_dt: Optional[datetime.datetime] = None,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Load sun data

    Args:
        zarr_path: the zarr_path to be loaded, can be local or gcs
        start_dt: optional start datetime. Both start and end need to be set to be used.
        end_dt: optional end datetime. Both start and end need to be set to be used.

    Returns: Tuple of dataframes for azimuth and elevation.
        The index is timestamps, and the columns are the x and y coordinates

    """
    logger.debug(f"Loading sun data from {zarr_path}")

    sun = xr.open_dataset(zarr_path, engine="zarr")

    if (start_dt is not None) and (end_dt is not None):
        sun = sun.sel(datetime_gmt=slice(start_dt, end_dt))

    elevation = pd.DataFrame(
        index=pd.to_datetime(sun.time_5), data=sun["elevation"].values, columns=sun.locations
    )
    azimuth = pd.DataFrame(
        index=pd.to_datetime(sun.time_5), data=sun["azimuth"].values, columns=sun.locations
    )

    return azimuth, elevation


def x_y_to_name(x, y) -> str:
    """
    Make name form x, y coords

    Args:
        x: x coordinate
        y: y cooridante

    Returns: name made from x and y

    """
    return f"{x},{y}"
