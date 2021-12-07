""" To make fake Datasets

Wanted to keep this out of the testing frame works, as other repos, might want to use this
"""
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.consts import NWP_VARIABLE_NAMES, SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.metadata.metadata_model import Metadata
from nowcasting_dataset.data_sources.nwp.nwp_model import NWP
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.satellite.satellite_model import HRVSatellite, Satellite
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.data_sources.topographic.topographic_model import Topographic
from nowcasting_dataset.dataset.xr_utils import (
    convert_coordinates_to_indexes,
    convert_coordinates_to_indexes_for_list_datasets,
    join_list_dataset_to_batch_dataset,
)
from nowcasting_dataset.geospatial import lat_lon_to_osgb


def gsp_fake(
    batch_size,
    seq_length_30,
    n_gsp_per_batch,
):
    """Create fake data"""
    # make batch of arrays
    xr_datasets = [
        create_gsp_pv_dataset(
            seq_length=seq_length_30,
            freq="30T",
            number_of_systems=n_gsp_per_batch,
        )
        for _ in range(batch_size)
    ]

    # change dimensions to dimension indexes
    xr_datasets = convert_coordinates_to_indexes_for_list_datasets(xr_datasets)

    # make dataset
    xr_dataset = join_list_dataset_to_batch_dataset(xr_datasets)

    return GSP(xr_dataset)


def metadata_fake(batch_size):
    """Make a xr dataset"""

    # get random OSGB center in the UK
    lat = np.random.uniform(51, 55, batch_size)
    lon = np.random.uniform(-2.5, 1, batch_size)
    x_centers_osgb, y_centers_osgb = lat_lon_to_osgb(lat=lat, lon=lon)

    # get random times
    all_datetimes = pd.date_range("2021-01-01", "2021-02-01", freq="5T")
    t0_datetimes_utc = np.random.choice(all_datetimes, batch_size, replace=False)
    # np.random.choice turns the pd.Timestamp objects into datetime.datetime objects.
    t0_datetimes_utc = pd.to_datetime(t0_datetimes_utc)

    metadata_dict = {}
    metadata_dict["batch_size"] = batch_size
    metadata_dict["x_center_osgb"] = list(x_centers_osgb)
    metadata_dict["y_center_osgb"] = list(y_centers_osgb)
    metadata_dict["t0_datetime_utc"] = list(t0_datetimes_utc)

    return Metadata(**metadata_dict)


def nwp_fake(
    batch_size=32,
    seq_length_60=2,
    image_size_pixels=64,
    number_nwp_channels=7,
) -> NWP:
    """Create fake data"""
    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length=seq_length_60,
            image_size_pixels=image_size_pixels,
            channels=NWP_VARIABLE_NAMES[0:number_nwp_channels],
            freq="60T",
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    xr_dataset["init_time"] = xr_dataset.time[:, 0]

    return NWP(xr_dataset)


def pv_fake(batch_size, seq_length_5, n_pv_systems_per_batch):
    """Create fake data"""
    # make batch of arrays
    xr_datasets = [
        create_gsp_pv_dataset(
            seq_length=seq_length_5,
            freq="5T",
            number_of_systems=n_pv_systems_per_batch,
            time_dependent_capacity=False,
        )
        for _ in range(batch_size)
    ]

    # change dimensions to dimension indexes
    xr_datasets = convert_coordinates_to_indexes_for_list_datasets(xr_datasets)

    # make dataset
    xr_dataset = join_list_dataset_to_batch_dataset(xr_datasets)

    return PV(xr_dataset)


def satellite_fake(
    batch_size=32,
    seq_length_5=19,
    satellite_image_size_pixels=64,
    number_satellite_channels=7,
) -> Satellite:
    """Create fake data"""
    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length=seq_length_5,
            image_size_pixels=satellite_image_size_pixels,
            channels=SAT_VARIABLE_NAMES[1:number_satellite_channels],
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return Satellite(xr_dataset)


def hrv_satellite_fake(
    batch_size=32,
    seq_length_5=19,
    satellite_image_size_pixels=64,
    number_satellite_channels=7,
) -> Satellite:
    """Create fake data"""
    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length=seq_length_5,
            image_size_pixels=satellite_image_size_pixels * 3,  # HRV images are 3x other images
            channels=SAT_VARIABLE_NAMES[0:1],
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return HRVSatellite(xr_dataset)


def optical_flow_fake(
    batch_size=32,
    seq_length_5=19,
    satellite_image_size_pixels=64,
    number_satellite_channels=7,
) -> OpticalFlow:
    """Create fake data"""
    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length=seq_length_5,
            freq="5T",
            image_size_pixels=satellite_image_size_pixels,
            channels=SAT_VARIABLE_NAMES[0:number_satellite_channels],
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return OpticalFlow(xr_dataset)


def sun_fake(batch_size, seq_length_5):
    """Create fake data"""
    # create dataset with both azimuth and elevation, index with time
    # make batch of arrays
    xr_arrays = [
        create_sun_dataset(
            seq_length=seq_length_5,
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_dataset_to_batch_dataset(xr_arrays)

    return Sun(xr_dataset)


def topographic_fake(batch_size, image_size_pixels):
    """Create fake data"""
    # make batch of arrays
    xr_arrays = [
        xr.DataArray(
            data=np.random.randn(
                image_size_pixels,
                image_size_pixels,
            ),
            dims=["x", "y"],
            coords=dict(
                x=np.sort(np.random.randn(image_size_pixels)),
                y=np.sort(np.random.randn(image_size_pixels))[::-1].copy(),
            ),
            name="data",
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return Topographic(xr_dataset)


def add_uk_centroid_osgb(x, y):
    """
    Add an OSGB value to make in center of UK

    Args:
        x: random values, OSGB
        y: random values, OSGB

    Returns: X,Y random coordinates [OSGB]
    """

    # get random OSGB center in the UK
    lat = np.random.uniform(51, 55)
    lon = np.random.uniform(-2.5, 1)
    x_center, y_center = lat_lon_to_osgb(lat=lat, lon=lon)

    # make average 0
    x = x - x.mean()
    y = y - y.mean()

    # put in the uk
    x = x + x_center
    y = y + y_center

    return x, y


def create_random_point_coordinates_osgb(size: int):
    """Make random coords [OSGB] for pv site, or gsp"""
    # this is about 100KM
    HUNDRED_KILOMETERS = 10 ** 5
    x = np.random.randint(0, HUNDRED_KILOMETERS, size)
    y = np.random.randint(0, HUNDRED_KILOMETERS, size)

    return add_uk_centroid_osgb(x, y)


def make_random_image_coords_osgb(size: int):
    """Make random coords for image. These are ranges for the pixels"""

    ONE_KILOMETER = 10 ** 3

    # 4 kilometer spacing seemed about right for real satellite images
    x = 4 * ONE_KILOMETER * np.array((range(0, size)))
    y = 4 * ONE_KILOMETER * np.array((range(size, 0, -1)))

    return add_uk_centroid_osgb(x, y)


def create_image_array(
    dims=("time", "x", "y", "channels"),
    seq_length=19,
    image_size_pixels=64,
    channels=SAT_VARIABLE_NAMES,
    freq="5T",
):
    """Create Satellite or NWP fake image data"""

    x, y = make_random_image_coords_osgb(size=image_size_pixels)

    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq=freq, periods=seq_length),
        "x": x,
        "y": y,
        "channels": np.array(channels),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    image_data_array = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(seq_length, image_size_pixels, image_size_pixels, len(channels)),
            )
        ),
        coords=coords,
        name="data",
    )  # Fake data for testing!

    return image_data_array


def create_gsp_pv_dataset(
    dims=("time", "id"),
    freq="5T",
    seq_length=19,
    number_of_systems=128,
    time_dependent_capacity: bool = True,
) -> xr.Dataset:
    """
    Create gsp or pv fake dataset

    Args:
        dims: the dims that are made for "power_mw"
        freq: the frequency of the time steps
        seq_length: the time sequence length
        number_of_systems: number of pv or gsp systems
        time_dependent_capacity: if the capacity is time dependent.
            GSP capacities increase over time,
            but PV systems are the same (or should be).

    Returns: xr.Dataset of fake data

    """
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq=freq, periods=seq_length),
        "id": np.random.choice(range(1000), number_of_systems, replace=False),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]

    # make pv yield.  randn samples from a Normal distribution (and so can go negative).
    # The values are clipped to be positive later.
    data = np.random.randn(seq_length, number_of_systems)

    # smooth the data, the convolution method smooths that data across systems first,
    # and then a bit across time (depending what you set N)
    N = int(seq_length / 2)
    data = np.convolve(data.ravel(), np.ones(N) / N, mode="same").reshape(
        (seq_length, number_of_systems)
    )
    # Need to clip  *after* smoothing, because the smoothing method might push
    # non-zero data below zero.  Clip at 0.1 instead of 0 so we don't get div-by-zero errors
    # if capacity is zero (capacity is computed as the max of the random numbers).
    data = data.clip(min=0.1)

    # make into a Data Array
    data_array = xr.DataArray(
        data,
        coords=coords,
    )  # Fake data for testing!

    capacity = data_array.max(dim="time")
    if time_dependent_capacity:
        capacity = capacity.expand_dims(time=seq_length)
        capacity.__setitem__("time", data_array.time.values)

    data = data_array.to_dataset(name="power_mw")

    # make random coords
    x, y = create_random_point_coordinates_osgb(size=number_of_systems)

    x_coords = xr.DataArray(
        data=x,
        dims=["id"],
    )

    y_coords = xr.DataArray(
        data=y,
        dims=["id"],
    )

    # make first coords centroid
    x_coords.data[0] = x_coords.data.mean()
    y_coords.data[0] = y_coords.data.mean()

    data["capacity_mwp"] = capacity
    data["x_coords"] = x_coords
    data["y_coords"] = y_coords

    # Add 1000 to the id numbers for the row numbers.
    # This is a quick way to make sure row number is different from id,
    data["pv_system_row_number"] = data["id"] + 1000

    data.__setitem__("power_mw", data.power_mw.clip(min=0))

    return data


def create_sun_dataset(
    dims=("time",),
    freq="5T",
    seq_length=19,
) -> xr.Dataset:
    """
    Create sun fake dataset

    Args:
        dims: # TODO
        freq: # TODO
        seq_length: # TODO

    Returns: # TODO

    """
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq=freq, periods=seq_length),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    data_array = xr.DataArray(
        np.random.randn(
            seq_length,
        ),
        coords=coords,
    )  # Fake data for testing!

    sun = data_array.to_dataset(name="elevation")
    sun["azimuth"] = sun.elevation

    sun.__setitem__("azimuth", sun.azimuth.clip(min=0, max=360))
    sun.__setitem__("elevation", sun.elevation.clip(min=-90, max=90))

    sun = convert_coordinates_to_indexes(sun)

    return sun


def create_metadata_dataset() -> xr.Dataset:
    """Create fake metadata dataset"""
    d = {
        "dims": ("t0_dt",),
        "data": pd.date_range("2021-01-01", freq="5T", periods=1) + pd.Timedelta("30T"),
    }

    data = (xr.DataArray.from_dict(d)).to_dataset(name="data")

    for v in ["x_meters_center", "y_meters_center", "object_at_center_label"]:
        d: dict = {"dims": ("t0_dt",), "data": [np.random.randint(0, 1000)]}
        d: xr.Dataset = (xr.DataArray.from_dict(d)).to_dataset(name=v)
        data[v] = getattr(d, v)

    return data


def create_datetime_dataset(
    seq_length=19,
) -> xr.Dataset:
    """Create fake datetime dataset"""
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq="5T", periods=seq_length),
    }
    coords = [("time", ALL_COORDS["time"])]
    data_array = xr.DataArray(
        np.random.randn(
            seq_length,
        ),
        coords=coords,
    )  # Fake data

    data = data_array.to_dataset()

    ds = data.rename({"data": "day_of_year_cos"})
    ds["day_of_year_sin"] = data.rename({"data": "day_of_year_sin"}).day_of_year_sin
    ds["hour_of_day_cos"] = data.rename({"data": "hour_of_day_cos"}).hour_of_day_cos
    ds["hour_of_day_sin"] = data.rename({"data": "hour_of_day_sin"}).hour_of_day_sin

    return data


def join_list_data_array_to_batch_dataset(data_arrays: List[xr.DataArray]) -> xr.Dataset:
    """Join a list of xr.DataArrays into an xr.Dataset by concatenating on the example dim."""
    datasets = [
        convert_coordinates_to_indexes(data_arrays[i].to_dataset()) for i in range(len(data_arrays))
    ]

    return join_list_dataset_to_batch_dataset(datasets)
