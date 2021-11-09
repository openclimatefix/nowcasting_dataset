""" To make fake Datasets

Wanted to keep this out of the testing frame works, as other repos, might want to use this
"""
import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.consts import NWP_VARIABLE_NAMES, SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.metadata.metadata_model import Metadata
from nowcasting_dataset.data_sources.nwp.nwp_model import NWP
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.data_sources.topographic.topographic_model import Topographic
from nowcasting_dataset.dataset.xr_utils import (
    convert_data_array_to_dataset,
    join_list_data_array_to_batch_dataset,
    join_list_dataset_to_batch_dataset,
)


def gsp_fake(
    batch_size,
    seq_length_30,
    n_gsp_per_batch,
):
    """Create fake data"""
    # make batch of arrays
    xr_arrays = [
        create_gsp_pv_dataset(
            seq_length=seq_length_30,
            freq="30T",
            number_of_systems=n_gsp_per_batch,
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_dataset_to_batch_dataset(xr_arrays)

    return GSP(xr_dataset)


def metadata_fake(batch_size):
    """Make a xr dataset"""
    xr_arrays = [create_metadata_dataset() for _ in range(batch_size)]

    # make dataset
    xr_dataset = join_list_dataset_to_batch_dataset(xr_arrays)

    return Metadata(xr_dataset)


def nwp_fake(
    batch_size=32,
    seq_length_5=19,
    image_size_pixels=64,
    number_nwp_channels=7,
) -> NWP:
    """Create fake data"""
    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length_5=seq_length_5,
            image_size_pixels=image_size_pixels,
            channels=NWP_VARIABLE_NAMES[0:number_nwp_channels],
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
    xr_arrays = [
        create_gsp_pv_dataset(
            seq_length=seq_length_5,
            freq="5T",
            number_of_systems=n_pv_systems_per_batch,
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_dataset_to_batch_dataset(xr_arrays)

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
            seq_length_5=seq_length_5,
            image_size_pixels=satellite_image_size_pixels,
            channels=SAT_VARIABLE_NAMES[0:number_satellite_channels],
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return Satellite(xr_dataset)


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
        )
        for _ in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return Topographic(xr_dataset)


def create_image_array(
    dims=("time", "x", "y", "channels"),
    seq_length_5=19,
    image_size_pixels=64,
    channels=SAT_VARIABLE_NAMES,
):
    """Create Satellite or NWP fake image data"""
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq="5T", periods=seq_length_5),
        "x": np.random.randint(low=0, high=1000, size=image_size_pixels),
        "y": np.random.randint(low=0, high=1000, size=image_size_pixels),
        "channels": np.array(channels),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    image_data_array = xr.DataArray(
        abs(
            np.random.randn(
                seq_length_5,
                image_size_pixels,
                image_size_pixels,
                len(channels),
            )
        ),
        coords=coords,
    )  # Fake data for testing!
    return image_data_array


def create_gsp_pv_dataset(
    dims=("time", "id"),
    freq="5T",
    seq_length=19,
    number_of_systems=128,
):
    """Create gsp or pv fake dataset"""
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq=freq, periods=seq_length),
        "id": np.random.randint(low=0, high=1000, size=number_of_systems),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    data_array = xr.DataArray(
        np.random.randn(
            seq_length,
            number_of_systems,
        ),
        coords=coords,
    )  # Fake data for testing!

    data = convert_data_array_to_dataset(data_array)

    x_coords = xr.DataArray(
        data=np.sort(np.random.randn(number_of_systems)),
        dims=["id_index"],
        coords=dict(
            id_index=range(number_of_systems),
        ),
    )

    y_coords = xr.DataArray(
        data=np.sort(np.random.randn(number_of_systems)),
        dims=["id_index"],
        coords=dict(
            id_index=range(number_of_systems),
        ),
    )

    data["x_coords"] = x_coords
    data["y_coords"] = y_coords

    # Add 1000 to the id numbers for the row numbers.
    # This is a quick way to make sure row number is different from id,
    data["pv_system_row_number"] = data["id"] + 1000

    data.__setitem__("data", data.data.clip(min=0))

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

    data = convert_data_array_to_dataset(data_array)
    sun = data.rename({"data": "elevation"})
    sun["azimuth"] = data.data

    sun.__setitem__("azimuth", sun.azimuth.clip(min=0, max=360))
    sun.__setitem__("elevation", sun.elevation.clip(min=-90, max=90))

    return sun


def create_metadata_dataset() -> xr.Dataset:
    """Create fake metadata dataset"""
    d = {
        "dims": ("t0_dt",),
        "data": pd.date_range("2021-01-01", freq="5T", periods=1) + pd.Timedelta("30T"),
    }

    data = convert_data_array_to_dataset(xr.DataArray.from_dict(d))

    for v in ["x_meters_center", "y_meters_center", "object_at_center_label"]:
        d: dict = {"dims": ("t0_dt",), "data": [np.random.randint(0, 1000)]}
        d: xr.Dataset = convert_data_array_to_dataset(xr.DataArray.from_dict(d)).rename({"data": v})
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

    data = convert_data_array_to_dataset(data_array)

    ds = data.rename({"data": "day_of_year_cos"})
    ds["day_of_year_sin"] = data.rename({"data": "day_of_year_sin"}).day_of_year_sin
    ds["hour_of_day_cos"] = data.rename({"data": "hour_of_day_cos"}).hour_of_day_cos
    ds["hour_of_day_sin"] = data.rename({"data": "hour_of_day_sin"}).hour_of_day_sin

    return data
