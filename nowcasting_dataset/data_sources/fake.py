""" To make fake Datasets

Wanted to keep this out of the testing frame works, as other repos, might want to use this
"""
import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.dataset.xr_utils import convert_data_array_to_dataset


def create_image_array(
    dims=("time", "x", "y", "channels"),
    seq_length_5=19,
    image_size_pixels=64,
    number_channels=7,
):
    """ Create Satellite or NWP fake image data"""
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq="5T", periods=seq_length_5),
        "x": np.random.randint(low=0, high=1000, size=image_size_pixels),
        "y": np.random.randint(low=0, high=1000, size=image_size_pixels),
        "channels": np.arange(number_channels),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    image_data_array = xr.DataArray(
        abs(
            np.random.randn(
                seq_length_5,
                image_size_pixels,
                image_size_pixels,
                number_channels,
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
    """ Create gsp or pv fake dataset """
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

    return sun


def create_metadata_dataset() -> xr.Dataset:
    """ Create fake metadata dataset"""
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
    """ Create fake datetime dataset"""
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
