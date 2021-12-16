""" To make fake Datasets

Wanted to keep this out of the testing frame works, as other repos, might want to use this
"""

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import NWP_VARIABLE_NAMES, SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.fake.coordinates import (
    create_random_point_coordinates_osgb,
    make_random_image_coords_osgb,
    make_random_x_and_y_osgb_centers,
)
from nowcasting_dataset.data_sources.fake.utils import (
    join_list_data_array_to_batch_dataset,
    make_t0_datetimes_utc,
)
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


def make_fake_batch(configuration: Configuration) -> dict:
    """Make fake batch"""
    batch_size = configuration.process.batch_size

    metadata = metadata_fake(batch_size=batch_size)

    return dict(
        metadata=metadata,
        satellite=satellite_fake(configuration=configuration, metadata=metadata),
        hrvsatellite=hrv_satellite_fake(
            configuration=configuration,
            metadata=metadata,
        ),
        opticalflow=optical_flow_fake(
            configuration=configuration,
            metadata=metadata,
        ),
        nwp=nwp_fake(
            configuration=configuration,
            metadata=metadata,
        ),
        pv=pv_fake(
            configuration=configuration,
            metadata=metadata,
        ),
        gsp=gsp_fake(
            configuration=configuration,
            metadata=metadata,
        ),
        sun=sun_fake(
            batch_size=batch_size,
            seq_length_5=configuration.input_data.sun.seq_length_5_minutes,
            metadata=metadata,
        ),
        topographic=topographic_fake(
            batch_size=batch_size,
            metadata=metadata,
            image_size_pixels=configuration.input_data.topographic.topographic_image_size_pixels,
        ),
    )


def gsp_fake(
    configuration: Configuration = None,
    metadata: Optional[Metadata] = None,
):
    """Create fake data"""

    if configuration is None:
        configuration = Configuration()
        configuration.input_data = Configuration().input_data.set_all_to_defaults()

    batch_size = configuration.process.batch_size
    seq_length_30 = configuration.input_data.gsp.seq_length_30_minutes
    history_seq_length = configuration.input_data.gsp.history_seq_length_30_minutes
    n_gsp_per_batch = configuration.input_data.gsp.n_gsp_per_example

    if metadata is None:
        t0_datetimes_utc = make_t0_datetimes_utc(batch_size)
        x_centers_osgb, y_centers_osgb = make_random_x_and_y_osgb_centers(batch_size)
    else:
        t0_datetimes_utc = metadata.t0_datetime_utc
        x_centers_osgb = metadata.x_center_osgb
        y_centers_osgb = metadata.y_center_osgb

    # make batch of arrays
    xr_datasets = [
        create_gsp_pv_dataset(
            seq_length=seq_length_30,
            history_seq_length=history_seq_length,
            freq="30T",
            number_of_systems=n_gsp_per_batch,
            t0_datetime_utc=t0_datetimes_utc[i],
            x_center_osgb=x_centers_osgb[i],
            y_center_osgb=y_centers_osgb[i],
            id_limit=338,
        )
        for i in range(batch_size)
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
    t0_datetimes_utc = make_t0_datetimes_utc(batch_size)

    metadata_dict = {}
    metadata_dict["batch_size"] = batch_size
    metadata_dict["x_center_osgb"] = list(x_centers_osgb)
    metadata_dict["y_center_osgb"] = list(y_centers_osgb)
    metadata_dict["t0_datetime_utc"] = list(t0_datetimes_utc)

    return Metadata(**metadata_dict)


def nwp_fake(
    configuration: Configuration = None,
    metadata: Optional[Metadata] = None,
) -> NWP:
    """Create fake data"""

    if configuration is None:
        configuration = Configuration()
        configuration.input_data = Configuration().input_data.set_all_to_defaults()

    batch_size = configuration.process.batch_size
    image_size_pixels = configuration.input_data.nwp.nwp_image_size_pixels
    history_seq_length = configuration.input_data.nwp.history_seq_length_60_minutes
    seq_length_60 = configuration.input_data.nwp.seq_length_60_minutes
    number_nwp_channels = len(configuration.input_data.nwp.nwp_channels)

    if metadata is None:
        t0_datetimes_utc = make_t0_datetimes_utc(batch_size)
        x_centers_osgb, y_centers_osgb = make_random_x_and_y_osgb_centers(batch_size)
    else:
        t0_datetimes_utc = metadata.t0_datetime_utc
        x_centers_osgb = metadata.x_center_osgb
        y_centers_osgb = metadata.y_center_osgb

    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length=seq_length_60,
            history_seq_length=history_seq_length,
            image_size_pixels=image_size_pixels,
            channels=NWP_VARIABLE_NAMES[0:number_nwp_channels],
            freq="60T",
            t0_datetime_utc=t0_datetimes_utc[i],
            x_center_osgb=x_centers_osgb[i],
            y_center_osgb=y_centers_osgb[i],
        )
        for i in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    xr_dataset["init_time"] = xr_dataset.time[:, 0]

    return NWP(xr_dataset)


def pv_fake(
    configuration: Configuration = None,
    metadata: Optional[Metadata] = None,
):
    """Create fake data"""

    if configuration is None:
        configuration = Configuration()
        configuration.input_data = Configuration().input_data.set_all_to_defaults()

    batch_size = configuration.process.batch_size
    seq_length_5 = configuration.input_data.pv.seq_length_5_minutes
    history_seq_length = configuration.input_data.pv.history_seq_length_5_minutes
    n_pv_systems_per_batch = configuration.input_data.pv.n_pv_systems_per_example

    if metadata is None:
        t0_datetimes_utc = make_t0_datetimes_utc(batch_size)
        x_centers_osgb, y_centers_osgb = make_random_x_and_y_osgb_centers(batch_size)
    else:
        t0_datetimes_utc = metadata.t0_datetime_utc
        x_centers_osgb = metadata.x_center_osgb
        y_centers_osgb = metadata.y_center_osgb

    # make batch of arrays
    xr_datasets = [
        create_gsp_pv_dataset(
            seq_length=seq_length_5,
            history_seq_length=history_seq_length,
            freq="5T",
            number_of_systems=n_pv_systems_per_batch,
            time_dependent_capacity=False,
            t0_datetime_utc=t0_datetimes_utc[i],
            x_center_osgb=x_centers_osgb[i],
            y_center_osgb=y_centers_osgb[i],
        )
        for i in range(batch_size)
    ]

    # change dimensions to dimension indexes
    xr_datasets = convert_coordinates_to_indexes_for_list_datasets(xr_datasets)

    # make dataset
    xr_dataset = join_list_dataset_to_batch_dataset(xr_datasets)

    return PV(xr_dataset)


def satellite_fake(
    configuration: Configuration = None,
    metadata: Optional[Metadata] = None,
) -> Satellite:
    """Create fake data"""

    if configuration is None:
        configuration = Configuration()
        configuration.input_data = Configuration().input_data.set_all_to_defaults()

    batch_size = configuration.process.batch_size
    image_size_pixels = configuration.input_data.satellite.satellite_image_size_pixels
    history_seq_length = configuration.input_data.satellite.history_seq_length_5_minutes
    seq_length_5 = configuration.input_data.satellite.seq_length_5_minutes
    number_satellite_channels = len(configuration.input_data.satellite.satellite_channels)

    if metadata is None:
        t0_datetimes_utc = make_t0_datetimes_utc(batch_size)
        x_centers_osgb, y_centers_osgb = make_random_x_and_y_osgb_centers(batch_size)
    else:
        t0_datetimes_utc = metadata.t0_datetime_utc
        x_centers_osgb = metadata.x_center_osgb
        y_centers_osgb = metadata.y_center_osgb

    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length=seq_length_5,
            history_seq_length=history_seq_length,
            image_size_pixels=image_size_pixels,
            channels=SAT_VARIABLE_NAMES[1:number_satellite_channels],
            t0_datetime_utc=t0_datetimes_utc[i],
            x_center_osgb=x_centers_osgb[i],
            y_center_osgb=y_centers_osgb[i],
        )
        for i in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return Satellite(xr_dataset)


def hrv_satellite_fake(
    configuration: Configuration = None,
    metadata: Optional[Metadata] = None,
) -> Satellite:
    """Create fake data"""

    if configuration is None:
        configuration = Configuration()
        configuration.input_data = Configuration().input_data.set_all_to_defaults()

    batch_size = configuration.process.batch_size
    image_size_pixels = configuration.input_data.hrvsatellite.hrvsatellite_image_size_pixels
    history_seq_length = configuration.input_data.hrvsatellite.history_seq_length_5_minutes
    seq_length_5 = configuration.input_data.hrvsatellite.seq_length_5_minutes

    if metadata is None:
        t0_datetimes_utc = make_t0_datetimes_utc(batch_size)
        x_centers_osgb, y_centers_osgb = make_random_x_and_y_osgb_centers(batch_size)
    else:
        t0_datetimes_utc = metadata.t0_datetime_utc
        x_centers_osgb = metadata.x_center_osgb
        y_centers_osgb = metadata.y_center_osgb

    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length=seq_length_5,
            history_seq_length=history_seq_length,
            image_size_pixels=image_size_pixels * 3,  # HRV images are 3x other images
            channels=SAT_VARIABLE_NAMES[0:1],
            t0_datetime_utc=t0_datetimes_utc[i],
            x_center_osgb=x_centers_osgb[i],
            y_center_osgb=y_centers_osgb[i],
        )
        for i in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return HRVSatellite(xr_dataset)


def optical_flow_fake(
    configuration: Configuration = None,
    metadata: Optional[Metadata] = None,
) -> OpticalFlow:
    """Create fake data"""

    if configuration is None:
        configuration = Configuration()
        configuration.input_data = Configuration().input_data.set_all_to_defaults()

    batch_size = configuration.process.batch_size
    image_size_pixels = configuration.input_data.opticalflow.opticalflow_input_image_size_pixels
    history_seq_length = configuration.input_data.opticalflow.history_seq_length_5_minutes
    seq_length_5 = configuration.input_data.opticalflow.seq_length_5_minutes
    number_satellite_channels = len(configuration.input_data.opticalflow.opticalflow_channels)

    if metadata is None:
        t0_datetimes_utc = make_t0_datetimes_utc(batch_size)
        x_centers_osgb, y_centers_osgb = make_random_x_and_y_osgb_centers(batch_size)
    else:
        t0_datetimes_utc = metadata.t0_datetime_utc
        x_centers_osgb = metadata.x_center_osgb
        y_centers_osgb = metadata.y_center_osgb

    # make batch of arrays
    xr_arrays = [
        create_image_array(
            seq_length=seq_length_5,
            history_seq_length=history_seq_length,
            freq="5T",
            image_size_pixels=image_size_pixels,
            channels=SAT_VARIABLE_NAMES[0:number_satellite_channels],
            t0_datetime_utc=t0_datetimes_utc[i],
            x_center_osgb=x_centers_osgb[i],
            y_center_osgb=y_centers_osgb[i],
        )
        for i in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return OpticalFlow(xr_dataset)


def sun_fake(
    batch_size,
    seq_length_5,
    metadata: Optional[Metadata] = None,
):
    """Create fake data"""

    if metadata is None:
        t0_datetimes_utc = make_t0_datetimes_utc(batch_size)
    else:
        t0_datetimes_utc = metadata.t0_datetime_utc

    # create dataset with both azimuth and elevation, index with time
    # make batch of arrays
    xr_arrays = [
        create_sun_dataset(seq_length=seq_length_5, t0_datetime_utc=t0_datetimes_utc[i])
        for i in range(batch_size)
    ]

    # make dataset
    xr_dataset = join_list_dataset_to_batch_dataset(xr_arrays)

    return Sun(xr_dataset)


def topographic_fake(batch_size, image_size_pixels, metadata: Optional[Metadata] = None):
    """Create fake data"""

    if metadata is None:
        x_centers_osgb, y_centers_osgb = make_random_x_and_y_osgb_centers(batch_size)
    else:
        x_centers_osgb = metadata.x_center_osgb
        y_centers_osgb = metadata.y_center_osgb

    # make batch of arrays
    xr_arrays = []
    for i in range(batch_size):

        x, y = make_random_image_coords_osgb(
            size=image_size_pixels,
            x_center_osgb=x_centers_osgb[i],
            y_center_osgb=y_centers_osgb[i],
        )

        xr_array = xr.DataArray(
            data=np.random.randn(
                image_size_pixels,
                image_size_pixels,
            ),
            dims=["x", "y"],
            coords=dict(
                x=x,
                y=y,
            ),
            name="data",
        )
        xr_arrays.append(xr_array)

    # make dataset
    xr_dataset = join_list_data_array_to_batch_dataset(xr_arrays)

    return Topographic(xr_dataset)


def create_image_array(
    dims=("time", "x", "y", "channels"),
    seq_length=19,
    history_seq_length=5,
    image_size_pixels=64,
    channels=SAT_VARIABLE_NAMES,
    freq="5T",
    t0_datetime_utc: Optional = None,
    x_center_osgb: Optional = None,
    y_center_osgb: Optional = None,
):
    """Create Satellite or NWP fake image data"""

    if t0_datetime_utc is None:
        t0_datetime_utc = make_t0_datetimes_utc(batch_size=1)[0]

    x, y = make_random_image_coords_osgb(
        size=image_size_pixels, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb
    )

    time = pd.date_range(end=t0_datetime_utc, freq=freq, periods=history_seq_length + 1).union(
        pd.date_range(start=t0_datetime_utc, freq=freq, periods=seq_length - history_seq_length)
    )

    ALL_COORDS = {
        "time": time,
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
    history_seq_length=5,
    number_of_systems=128,
    time_dependent_capacity: bool = True,
    t0_datetime_utc: Optional = None,
    x_center_osgb: Optional = None,
    y_center_osgb: Optional = None,
    id_limit: int = 1000,
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
        history_seq_length: The historic length
        t0_datetime_utc: the time now, if this is not given, a random one will be made.
        x_center_osgb: the x center of the example. If not given, a random one will be made.
        y_center_osgb: the y center of the example. If not given, a random one will be made.
        id_limit: The maximum id number allowed. For example for GSP it should be 338

    Returns: xr.Dataset of fake data

    """

    if t0_datetime_utc is None:
        t0_datetime_utc = make_t0_datetimes_utc(batch_size=1)[0]

    time = pd.date_range(end=t0_datetime_utc, freq=freq, periods=history_seq_length + 1).union(
        pd.date_range(start=t0_datetime_utc, freq=freq, periods=seq_length - history_seq_length)
    )

    ALL_COORDS = {
        "time": time,
        "id": np.random.choice(range(id_limit), number_of_systems, replace=False),
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
    x, y = create_random_point_coordinates_osgb(
        size=number_of_systems, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb
    )

    x_coords = xr.DataArray(
        data=x,
        dims=["id"],
    )

    y_coords = xr.DataArray(
        data=y,
        dims=["id"],
    )

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
    t0_datetime_utc: Optional = None,
) -> xr.Dataset:
    """
    Create sun fake dataset

    Args:
        dims: # TODO
        freq: # TODO
        seq_length: # TODO
        t0_datetime_utc: # TODO

    Returns: # TODO

    """

    if t0_datetime_utc is None:
        t0_datetime_utc = make_t0_datetimes_utc(batch_size=1)[0]

    ALL_COORDS = {
        "time": pd.date_range(t0_datetime_utc, freq=freq, periods=seq_length),
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
