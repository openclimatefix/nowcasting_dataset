from typing import TypedDict
import pandas as pd
from nowcasting_dataset.consts import *
import numpy as np
from numbers import Number


class Example(TypedDict):
    """Simple class for structuring data for each ML example.

    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of the fields
         in each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple,
    which would provide runtime checks, but the deal-breaker with Tuples is
    that they're immutable so we cannot change the values in the transforms.
    """

    # timestamp of now. In this data set there will be both
    # - historic data before this timestamp,
    # - and future data after this timestamp
    # shape is [batch_size,]
    t0_dt = Array

    # IMAGES
    # Shape: [batch_size,] seq_length, width, height, channel
    sat_data: Array
    sat_x_coords: Array  #: OSGB geo-spatial coordinates.
    sat_y_coords: Array

    #: PV yield from all PV systems in the region of interest (ROI).
    #: Includes central PV system, which will always be the first entry.
    #: shape = [batch_size, ] seq_length, n_pv_systems_per_example
    pv_yield: Array

    # PV azimuth and elevation angles i.e where the sun is.
    #: shape = [batch_size, ] seq_length, n_pv_systems_per_example
    pv_azimuth_angle: Array
    pv_elevation_angle: Array

    #: PV identification.
    #: shape = [batch_size, ] n_pv_systems_per_example
    pv_system_id: Array
    pv_system_row_number: Array  #: In the range [0, len(pv_metadata)].

    #: PV system geographical location (in OSGB coords).
    #: shape = [batch_size, ] n_pv_systems_per_example
    pv_system_x_coords: Array
    pv_system_y_coords: Array
    pv_datetime_index: Array  #: shape = [batch_size, ] seq_length

    # Numerical weather predictions (NWPs)
    nwp: Array  #: Shape: [batch_size,] channel, seq_length, width, height
    nwp_x_coords: Array
    nwp_y_coords: Array

    # METADATA
    x_meters_center: Number  #: In OSGB coordinations
    y_meters_center: Number  #: In OSGB coordinations

    # Datetimes (abbreviated to "dt")
    # At 5-minutes past the hour {0, 5, ..., 55}
    # *not* the {4, 9, ..., 59} timings of the satellite imagery.
    # Datetimes become Unix epochs (UTC) represented as int64 just before being
    # passed into the ML model.
    # t0_dt is 'now', the most recent observation.
    sat_datetime_index: Array
    nwp_target_time: Array
    hour_of_day_sin: Array  #: Shape: [batch_size,] seq_length
    hour_of_day_cos: Array
    day_of_year_sin: Array
    day_of_year_cos: Array

    #: GSP PV yield from all GSP in the region of interest (ROI).
    # : Includes central GSP, which will always be the first entry. This will be a numpy array of values.
    gsp_yield: Array  #: shape = [batch_size, ] seq_length, n_gsp_systems_per_example
    # GSP identification.
    gsp_id: Array  #: shape = [batch_size, ] n_pv_systems_per_example
    #: GSP geographical location (in OSGB coords).
    gsp_x_coords: Array  #: shape = [batch_size, ] n_pv_systems_per_example
    gsp_y_coords: Array  #: shape = [batch_size, ] n_pv_systems_per_example
    gsp_datetime_index: Array  #: shape = [batch_size, ] seq_length

    # if the centroid type is a GSP, or a PV system
    object_at_center: str  #: shape = [batch_size, ]


def to_numpy(example: Example) -> Example:
    for key, value in example.items():
        if isinstance(value, xr.DataArray):
            # TODO: Use to_numpy() or as_numpy(), introduced in xarray v0.19?
            value = value.data

        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.values
        elif isinstance(value, pd.DatetimeIndex):
            value = value.values.astype("datetime64[s]").astype(np.int32)
        elif isinstance(value, pd.Timestamp):
            value = np.int32(value.timestamp())
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.datetime64):
            value = value.astype("datetime64[s]").astype(np.int32)

        example[key] = value
    return example


def validate_example(
    data: Example,
    seq_len_30_minutes: int,
    seq_len_5_minutes: int,
    sat_image_size: int = 64,
    n_sat_channels: int = 1,
    nwp_image_size: int = 0,
    n_nwp_channels: int = 1,
    n_pv_systems_per_example: int = DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    n_gsp_per_example: int = DEFAULT_N_GSP_PER_EXAMPLE,
):
    """
    Validate the size and shape of the data
    Args:
        data: Typed dictionary of the data
        seq_len_30_minutes: the length of the sequence for 30 minutely data
        seq_len_5_minutes: the length of the sequence for 5 minutely data
        sat_image_size: the satellite image size
        n_sat_channels: the number of satellite channgles
        nwp_image_size: the nwp image size
        n_nwp_channels: the number of nwp channels
        n_pv_systems_per_example: the number pv systems with nan padding
        n_gsp_per_example: the number gsp systems with nan padding
    """

    assert len(data[GSP_ID]) == n_gsp_per_example
    n_gsp_system_id = len(data[GSP_ID])
    assert data[GSP_YIELD].shape == (seq_len_30_minutes, n_gsp_system_id)
    assert len(data[GSP_X_COORDS]) == n_gsp_system_id
    assert len(data[GSP_Y_COORDS]) == n_gsp_system_id
    assert len(data[GSP_DATETIME_INDEX]) == seq_len_30_minutes

    assert data[OBJECT_AT_CENTER] == "gsp"
    assert type(data["x_meters_center"]) == np.float64
    assert type(data["y_meters_center"]) == np.float64

    n_pv_systems = len(data[PV_SYSTEM_ID][~np.isnan(data[PV_SYSTEM_ID])])

    assert len(data[PV_SYSTEM_ID]) == n_pv_systems_per_example
    assert data[PV_YIELD].shape == (seq_len_5_minutes, n_pv_systems_per_example)
    assert len(data[PV_SYSTEM_X_COORDS]) == n_pv_systems_per_example
    assert len(data[PV_SYSTEM_Y_COORDS]) == n_pv_systems_per_example
    assert len(data[PV_SYSTEM_ROW_NUMBER][~np.isnan(data[PV_SYSTEM_ROW_NUMBER])]) == n_pv_systems
    assert len(data[PV_SYSTEM_ROW_NUMBER][~np.isnan(data[PV_SYSTEM_ROW_NUMBER])]) == n_pv_systems

    if PV_AZIMUTH_ANGLE in data.keys():
        assert data[PV_AZIMUTH_ANGLE].shape == (seq_len_5_minutes, n_pv_systems_per_example)
    if PV_AZIMUTH_ANGLE in data.keys():
        assert data[PV_ELEVATION_ANGLE].shape == (seq_len_5_minutes, n_pv_systems_per_example)

    assert data["sat_data"].shape == (
        seq_len_5_minutes,
        sat_image_size,
        sat_image_size,
        n_sat_channels,
    )
    assert len(data["sat_x_coords"]) == sat_image_size
    assert len(data["sat_y_coords"]) == sat_image_size
    assert len(data["sat_datetime_index"]) == seq_len_5_minutes

    assert data["nwp"].shape == (n_nwp_channels, seq_len_5_minutes, nwp_image_size, nwp_image_size)
    assert len(data["nwp_x_coords"]) == nwp_image_size
    assert len(data["nwp_y_coords"]) == nwp_image_size
    assert len(data["nwp_target_time"]) == seq_len_5_minutes

    for feature in DATETIME_FEATURE_NAMES:
        assert len(data[feature]) == seq_len_5_minutes
