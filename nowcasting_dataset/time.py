""" Time functions """
import logging
import warnings
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
import pvlib
import random

from nowcasting_dataset import geospatial, utils
from nowcasting_dataset.data_sources.datetime.datetime_model import Datetime

logger = logging.getLogger(__name__)


FIVE_MINUTES = pd.Timedelta("5 minutes")
THIRTY_MINUTES = pd.Timedelta("30 minutes")


def select_daylight_datetimes(
    datetimes: pd.DatetimeIndex, locations: Iterable[Tuple[float, float]], ghi_threshold: float = 10
) -> pd.DatetimeIndex:
    """
    Select only the day time datetimes

    Args:
        datetimes: DatetimeIndex to filter.
        locations: List of Tuples of x, y coordinates in OSGB projection.
        For example, use the four corners of the satellite imagery.
        ghi_threshold: Global horizontal irradiance threshold.
          (Watts per square meter?)

    Returns: datetimes for which the global horizontal irradiance (GHI) is above ghi_threshold across all locations.

    """
    ghi_for_all_locations = []
    for x, y in locations:
        lat, lon = geospatial.osgb_to_lat_lon(x, y)
        location = pvlib.location.Location(latitude=lat, longitude=lon)
        with warnings.catch_warnings():
            # PyTables triggers a DeprecationWarning in Numpy >= 1.20:
            # "tables/array.py:241: DeprecationWarning: `np.object` is a
            # deprecated alias for the builtin `object`."
            # See https://github.com/PyTables/PyTables/issues/898
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            clearsky = location.get_clearsky(datetimes)
        ghi = clearsky["ghi"]
        ghi_for_all_locations.append(ghi)

    ghi_for_all_locations = pd.concat(ghi_for_all_locations, axis="columns")
    max_ghi = ghi_for_all_locations.max(axis="columns")
    mask = max_ghi > ghi_threshold
    return datetimes[mask]


def intersection_of_datetimeindexes(indexes: List[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Get intersections of datetime indexes"""
    assert len(indexes) > 0
    intersection = indexes[0]
    for index in indexes[1:]:
        intersection = intersection.intersection(index)
    return intersection


def get_start_datetimes(
    datetimes: pd.DatetimeIndex, total_seq_len: int, max_gap: pd.Timedelta = THIRTY_MINUTES
) -> pd.DatetimeIndex:
    """Returns a datetime index of valid start datetimes.

    Valid start datetimes are those where there is certain to be
    at least total_seq_len contiguous timesteps ahead.

    For each contiguous_segment, remove the last total_seq_len datetimes,
    and then check the resulting segment is large enough.

    max_gap defines the threshold for what constitutes a 'gap' between
    contiguous segments.

    Throw away any timesteps in a sequence shorter than min_timesteps long.
    """
    assert len(datetimes) > 0
    min_timesteps = total_seq_len * 2
    assert min_timesteps > 1

    gap_mask = np.diff(datetimes) > max_gap
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately
    # *before* the gap.  e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05
    # then gap_indicies will be [1].  So we add 1 to gap_indices to get
    # segment_boundaries, an index into dt_index which identifies the _start_
    # of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(datetimes)]))

    start_dt_index = []
    start_i = 0
    for next_start_i in segment_boundaries:
        n_timesteps = next_start_i - start_i
        if n_timesteps >= min_timesteps:
            end_i = next_start_i + 1 - total_seq_len
            start_dt_index.append(datetimes[start_i:end_i])
        start_i = next_start_i

    assert len(start_dt_index) > 0

    return pd.DatetimeIndex(np.concatenate(start_dt_index))


def get_t0_datetimes(
    datetimes: pd.DatetimeIndex,
    total_seq_len: int,
    history_len: int,
    minute_delta: int = 5,
    max_gap: pd.Timedelta = FIVE_MINUTES,
) -> pd.DatetimeIndex:
    """
    Get datetimes for ML learning batches. T0 refers to the time 'now'.

    Args:
        datetimes: list of datetimes when data is available
        total_seq_len: total sequence length of data for ml model
        history_len: the number of historic timestemps
        minute_delta: the amount of minutes in one time step
        max_gap: The maximum allowed gap in the datetimes for it to be valid

    Returns: Datetimes that ml learning data can be built around.

    """
    logger.debug("Getting t0 datetimes")

    start_datetimes = get_start_datetimes(
        datetimes=datetimes, total_seq_len=total_seq_len, max_gap=max_gap
    )

    logger.debug("Adding history during to t0 datetimes")
    history_dur = timesteps_to_duration(history_len, minute_delta=minute_delta)
    t0_datetimes = start_datetimes + history_dur

    return t0_datetimes


def timesteps_to_duration(n_timesteps: int, minute_delta: int = 5) -> pd.Timedelta:
    """Change timesteps to a time duration"""
    assert n_timesteps >= 0
    return pd.Timedelta(n_timesteps * minute_delta, unit="minutes")


def datetime_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Make datetime features, hour_of_day and day_of_year

    Args:
        index: index of datestamps

    Returns: Example data with datetime features

    """
    features = {}
    features["hour_of_day"] = index.hour + (index.minute / 60)
    features["day_of_year"] = index.day_of_year
    return pd.DataFrame(features, index=index).astype(np.float32)


def datetime_features_in_example(index: pd.DatetimeIndex) -> Datetime:
    """
    Make datetime features with sin and cos

    Args:
        index: index of datestamps

    Returns: Example data with datetime features

    """
    dt_features = datetime_features(index)
    dt_features["hour_of_day"] /= 24
    dt_features["day_of_year"] /= 365
    dt_features = utils.sin_and_cos(dt_features)

    datetime_dict = {}
    for col_name, series in dt_features.iteritems():
        datetime_dict[col_name] = series.values

    datetime_dict["datetime_index"] = series.index.values

    return Datetime(**datetime_dict)


def fill_30_minutes_timestamps_to_5_minutes(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Fill a 30 minute index with 5 minute timestamps too. Note any gaps in 30 mins are not filled
    """
    # resample index to 5 mins
    index_5 = pd.Series(0, index=index).resample("5T")

    # calculate forward fill and backward fill
    index_5_ffill = index_5.ffill(limit=5)
    index_5_bfill = index_5.bfill(limit=5)

    # Time forward fill and backward together.
    # This means there will be NaNs if the original index is not in surrounding the values
    # for example:
    # index = [00:00:00, 01:00:00, 01:30:00, 02:00:00]
    #
    # index_5   ffill   bfill   ffill*bfill
    # 00:00:00  0       0       0
    # 00:05:00  0       NaN     NaN
    # 00:10:00  0       NaN     NaN
    # 00:15:00  0       NaN     NaN
    # 00:20:00  0       NaN     NaN
    # 00:25:00  0       NaN     NaN
    # 00:30:00  NaN     NaN     NaN
    # 00:35:00  NaN     0       NaN
    # 00:40:00  NaN     0       NaN
    # 00:45:00  NaN     0       NaN
    # 00:50:00  NaN     0       NaN
    # 00:55:00  NaN     0       NaN
    # 01:00:00  0       0       0
    # 01:05:00  0       0       0
    # 01:10:00  0       0       0
    # 01:15:00  0       0       0
    # 01:20:00  0       0       0
    # 01:25:00  0       0       0
    # 01:30:00  0       0       0
    # .....
    # 02:00:00  0       0       0
    index_with_gaps = index_5_ffill * index_5_bfill

    # drop nans and take index
    return index_with_gaps.dropna().index


def make_random_time_vectors(batch_size, seq_len_5_minutes, seq_len_30_minutes):
    """
    Make random time vectors

    1. t0_dt, Get random datetimes from 2019
    2. Exapnd t0_dt to make 5 and 30 mins sequences

    Args:
        batch_size: the batch size
        seq_len_5_minutes: the length of the sequence in 5 mins deltas
        seq_len_30_minutes: the length of the sequence in 30 mins deltas

    Returns:
        - t0_dt: [batch_size] random init datetimes
        - time_5: [batch_size, seq_len_5_minutes] random sequence of datetimes, with 5 mins deltas.
        t0_dt is in the middle of the sequence
        - time_30: [batch_size, seq_len_30_minutes] random sequence of datetimes, with 30 mins deltas.
        t0_dt is in the middle of the sequence
    """
    delta_5 = pd.Timedelta(minutes=5)
    delta_30 = pd.Timedelta(minutes=30)

    data_range = pd.date_range("2019-01-01", "2021-01-01", freq="5T")
    t0_dt = pd.Series(random.choices(data_range, k=batch_size))
    time_5 = (
        pd.DataFrame([t0_dt + i * delta_5 for i in range(seq_len_5_minutes)])
        - int(seq_len_5_minutes / 2) * delta_5
    )
    time_30 = (
        pd.DataFrame([t0_dt + i * delta_30 for i in range(seq_len_30_minutes)])
        - int(seq_len_30_minutes / 2) * delta_5
    )

    t0_dt = utils.to_numpy(t0_dt)
    time_5 = utils.to_numpy(time_5.T)
    time_30 = utils.to_numpy(time_30.T)

    return t0_dt, time_5, time_30
