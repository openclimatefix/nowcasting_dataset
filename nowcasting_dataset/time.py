""" Time functions """
import logging
import random
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pvlib

from nowcasting_dataset import geospatial, utils

logger = logging.getLogger(__name__)


FIVE_MINUTES = pd.Timedelta("5 minutes")
THIRTY_MINUTES = pd.Timedelta("30 minutes")
ONE_HOUR = pd.Timedelta("1 hour")


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

    Returns: datetimes for which the global horizontal irradiance (GHI) is above ghi_threshold
    across all locations.

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


def single_period_to_datetime_index(period: pd.Series, freq: str) -> pd.DatetimeIndex:
    """Return a DatetimeIndex from period['start_dt'] to period['end_dt'] at frequency freq.

    Before computing the date_range, this function first takes the ceiling of the
    start_dt (at frequency `freq`); and takes the floor of end_dt.  For example,
    if `freq` is '5 minutes' and `start_dt` is 12:03, then the ceiling of `start_dt`
    will be 12:05.  This is done so that all the returned datetimes are aligned to `freq`
    (e.g. if `freq` is '5 minutes' then every returned datetime will be at 00, 05, ..., 55
    minutes past the hour.
    """
    start_dt = period["start_dt"].ceil(freq)
    end_dt = period["end_dt"].floor(freq)
    return pd.date_range(start_dt, end_dt, freq=freq)


def time_periods_to_datetime_index(time_periods: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    """Convert a DataFrame of time periods into a DatetimeIndex at a particular frequency.

    See the docstring of intersection_of_2_dataframes_of_periods() for more details.
    """
    assert len(time_periods) > 0
    dt_index = single_period_to_datetime_index(time_periods.iloc[0], freq=freq)
    for _, time_period in time_periods.iloc[1:].iterrows():
        new_dt_index = single_period_to_datetime_index(time_period, freq=freq)
        dt_index = dt_index.union(new_dt_index)
    return dt_index


def intersection_of_multiple_dataframes_of_periods(
    time_periods: list[pd.DataFrame],
) -> pd.DataFrame:
    """Finds the intersection of a list of time periods.

    See the docstring of intersection_of_2_dataframes_of_periods() for more details.
    """
    assert len(time_periods) > 0
    if len(time_periods) == 1:
        return time_periods[0]
    intersection = intersection_of_2_dataframes_of_periods(time_periods[0], time_periods[1])
    for time_period in time_periods[2:]:
        intersection = intersection_of_2_dataframes_of_periods(intersection, time_period)
    return intersection


def intersection_of_2_dataframes_of_periods(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Finds the intersection of two pd.DataFrames of time periods.

    Each row of each pd.DataFrame represents a single time period.  Each pd.DataFrame has
    two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

    A typical use-case is that each pd.DataFrame represents all the time periods where
    a `DataSource` has contiguous, valid data.

    Here's a graphical example of two pd.DataFrames of time periods and their intersection:

                 ----------------------> TIME ->---------------------
               a: |-----|   |----|     |----------|     |-----------|
               b:    |--------|                       |----|    |---|
    intersection:    |--|   |-|                         |--|    |---|

    Args:
        a, b: pd.DataFrame where each row represents a time period.  The pd.DataFrame has
        two columns: start_dt and end_dt.

    Returns:
        Sorted list of intersecting time periods represented as a pd.DataFrame with two columns:
        start_dt and end_dt.
    """
    if a.empty or b.empty:
        return pd.DataFrame(columns=["start_dt", "end_dt"])

    all_intersecting_periods = []
    for a_period in a.itertuples():
        # Five ways in which two periods may overlap:
        # a: |----| or |---|   or  |---| or   |--|   or |-|
        # b:  |--|       |---|   |---|      |------|    |-|
        # In all five, `a` must always start before `b` ends,
        # and `a` must always end after `b` starts:
        overlapping_periods = b[(a_period.start_dt < b.end_dt) & (a_period.end_dt > b.start_dt)]

        # There are two ways in which two periods may *not* overlap:
        # a: |---|        or        |---|
        # b:       |---|      |---|
        # `overlapping` will not include periods which do *not* overlap.

        # Now find the intersection of each period in `overlapping_periods` with
        # the period from `a` that starts at `a_start_dt` and ends at `a_end_dt`.
        # We do this by clipping each row of `overlapping_periods`
        # to start no earlier than `a_start_dt`, and end no later than `a_end_dt`.

        # First, make a copy, so we don't clip the underlying data in `b`.
        intersecting_periods = overlapping_periods.copy()
        intersecting_periods.start_dt.clip(lower=a_period.start_dt, inplace=True)
        intersecting_periods.end_dt.clip(upper=a_period.end_dt, inplace=True)

        all_intersecting_periods.append(intersecting_periods)

    all_intersecting_periods = pd.concat(all_intersecting_periods)
    return all_intersecting_periods.sort_values(by="start_dt").reset_index(drop=True)


def get_contiguous_time_periods(
    datetimes: pd.DatetimeIndex,
    min_seq_length: int,
    max_gap_duration: pd.Timedelta = THIRTY_MINUTES,
) -> pd.DataFrame:
    """Returns a pd.DataFrame where each row records the boundary of a contiguous time periods.

    Args:
      datetimes: The pd.DatetimeIndex of the timeseries. Must be sorted.
      min_seq_length: Sequences of min_seq_length or shorter will be discarded.  Typically, this
        would be set to the `total_seq_length` of each machine learning example.
      max_gap_duration: If any pair of consecutive `datetimes` is more than `max_gap_duration`
        apart, then this pair of `datetimes` will be considered a "gap" between two contiguous
        sequences. Typically, `max_gap_duration` would be set to the sample period of
        the timeseries.

    Returns:
      pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    assert len(datetimes) > 0
    assert min_seq_length > 1
    assert datetimes.is_monotonic_increasing
    assert datetimes.is_unique

    # Find indices of gaps larger than max_gap:
    gap_mask = np.diff(datetimes) > max_gap_duration
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately
    # *before* the gap.  e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05
    # then gap_indicies will be [1].  So we add 1 to gap_indices to get
    # segment_boundaries, an index into dt_index which identifies the _start_
    # of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(datetimes)]))

    periods: List[Dict[str, pd.Timestamp]] = []
    start_i = 0
    for next_start_i in segment_boundaries:
        n_timesteps = next_start_i - start_i
        if n_timesteps > min_seq_length:
            end_i = next_start_i - 1
            period = {"start_dt": datetimes[start_i], "end_dt": datetimes[end_i]}
            periods.append(period)
        start_i = next_start_i

    assert len(periods) > 0

    return pd.DataFrame(periods)


def make_random_time_vectors(
    batch_size,
    seq_length_5_minutes,
    seq_length_30_minutes,
    seq_length_60_minutes: Optional[int] = None,
) -> dict:
    """
    Make random time vectors

    1. t0_dt, Get random datetimes from 2019
    2. Exapnd t0_dt to make 5 and 30 mins sequences

    Args:
        batch_size: the batch size
        seq_length_5_minutes: the length of the sequence in 5 mins deltas
        seq_length_30_minutes: the length of the sequence in 30 mins deltas
        seq_length_60_minutes: the length of the sequence in 60 mins deltas

    Returns:
        - t0_dt: [batch_size] random init datetimes
        - time_5: [batch_size, seq_length_5_minutes] random sequence of datetimes, with
            5 mins deltas. t0_dt is in the middle of the sequence
        - time_30: [batch_size, seq_length_30_minutes] random sequence of datetimes, with
            30 mins deltas. t0_dt is in the middle of the sequence
        - time_60 : [batch_size, seq_length_60_minutes] random sequence of datetimes, with
            60 mins deltas. t0_dt is in the middle of the sequence
    """
    delta_5 = pd.Timedelta(minutes=5)
    delta_30 = pd.Timedelta(minutes=30)

    if seq_length_60_minutes is None:
        seq_length_60_minutes = int(seq_length_5_minutes / 12)

    data_range = pd.date_range("2019-01-01", "2021-01-01", freq="5T")
    t0_dt = pd.Series(random.choices(data_range, k=batch_size))
    time_5 = (
        pd.DataFrame([t0_dt + i * delta_5 for i in range(seq_length_5_minutes)])
        - int(seq_length_5_minutes / 2) * delta_5
    )
    time_30 = (
        pd.DataFrame([t0_dt + i * delta_30 for i in range(seq_length_30_minutes)])
        - int(seq_length_30_minutes / 2) * delta_5
    )
    time_60 = (
        pd.DataFrame([t0_dt + i * ONE_HOUR for i in range(seq_length_60_minutes)])
        - int(seq_length_60_minutes / 2) * delta_5
    )

    return {
        "t0_dt": utils.to_numpy(t0_dt).astype(np.int32),
        "time_5": utils.to_numpy(time_5.T).astype(np.int32),
        "time_30": utils.to_numpy(time_30.T).astype(np.int32),
        "time_60": utils.to_numpy(time_60.T).astype(np.int32),
    }
