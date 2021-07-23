import pandas as pd
import numpy as np
from typing import Iterable, Tuple, List
from nowcasting_dataset import geospatial, utils
from nowcasting_dataset.example import Example
import warnings
import pvlib


FIVE_MINUTES = pd.Timedelta('5 minutes')


def select_daylight_datetimes(
        datetimes: pd.DatetimeIndex,
        locations: Iterable[Tuple[float, float]],
        ghi_threshold: float = 10) -> pd.DatetimeIndex:
    """Returns datetimes for which the global horizontal irradiance
    (GHI) is above ghi_threshold across all locations.

    Args:
      dt_index: DatetimeIndex to filter.
      locations: List of Tuples of x, y coordinates in OSGB projection.
        For example, use the four corners of the satellite imagery.
      ghi_threshold: Global horizontal irradiance threshold.
          (Watts per square meter?)

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
        ghi = clearsky['ghi']
        ghi_for_all_locations.append(ghi)

    ghi_for_all_locations = pd.concat(ghi_for_all_locations, axis='columns')
    max_ghi = ghi_for_all_locations.max(axis='columns')
    mask = max_ghi > ghi_threshold
    return datetimes[mask]


def intersection_of_datetimeindexes(
        indexes: List[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    assert len(indexes) > 0
    intersection = indexes[0]
    for index in indexes[1:]:
        intersection = intersection.intersection(index)
    return intersection


def get_start_datetimes(
        datetimes: pd.DatetimeIndex,
        total_seq_len: int,
        max_gap: pd.Timedelta = FIVE_MINUTES) -> pd.DatetimeIndex:
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

    return pd.DatetimeIndex(np.concatenate(start_dt_index))


def get_t0_datetimes(
        datetimes: pd.DatetimeIndex,
        total_seq_len: int,
        history_len: int,
        max_gap: pd.Timedelta = FIVE_MINUTES) -> pd.DatetimeIndex:
    start_datetimes = get_start_datetimes(
        datetimes=datetimes, total_seq_len=total_seq_len, max_gap=max_gap)
    history_dur = timesteps_to_duration(history_len)
    t0_datetimes = start_datetimes + history_dur
    return t0_datetimes


def timesteps_to_duration(n_timesteps: int) -> pd.Timedelta:
    assert n_timesteps >= 0
    return pd.Timedelta(n_timesteps * 5, unit='minutes')


def datetime_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    features = {}
    features['hour_of_day'] = index.hour + (index.minute / 60)
    features['day_of_year'] = index.day_of_year
    return pd.DataFrame(features, index=index).astype(np.float32)


def datetime_features_in_example(index: pd.DatetimeIndex) -> Example:
    dt_features = datetime_features(index)
    dt_features['hour_of_day'] /= 24
    dt_features['day_of_year'] /= 365
    dt_features = utils.sin_and_cos(dt_features)
    example = Example()
    for col_name, series in dt_features.iteritems():
        example[col_name] = series
    return example
