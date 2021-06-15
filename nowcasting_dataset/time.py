import pandas as pd
import numpy as np
from typing import Iterable, Tuple, List, NamedTuple
import pvlib
from nowcasting_dataset import geospatial
import warnings


FIVE_MINUTES = pd.Timedelta('5 minutes')


def select_daylight_timestamps(
        dt_index: pd.DatetimeIndex,
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
            clearsky = location.get_clearsky(dt_index)
        ghi = clearsky['ghi']
        ghi_for_all_locations.append(ghi)

    ghi_for_all_locations = pd.concat(ghi_for_all_locations, axis='columns')
    max_ghi = ghi_for_all_locations.max(axis='columns')
    mask = max_ghi > ghi_threshold
    return dt_index[mask]


def intersection_of_datetimeindexes(
        indexes: List[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    assert len(indexes) > 0
    intersection = indexes[0]
    for index in indexes[1:]:
        intersection = intersection.intersection(index)
    return intersection


class Segment(NamedTuple):
    """Represents the start and end datetimes of a segment of contiguous samples

    The Segment covers the range [start, end].
    """
    start: pd.Timestamp
    end: pd.Timestamp

    def duration(self) -> pd.Timedelta:
        return self.end - self.start


def get_contiguous_segments(
        dt_index: pd.DatetimeIndex,
        min_timesteps: int = 12,
        max_gap: pd.Timedelta = FIVE_MINUTES) -> List[Segment]:
    """Chunk datetime index into contiguous segments, each at least
    min_timesteps long.

    max_gap defines the threshold for what constitutes a 'gap' between
    contiguous segments.

    Throw away any timesteps in a sequence shorter than min_timesteps long.
    """
    assert len(dt_index) > 0
    assert min_timesteps > 1

    gap_mask = np.diff(dt_index) > max_gap
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately
    # *before* the gap.  e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05
    # then gap_indicies will be [1].  So we add 1 to gap_indices to get
    # segment_boundaries, an index into dt_index which identifies the _start_
    # of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(dt_index)]))

    segments: List[Segment] = []
    start_i = 0
    for next_start_i in segment_boundaries:
        n_timesteps = next_start_i - start_i
        if n_timesteps >= min_timesteps:
            end_i = next_start_i - 1
            segment = Segment(start=dt_index[start_i], end=dt_index[end_i])
            segments.append(segment)
        start_i = next_start_i

    return segments
