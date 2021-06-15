from nowcasting_dataset import time as nd_time
import pandas as pd
import numpy as np


def test_select_daylight_timestamps():
    dt_index = pd.date_range("2020-01-01 00:00", "2020-01-02 00:00", freq="H")
    locations = [(0, 0), (20_000, 20_000)]
    daylight_index = nd_time.select_daylight_timestamps(
        dt_index=dt_index, locations=locations)
    correct_daylight_index = pd.date_range(
        "2020-01-01 09:00", "2020-01-01 16:00", freq="H")
    np.testing.assert_array_equal(daylight_index, correct_daylight_index)


def test_intersection_of_datetimeindexes():
    # Test with just one
    index = pd.date_range('2010-01-01', '2010-01-02', freq='H')
    intersection = nd_time.intersection_of_datetimeindexes([index])
    np.testing.assert_array_equal(index, intersection)

    # Test with two identical:
    intersection = nd_time.intersection_of_datetimeindexes([index, index])
    np.testing.assert_array_equal(index, intersection)

    # Test with three with no intersection:
    index2 = pd.date_range('2020-01-01', '2010-01-02', freq='H')
    intersection = nd_time.intersection_of_datetimeindexes([index, index2])
    assert len(intersection) == 0

    # Test with three, with some intersection:
    index3 = pd.date_range('2010-01-01 06:00', '2010-01-02 06:00', freq='H')
    index4 = pd.date_range('2010-01-01 12:00', '2010-01-02 12:00', freq='H')
    intersection = nd_time.intersection_of_datetimeindexes(
        [index, index3, index4])
    np.testing.assert_array_equal(
        intersection,
        pd.date_range('2010-01-01 12:00', '2010-01-02', freq='H'))


def test_get_contiguous_segments():
    dt_index1 = pd.date_range('2010-01-01', '2010-01-02', freq='5 min')
    segments = nd_time.get_contiguous_segments(dt_index1)
    assert len(segments) == 1
    assert segments[0].start == dt_index1[0]
    assert segments[0].end == dt_index1[-1]

    dt_index2 = pd.date_range('2010-02-01', '2010-02-02', freq='5 min')
    dt_index_union = dt_index1.union(dt_index2)
    segments = nd_time.get_contiguous_segments(dt_index_union)
    assert len(segments) == 2
    assert segments[0].start == dt_index1[0]
    assert segments[0].end == dt_index1[-1]
    assert segments[1].start == dt_index2[0]
    assert segments[1].end == dt_index2[-1]
