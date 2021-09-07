import pytest
from nowcasting_dataset import time as nd_time
import pandas as pd
import numpy as np


def test_select_daylight_datetimes():
    datetimes = pd.date_range("2020-01-01 00:00", "2020-01-02 00:00", freq="H")
    locations = [(0, 0), (20_000, 20_000)]
    daylight_datetimes = nd_time.select_daylight_datetimes(
        datetimes=datetimes, locations=locations)
    correct_daylight_datetimes = pd.date_range(
        "2020-01-01 09:00", "2020-01-01 16:00", freq="H")
    np.testing.assert_array_equal(
        daylight_datetimes, correct_daylight_datetimes)


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


@pytest.mark.parametrize(
    "total_seq_len",
    [2, 3, 12]
)
def test_get_start_datetimes_1(total_seq_len):
    dt_index1 = pd.date_range('2010-01-01', '2010-01-02', freq='5 min')
    start_datetimes = nd_time.get_start_datetimes(
        dt_index1, total_seq_len=total_seq_len)
    np.testing.assert_array_equal(start_datetimes, dt_index1[:1-total_seq_len])


@pytest.mark.parametrize(
    "total_seq_len",
    [2, 3, 12]
)
def test_get_start_datetimes_2(total_seq_len):
    dt_index1 = pd.date_range('2010-01-01', '2010-01-02', freq='5 min')
    dt_index2 = pd.date_range('2010-02-01', '2010-02-02', freq='5 min')
    dt_index = dt_index1.union(dt_index2)
    start_datetimes = nd_time.get_start_datetimes(
        dt_index, total_seq_len=total_seq_len)
    correct_start_datetimes = dt_index1[:1-total_seq_len].union(
        dt_index2[:1-total_seq_len])
    np.testing.assert_array_equal(start_datetimes, correct_start_datetimes)


def test_timesteps_to_duration():
    assert nd_time.timesteps_to_duration(0) == pd.Timedelta(0)
    assert nd_time.timesteps_to_duration(1) == pd.Timedelta('5T')
    assert nd_time.timesteps_to_duration(12) == pd.Timedelta('1H')


def test_datetime_features_in_example():
    index = pd.date_range('2020-01-01', '2020-01-06 23:00', freq='h')
    example = nd_time.datetime_features_in_example(index)
    assert len(example['hour_of_day_sin']) == len(index)
    for col_name in ['hour_of_day_sin', 'hour_of_day_cos']:
        assert col_name in example
        np.testing.assert_array_almost_equal(
            example[col_name],
            np.tile(example[col_name][:24], reps=6))

    assert 'day_of_year_sin' in example
    assert 'day_of_year_cos' in example
