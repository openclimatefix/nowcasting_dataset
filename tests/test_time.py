from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from nowcasting_dataset import time as nd_time
from nowcasting_dataset.time import THIRTY_MINUTES, FIVE_MINUTES


def test_select_daylight_datetimes():
    datetimes = pd.date_range("2020-01-01 00:00", "2020-01-02 00:00", freq="H")
    locations = [(0, 0), (20_000, 20_000)]
    daylight_datetimes = nd_time.select_daylight_datetimes(datetimes=datetimes, locations=locations)
    correct_daylight_datetimes = pd.date_range("2020-01-01 09:00", "2020-01-01 16:00", freq="H")
    np.testing.assert_array_equal(daylight_datetimes, correct_daylight_datetimes)


def test_intersection_of_datetimeindexes():
    # Test with just one
    index = pd.date_range("2010-01-01", "2010-01-02", freq="H")
    intersection = nd_time.intersection_of_datetimeindexes([index])
    np.testing.assert_array_equal(index, intersection)

    # Test with two identical:
    intersection = nd_time.intersection_of_datetimeindexes([index, index])
    np.testing.assert_array_equal(index, intersection)

    # Test with three with no intersection:
    index2 = pd.date_range("2020-01-01", "2010-01-02", freq="H")
    intersection = nd_time.intersection_of_datetimeindexes([index, index2])
    assert len(intersection) == 0

    # Test with three, with some intersection:
    index3 = pd.date_range("2010-01-01 06:00", "2010-01-02 06:00", freq="H")
    index4 = pd.date_range("2010-01-01 12:00", "2010-01-02 12:00", freq="H")
    intersection = nd_time.intersection_of_datetimeindexes([index, index3, index4])
    np.testing.assert_array_equal(
        intersection, pd.date_range("2010-01-01 12:00", "2010-01-02", freq="H")
    )


# TODO: Delete this test.
# TODO tracked on https://github.com/openclimatefix/nowcasting_dataset/issues/223
@pytest.mark.parametrize("total_seq_length", [2, 3, 12])
def test_get_start_datetimes_1(total_seq_length):
    dt_index1 = pd.date_range("2010-01-01", "2010-01-02", freq="5 min")
    start_datetimes = nd_time.get_start_datetimes(dt_index1, total_seq_length=total_seq_length)
    np.testing.assert_array_equal(start_datetimes, dt_index1[: 1 - total_seq_length])


# TODO: Delete this test.
# TODO tracked on https://github.com/openclimatefix/nowcasting_dataset/issues/223
@pytest.mark.parametrize("total_seq_length", [2, 3, 12])
def test_get_start_datetimes_2(total_seq_length):
    dt_index1 = pd.date_range("2010-01-01", "2010-01-02", freq="5 min")
    dt_index2 = pd.date_range("2010-02-01", "2010-02-02", freq="5 min")
    dt_index = dt_index1.union(dt_index2)
    start_datetimes = nd_time.get_start_datetimes(dt_index, total_seq_length=total_seq_length)
    correct_start_datetimes = dt_index1[: 1 - total_seq_length].union(
        dt_index2[: 1 - total_seq_length]
    )
    np.testing.assert_array_equal(start_datetimes, correct_start_datetimes)


@pytest.mark.parametrize("min_seq_length", [2, 3, 12])
def test_get_contiguous_time_periods_1_with_1_chunk(min_seq_length):
    freq = pd.Timedelta(5, unit="minutes")
    dt_index = pd.date_range("2010-01-01", "2010-01-02", freq=freq)
    periods: pd.DataFrame = nd_time.get_contiguous_time_periods(
        dt_index, min_seq_length=min_seq_length, max_gap_duration=freq
    )
    correct_periods = pd.DataFrame([{"start_dt": dt_index[0], "end_dt": dt_index[-1]}])
    pd.testing.assert_frame_equal(periods, correct_periods)


@pytest.mark.parametrize("min_seq_length", [2, 3, 12])
def test_get_contiguous_time_periods_2_with_2_chunks(min_seq_length):
    freq = pd.Timedelta(5, unit="minutes")
    dt_index1 = pd.date_range("2010-01-01", "2010-01-02", freq=freq)
    dt_index2 = pd.date_range("2010-02-01", "2010-02-02", freq=freq)
    dt_index = dt_index1.union(dt_index2)
    periods: pd.DataFrame = nd_time.get_contiguous_time_periods(
        dt_index, min_seq_length=min_seq_length, max_gap_duration=freq
    )
    correct_periods = pd.DataFrame(
        [
            {"start_dt": dt_index1[0], "end_dt": dt_index1[-1]},
            {"start_dt": dt_index2[0], "end_dt": dt_index2[-1]},
        ]
    )
    pd.testing.assert_frame_equal(periods, correct_periods)


def test_datetime_features_in_example():
    index = pd.date_range("2020-01-01", "2020-01-06 23:00", freq="h")
    example = nd_time.datetime_features_in_example(index)
    assert len(example.hour_of_day_sin) == len(index)
    for col_name in ["hour_of_day_sin", "hour_of_day_cos"]:
        np.testing.assert_array_almost_equal(
            getattr(example, col_name),
            np.tile(getattr(example, col_name)[:24], reps=6),
        )


@pytest.mark.parametrize("history_length", [2, 3, 12])
@pytest.mark.parametrize("forecast_length", [2, 3, 12])
def test_get_t0_datetimes(history_length, forecast_length):
    index = pd.date_range("2020-01-01", "2020-01-06 23:00", freq="30T")
    total_seq_length = history_length + forecast_length + 1
    sample_period_duration = THIRTY_MINUTES
    history_duration = sample_period_duration * history_length

    t0_datetimes = nd_time.get_t0_datetimes(
        datetimes=index,
        total_seq_length=total_seq_length,
        history_duration=history_duration,
        max_gap=THIRTY_MINUTES,
    )

    assert len(t0_datetimes) == len(index) - history_length - forecast_length
    assert t0_datetimes[0] == index[0] + timedelta(minutes=30 * history_length)
    assert t0_datetimes[-1] == index[-1] - timedelta(minutes=30 * forecast_length)


def test_get_t0_datetimes_night():
    history_length = 6
    forecast_length = 12
    sample_period_duration = FIVE_MINUTES
    index = pd.date_range("2020-06-15", "2020-06-15 22:15", freq=sample_period_duration)
    total_seq_length = history_length + forecast_length + 1
    history_duration = history_length * sample_period_duration

    t0_datetimes = nd_time.get_t0_datetimes(
        datetimes=index,
        total_seq_length=total_seq_length,
        history_duration=history_duration,
        max_gap=sample_period_duration,
    )

    assert len(t0_datetimes) == len(index) - history_length - forecast_length
    assert t0_datetimes[0] == index[0] + timedelta(minutes=5 * history_length)
    assert t0_datetimes[-1] == index[-1] - timedelta(minutes=5 * forecast_length)


def test_intersection_of_2_dataframes_of_periods():
    dt = pd.Timestamp("2020-01-01 00:00")
    a = []
    b = []
    c = []  # The correct intersection

    # a: |----|       |--|
    # b:  |--|  and |------|
    # c:  |--|        |--|
    a.append({"start_dt": dt, "end_dt": dt.replace(hour=3)})
    b.append({"start_dt": dt.replace(hour=1), "end_dt": dt.replace(hour=2)})
    c.append({"start_dt": dt.replace(hour=1), "end_dt": dt.replace(hour=2)})

    # a: |---|         |---|
    # b:   |---| and |---|
    # c:   |-|         |-|
    a.append({"start_dt": dt.replace(hour=4), "end_dt": dt.replace(hour=6)})
    b.append({"start_dt": dt.replace(hour=5), "end_dt": dt.replace(hour=7)})
    c.append({"start_dt": dt.replace(hour=5), "end_dt": dt.replace(hour=6)})

    # Test identical periods:
    # a: |---|
    # b: |---|
    # c: |---|
    a.append({"start_dt": dt.replace(hour=8), "end_dt": dt.replace(hour=10)})
    b.append({"start_dt": dt.replace(hour=8), "end_dt": dt.replace(hour=10)})
    c.append({"start_dt": dt.replace(hour=8), "end_dt": dt.replace(hour=10)})

    # Test non-overlapping but adjacent periods:
    # a: |---|
    # b:     |---|
    # c:
    a.append({"start_dt": dt.replace(hour=12), "end_dt": dt.replace(hour=13)})
    b.append({"start_dt": dt.replace(hour=13), "end_dt": dt.replace(hour=14)})

    # Test non-overlapping periods:
    # a: |---|
    # b:       |---|
    # c:
    a.append({"start_dt": dt.replace(hour=15), "end_dt": dt.replace(hour=16)})
    b.append({"start_dt": dt.replace(hour=17), "end_dt": dt.replace(hour=18)})

    # Convert these lists to DataFrames:
    a = pd.DataFrame(a)
    b = pd.DataFrame(b)
    c = pd.DataFrame(c)

    # Test intersection(a, b)
    intersection = nd_time.intersection_of_2_dataframes_of_periods(a, b)
    pd.testing.assert_frame_equal(intersection, c)

    # Test intersection(b, a)
    intersection = nd_time.intersection_of_2_dataframes_of_periods(b, a)
    pd.testing.assert_frame_equal(intersection, c)

    # Test with empty DataFrames
    empty_df = pd.DataFrame(columns=["start_dt", "end_dt"])
    test_cases = [(a, empty_df), (empty_df, b), (empty_df, empty_df)]
    for test_case in test_cases:
        pd.testing.assert_frame_equal(
            nd_time.intersection_of_2_dataframes_of_periods(*test_case), empty_df
        )
