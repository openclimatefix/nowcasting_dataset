from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from nowcasting_dataset import time as nd_time
from nowcasting_dataset.time import FIVE_MINUTES, THIRTY_MINUTES


def test_select_daylight_datetimes():
    datetimes = pd.date_range("2020-01-01 00:00", "2020-01-02 00:00", freq="H")
    locations = [(0, 0), (20_000, 20_000)]
    daylight_datetimes = nd_time.select_daylight_datetimes(datetimes=datetimes, locations=locations)
    correct_daylight_datetimes = pd.date_range("2020-01-01 09:00", "2020-01-01 16:00", freq="H")
    np.testing.assert_array_equal(daylight_datetimes, correct_daylight_datetimes)


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


def test_intersection_of_multiple_dataframes_of_periods():
    dt = pd.Timestamp("2020-01-01 00:00")

    # a: |-----|
    # b:  |---|
    # c:   |-----|
    # i:   |-|  # The correct intersection of a, b, c.
    a = pd.DataFrame([{"start_dt": dt, "end_dt": dt.replace(hour=4)}])
    b = pd.DataFrame([{"start_dt": dt.replace(hour=1), "end_dt": dt.replace(hour=3)}])
    c = pd.DataFrame([{"start_dt": dt.replace(hour=2), "end_dt": dt.replace(hour=5)}])

    i = nd_time.intersection_of_multiple_dataframes_of_periods([a, b, c])
    correct = pd.DataFrame([{"start_dt": dt.replace(hour=2), "end_dt": dt.replace(hour=3)}])
    pd.testing.assert_frame_equal(i, correct)


def test_time_periods_to_datetime_index():
    dt = pd.Timestamp("2020-01-01 00:00")
    time_periods = pd.DataFrame(
        [
            {"start_dt": dt.replace(hour=1), "end_dt": dt.replace(hour=3)},
            {"start_dt": dt.replace(hour=5), "end_dt": dt.replace(hour=10)},
        ]
    )

    FREQ = "5T"
    dt_index = nd_time.time_periods_to_datetime_index(time_periods, freq=FREQ)

    correct_dt_index = pd.date_range("2020-01-01 01:00", "2020-01-01 03:00", freq=FREQ).union(
        pd.date_range("2020-01-01 05:00", "2020-01-01 10:00", freq=FREQ)
    )
    pd.testing.assert_index_equal(dt_index, correct_dt_index)
