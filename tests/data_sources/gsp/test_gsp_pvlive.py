from datetime import datetime

import pandas as pd
import pytz

from nowcasting_dataset.data_sources.gsp.pvlive import (
    load_pv_gsp_raw_data_from_pvlive,
    get_installed_capacity,
)


def test_load_gsp_raw_data_from_pvlive_one_gsp_one_day():
    """
    Test that one gsp system data can be loaded, just for one day
    """

    start = datetime(2019, 1, 1, tzinfo=pytz.utc)
    end = datetime(2019, 1, 2, tzinfo=pytz.utc)

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end, number_of_gsp=1)

    assert isinstance(gsp_pv_df, pd.DataFrame)
    assert len(gsp_pv_df) == (48 + 1)
    assert "datetime_gmt" in gsp_pv_df.columns
    assert "generation_mw" in gsp_pv_df.columns


def test_load_gsp_raw_data_from_pvlive_one_gsp_one_day_not_normalised():
    """
    Test that one gsp system data can be loaded, just for one day, and is normalized correctly
    """

    # pick a summer day
    start = datetime(2019, 6, 21, tzinfo=pytz.utc)
    end = datetime(2019, 6, 22, tzinfo=pytz.utc)

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(
        start=start, end=end, number_of_gsp=1, normalize_data=False
    )
    assert gsp_pv_df["generation_mw"].max() > 1

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(
        start=start, end=end, number_of_gsp=1, normalize_data=True
    )
    assert gsp_pv_df["generation_mw"].max() <= 1


def test_load_gsp_raw_data_from_pvlive_one_gsp():
    """a
    Test that one gsp system data can be loaded
    """

    start = datetime(2019, 1, 1, tzinfo=pytz.utc)
    end = datetime(2019, 3, 1, tzinfo=pytz.utc)

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end, number_of_gsp=1)

    assert isinstance(gsp_pv_df, pd.DataFrame)
    assert len(gsp_pv_df) == (48 * 59 + 1)
    # 30 days in january, 29 days in february, plus one for the first timestamp in march
    assert "datetime_gmt" in gsp_pv_df.columns
    assert "generation_mw" in gsp_pv_df.columns


def test_load_gsp_raw_data_from_pvlive_many_gsp():
    """
    Test that one gsp system data can be loaded
    """

    start = datetime(2019, 1, 1, tzinfo=pytz.utc)
    end = datetime(2019, 1, 2, tzinfo=pytz.utc)

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end, number_of_gsp=10)

    assert isinstance(gsp_pv_df, pd.DataFrame)
    assert len(gsp_pv_df) == (48 + 1) * 10
    assert "datetime_gmt" in gsp_pv_df.columns
    assert "generation_mw" in gsp_pv_df.columns


def test_get_installed_capacity():

    installed_capacity = get_installed_capacity(maximum_number_of_gsp=10)

    assert len(installed_capacity) == 10
    assert "installedcapacity_mwp" == installed_capacity.name
    assert installed_capacity.iloc[0] == 342.02623
    assert installed_capacity.iloc[9] == 308.00432
