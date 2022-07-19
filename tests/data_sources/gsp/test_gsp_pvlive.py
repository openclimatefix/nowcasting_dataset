""" Test for PV live data """
from datetime import datetime

import pandas as pd
import pytz

from nowcasting_dataset.data_sources.gsp.pvlive import (
    get_installed_capacity,
    load_pv_gsp_raw_data_from_pvlive,
)


def test_load_gsp_raw_data_from_pvlive_one_gsp_one_day():
    """
    Test that one gsp system data can be loaded, just for one day
    """

    start = datetime(2022, 7, 17, tzinfo=pytz.utc)
    end = datetime(2022, 7, 18, tzinfo=pytz.utc)

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
    start = datetime(2022, 7, 17, tzinfo=pytz.utc)
    end = datetime(2022, 7, 18, tzinfo=pytz.utc)

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(
        start=start, end=end, number_of_gsp=1, normalize_data=False
    )
    assert gsp_pv_df["generation_mw"].max() > 1

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(
        start=start, end=end, number_of_gsp=1, normalize_data=True
    )
    assert gsp_pv_df["generation_mw"].max() <= 1


def test_load_gsp_raw_data_from_pvlive_one_gsp():
    """
    Test that one gsp system data can be loaded
    """

    start = datetime(2022, 1, 1, tzinfo=pytz.utc)
    end = datetime(2022, 1, 31, tzinfo=pytz.utc)

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end, number_of_gsp=1)

    assert isinstance(gsp_pv_df, pd.DataFrame)
    print(gsp_pv_df)
    assert len(gsp_pv_df) == (48 * 30)
    # 30 days in january,
    assert "datetime_gmt" in gsp_pv_df.columns
    assert "generation_mw" in gsp_pv_df.columns


def test_load_gsp_raw_data_from_pvlive_many_gsp():
    """
    Test that one gsp system data can be loaded
    """

    start = datetime(2022, 7, 17, tzinfo=pytz.utc)
    end = datetime(2022, 7, 18, tzinfo=pytz.utc)

    gsp_pv_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end, number_of_gsp=3)

    assert isinstance(gsp_pv_df, pd.DataFrame)
    assert len(gsp_pv_df) == (48 + 1) * 3
    assert "datetime_gmt" in gsp_pv_df.columns
    assert "generation_mw" in gsp_pv_df.columns


def test_get_installed_capacity():
    """
    Test thhat we can get installed capacity
    """

    installed_capacity = get_installed_capacity(maximum_number_of_gsp=3)

    assert len(installed_capacity) == 3
    assert "installedcapacity_mwp" == installed_capacity.name
    assert installed_capacity.iloc[0] == 177.0772
