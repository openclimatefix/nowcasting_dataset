from nowcasting_dataset.data_sources.gsp.pvlive import load_pv_gsp_raw_data_from_pvlive
from nowcasting_dataset.data_sources.gsp.eso import get_pv_gsp_metadata_from_eso, get_pv_gsp_shape_from_eso
import pandas as pd
import geopandas as gpd
from datetime import datetime
import pytz


def test_get_gsp_metadata_from_eso():
    """
    Test to get the gsp metadata from eso. This should take ~1 second.
    @return:
    """
    metadata = get_pv_gsp_metadata_from_eso()

    assert isinstance(metadata, pd.DataFrame)
    assert len(metadata) > 100
    assert "gnode_name" in metadata.columns
    assert "gnode_lat" in metadata.columns
    assert "gnode_lon" in metadata.columns


def test_get_pv_gsp_shape():
    """
    Test to get the gsp metadata from eso. This should take ~1 second.
    @return:
    """

    gsp_shapes = get_pv_gsp_shape_from_eso()

    assert isinstance(gsp_shapes, gpd.GeoDataFrame)
    assert "RegionID" in gsp_shapes.columns
    assert "RegionName" in gsp_shapes.columns
    assert "geometry" in gsp_shapes.columns


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


def test_load_gsp_raw_data_from_pvlive_one_gsp():
    """
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
