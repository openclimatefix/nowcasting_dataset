from nowcasting_dataset.data_sources.gsp.pvlive import load_pv_gsp_raw_data_from_pvlive
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso, get_gsp_shape_from_eso
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
import pandas as pd
import geopandas as gpd
from datetime import datetime
import pytz
import nowcasting_dataset
import os


def test_gsp_pv_data_source_init():
    local_path = os.path.dirname(nowcasting_dataset.__file__) + '/..'

    gsp = GSPDataSource(filename=f"{local_path}/tests/data/gsp/test.zarr",
                          start_dt=datetime(2019, 1, 1),
                          end_dt=datetime(2019, 1, 2),
                          history_minutes=30,
                          forecast_minutes=60,
                          convert_to_numpy=True,
                          image_size_pixels=64,
                          meters_per_pixel=2000)


def test_gsp_pv_data_source_get_locations_for_batch():
    local_path = os.path.dirname(nowcasting_dataset.__file__) + '/..'

    gsp = GSPDataSource(filename=f"{local_path}/tests/data/gsp/test.zarr",
                          start_dt=datetime(2019, 1, 1),
                          end_dt=datetime(2019, 1, 2),
                          history_minutes=30,
                          forecast_minutes=60,
                          convert_to_numpy=True,
                          image_size_pixels=64,
                          meters_per_pixel=2000)

    locations_x, locations_y = gsp.get_locations_for_batch(t0_datetimes=gsp.gsp_power.index[0:10])

    assert len(locations_x) == len(locations_y)


def test_gsp_pv_data_source_get_example():
    local_path = os.path.dirname(nowcasting_dataset.__file__) + '/..'

    gsp = GSPDataSource(filename=f"{local_path}/tests/data/gsp/test.zarr",
                          start_dt=datetime(2019, 1, 1),
                          end_dt=datetime(2019, 1, 2),
                          history_minutes=30,
                          forecast_minutes=60,
                          convert_to_numpy=True,
                          image_size_pixels=64,
                          meters_per_pixel=2000)

    x_locations, y_locations = gsp.get_locations_for_batch(t0_datetimes=gsp.gsp_power.index[0:10])
    l = gsp.get_example(t0_dt=gsp.gsp_power.index[0], x_meters_center=x_locations[0], y_meters_center=y_locations[0])

    assert len(l['gsp_system_id']) == len(l['gsp_yield'][0])
    assert len(l['gsp_system_x_coords']) == len(l['gsp_system_y_coords'])
    assert len(l['gsp_system_x_coords']) > 0


def test_gsp_pv_data_source_get_batch():
    local_path = os.path.dirname(nowcasting_dataset.__file__) + '/..'

    gsp = GSPDataSource(filename=f"{local_path}/tests/data/gsp/test.zarr",
                          start_dt=datetime(2019, 1, 1),
                          end_dt=datetime(2019, 1, 2),
                          history_minutes=30,
                          forecast_minutes=60,
                          minute_delta=30,
                          convert_to_numpy=True,
                          image_size_pixels=64,
                          meters_per_pixel=2000)

    batch_size = 10

    x_locations, y_locations = gsp.get_locations_for_batch(t0_datetimes=gsp.gsp_power.index[0:batch_size])

    batch = gsp.get_batch(t0_datetimes=gsp.gsp_power.index[batch_size:2*batch_size],
                      x_locations=x_locations[0:batch_size],
                      y_locations=y_locations[0:batch_size])

    assert len(batch) == batch_size
    assert len(batch[0]['gsp_yield']) == 4
    assert len(batch[0]['gsp_system_id']) == len(batch[0]['gsp_system_x_coords'])
    assert len(batch[1]['gsp_system_x_coords']) == len(batch[1]['gsp_system_y_coords'])
    assert len(batch[2]['gsp_system_x_coords']) > 0



def test_get_gsp_metadata_from_eso():
    """
    Test to get the gsp metadata from eso. This should take ~1 second.
    @return:
    """
    metadata = get_gsp_metadata_from_eso()

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

    gsp_shapes = get_gsp_shape_from_eso()

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
