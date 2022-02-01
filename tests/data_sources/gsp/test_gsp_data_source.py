""" Tests for GSPDataSource """
import os
from datetime import datetime

import pandas as pd

import nowcasting_dataset
from nowcasting_dataset.data_sources.gsp.gsp_data_source import (
    GSPDataSource,
    drop_gsp_north_of_boundary,
)
from nowcasting_dataset.geospatial import osgb_to_lat_lon


def test_gsp_pv_data_source_init():
    """Test GSP init"""
    local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."

    _ = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )


def test_gsp_pv_data_source_get_locations():
    """Test GSP locations"""
    local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    locations_x, locations_y, ids = gsp.get_locations(t0_datetimes_utc=gsp.gsp_power.index[0:10])

    assert len(locations_x) == len(locations_y)
    # This makes sure it is not in lat/lon.
    # Note that OSGB could be <= than 90, but that would mean a location in the middle of the sea,
    # which is impossible for GSP data
    assert locations_x[0] > 90
    assert locations_y[0] > 90

    lat, lon = osgb_to_lat_lon(locations_x, locations_y)

    assert 0 < lat[0] < 90  # this makes sure it is in lat/lon
    assert -90 < lon[0] < 90  # this makes sure it is in lat/lon


def test_gsp_pv_data_source_get_all_locations():
    """Test GSP example"""
    local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    N_gsps = len(gsp.metadata)

    t0_datetimes_utc = gsp.gsp_power.index[0:10]
    x_locations = gsp.metadata.location_x

    (
        t0_datetimes_utc_all_gsps,
        x_centers_osgb_all_gsps,
        y_centers_osgb_all_gsps,
        ids,
    ) = gsp.get_all_locations(t0_datetimes_utc=t0_datetimes_utc)

    assert len(t0_datetimes_utc_all_gsps) == len(x_centers_osgb_all_gsps)
    assert len(t0_datetimes_utc_all_gsps) == len(y_centers_osgb_all_gsps)
    assert len(t0_datetimes_utc_all_gsps) == len(x_locations) * len(t0_datetimes_utc)

    # check first few are the same datetime
    assert (x_centers_osgb_all_gsps[0:N_gsps] == x_locations.values).all()
    assert (t0_datetimes_utc_all_gsps[0:N_gsps] == t0_datetimes_utc[0]).all()

    # check second set of datetimes
    assert (x_centers_osgb_all_gsps[N_gsps : 2 * N_gsps] == x_locations.values).all()
    assert (t0_datetimes_utc_all_gsps[N_gsps : 2 * N_gsps] == t0_datetimes_utc[1]).all()

    # check all datetimes
    t0_datetimes_utc_all_gsps_overlap = t0_datetimes_utc_all_gsps.union(t0_datetimes_utc)
    assert len(t0_datetimes_utc_all_gsps_overlap) == len(t0_datetimes_utc_all_gsps)


def test_gsp_pv_data_source_get_example():
    """Test GSP example"""
    local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."

    start_dt = datetime(2020, 4, 1)
    end_dt = datetime(2020, 4, 1)

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    x_locations, y_locations, ids = gsp.get_locations(t0_datetimes_utc=gsp.gsp_power.index[0:10])
    example = gsp.get_example(
        t0_datetime_utc=gsp.gsp_power.index[0],
        x_center_osgb=x_locations[0],
        y_center_osgb=y_locations[0],
    )

    assert len(example.id) == len(example.power_mw[0])
    assert len(example.x_osgb) == len(example.y_osgb)
    assert len(example.x_osgb) > 0
    assert pd.Timestamp(example.time[0].values) <= end_dt
    assert pd.Timestamp(example.time[0].values) >= start_dt


def test_gsp_pv_data_source_get_batch():
    """Test GSP batch"""
    local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    batch_size = 10

    x_locations, y_locations, ids = gsp.get_locations(
        t0_datetimes_utc=gsp.gsp_power.index[0:batch_size]
    )

    batch = gsp.get_batch(
        t0_datetimes_utc=gsp.gsp_power.index[batch_size : 2 * batch_size],
        x_centers_osgb=x_locations[0:batch_size],
        y_centers_osgb=y_locations[0:batch_size],
    )

    assert len(batch.power_mw[0]) == 4
    assert len(batch.id[0]) == len(batch.x_osgb[0])
    assert len(batch.x_osgb[1]) == len(batch.y_osgb[1])
    assert len(batch.x_osgb[2]) > 0
    # assert T0_DT in batch[3].keys()


def test_drop_gsp_north_of_boundary(test_data_folder):
    """Test that dropping GSP north of a boundary works"""

    gsp = GSPDataSource(
        zarr_path=f"{test_data_folder}/gsp/test.zarr",
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
        northern_boundary_osgb=None,
    )

    # remove all gsp systems
    gsp_power, metadata = drop_gsp_north_of_boundary(
        gsp.gsp_power, gsp.metadata, northern_boundary_osgb=0
    )
    assert len(gsp_power.columns) == 0
    assert len(metadata) == 0

    # remove half the systems
    north_osgb_median = int(gsp.metadata.location_y.median())
    gsp_power, metadata = drop_gsp_north_of_boundary(
        gsp.gsp_power, gsp.metadata, northern_boundary_osgb=north_osgb_median
    )
    assert len(gsp_power.columns) == len(gsp.gsp_power.columns) / 2
    assert len(metadata) == len(gsp.metadata) / 2
