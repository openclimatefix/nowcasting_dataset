""" Tests for GSPDataSource """
import os
from datetime import datetime

import pandas as pd

import nowcasting_dataset
from nowcasting_dataset.data_sources.gsp.gsp_data_source import (
    GSPDataSource,
    drop_gsp_north_of_boundary,
)
from nowcasting_dataset.data_sources.metadata.metadata_model import Metadata
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
        image_size_pixels_height=64,
        image_size_pixels_width=64,
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
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
    )

    locations = gsp.get_locations(t0_datetimes_utc=gsp.gsp_power.index[0:10])

    # This makes sure it is not in lat/lon.
    # Note that OSGB could be <= than 90, but that would mean a location in the middle of the sea,
    # which is impossible for GSP data
    assert locations[0].x_center_osgb > 90
    assert locations[0].y_center_osgb > 90

    lat, lon = osgb_to_lat_lon(locations[0].x_center_osgb, locations[0].y_center_osgb)

    assert 0 < lat < 90  # this makes sure it is in lat/lon
    assert -90 < lon < 90  # this makes sure it is in lat/lon


def test_gsp_pv_data_source_get_all_locations():
    """Test GSP example"""
    local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
    )

    N_gsps = len(gsp.metadata)

    t0_datetimes_utc = gsp.gsp_power.index[0:10]
    x_locations = gsp.metadata.location_x

    locations = gsp.get_all_locations(t0_datetimes_utc=t0_datetimes_utc)
    metadata = Metadata(space_time_locations=locations, batch_size=32)

    # check first few are the same datetime
    assert (metadata.x_centers_osgb[0:N_gsps] == x_locations.values).all()
    assert (pd.DatetimeIndex(metadata.t0_datetimes_utc[0:N_gsps]) == t0_datetimes_utc[0]).all()

    # check second set of datetimes
    assert (metadata.x_centers_osgb[N_gsps : 2 * N_gsps] == x_locations.values).all()
    assert (
        pd.DatetimeIndex(metadata.t0_datetimes_utc[N_gsps : 2 * N_gsps]) == t0_datetimes_utc[1]
    ).all()

    # check all datetimes
    t0_datetimes_utc_all_gsps_overlap = pd.DatetimeIndex(metadata.t0_datetimes_utc).union(
        t0_datetimes_utc
    )
    assert len(t0_datetimes_utc_all_gsps_overlap) == len(metadata.t0_datetimes_utc)


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
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
    )

    locations = gsp.get_locations(t0_datetimes_utc=gsp.gsp_power.index[0:10])
    example = gsp.get_example(location=locations[0])

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
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
    )

    batch_size = 10

    locations = gsp.get_locations(t0_datetimes_utc=gsp.gsp_power.index[batch_size : 2 * batch_size])

    batch = gsp.get_batch(locations=locations[:batch_size])

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
        image_size_pixels_height=64,
        image_size_pixels_width=64,
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
