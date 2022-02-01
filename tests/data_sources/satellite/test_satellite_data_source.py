"""Test SatelliteDataSource."""
import numpy as np
import pandas as pd
import pytest

from nowcasting_dataset.data_sources.metadata.metadata_model import Location
from nowcasting_dataset.data_sources.satellite.satellite_data_source import SatelliteDataSource


def test_satellite_data_source_init(sat_data_source):  # noqa: D103
    pass


def test_hrvsatellite_data_source_init(hrv_sat_data_source):  # noqa: D103
    pass


def test_open(sat_data_source):  # noqa: D103
    sat_data_source.open()
    assert sat_data_source.data is not None


def test_datetime_index(sat_data_source):  # noqa: D103
    datetimes = sat_data_source.datetime_index()
    assert isinstance(datetimes, pd.DatetimeIndex)
    assert len(datetimes) > 0
    assert len(np.unique(datetimes)) == len(datetimes)
    assert np.all(np.diff(datetimes.view(int)) > 0)


def _test_get_example(
    data_source,
    x_center_osgb,
    y_center_osgb,
    left_geostationary,
    right_geostationary,
    top_geostationary,
    bottom_geostationary,
):  # noqa: D103
    data_source.open()
    t0_dt = pd.Timestamp("2020-04-01T13:00")
    sat_data = data_source.get_example(
        Location(t0_datetime_utc=t0_dt, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb)
    )

    # sat_data.y is top-to-bottom.
    assert np.isclose(left_geostationary, sat_data.x_geostationary.values[0])
    assert np.isclose(right_geostationary, sat_data.x_geostationary.values[-1])
    assert np.isclose(top_geostationary, sat_data.y_geostationary.values[-1])
    assert np.isclose(bottom_geostationary, sat_data.y_geostationary.values[0])
    assert len(sat_data.x_geostationary) == pytest.IMAGE_SIZE_PIXELS
    assert len(sat_data.y_geostationary) == pytest.IMAGE_SIZE_PIXELS
    assert len(sat_data.x_geostationary.shape) == 1


@pytest.mark.parametrize(
    "x_center_osgb, y_center_osgb, left_geostationary, right_geostationary,"
    " top_geostationary, bottom_geostationary",
    [
        (0, 0, -1326178.250, -945127.000, 4698631.500, 4317580.000),
        (10, 0, -1326178.250, -945127.000, 4698631.500, 4317580.000),
        (30, 0, -1326178.250, -945127.000, 4698631.500, 4317580.000),
        (1000, 0, -1326178.250, -945127.000, 4698631.500, 4317580.000),
        (0, 1000, -1326178.250, -945127.000, 4698631.500, 4317580.000),
        (1000, 1000, -1326178.250, -945127.000, 4698631.500, 4317580.000),
        (2000, 2000, -1326178.250, -945127.000, 4698631.500, 4317580.000),
        (2000, 1000, -1326178.250, -945127.000, 4698631.500, 4317580.000),
        (10001, 10001, -1317177.000, -936125.812, 4704632.000, 4323581.000),
    ],
)
def test_get_example(
    sat_data_source,
    x_center_osgb,
    y_center_osgb,
    left_geostationary,
    right_geostationary,
    top_geostationary,
    bottom_geostationary,
):  # noqa: D103
    _test_get_example(
        data_source=sat_data_source,
        x_center_osgb=x_center_osgb,
        y_center_osgb=y_center_osgb,
        left_geostationary=left_geostationary,
        right_geostationary=right_geostationary,
        top_geostationary=top_geostationary,
        bottom_geostationary=bottom_geostationary,
    )


@pytest.mark.parametrize(
    "x_center_osgb, y_center_osgb, left_geostationary, right_geostationary,"
    " top_geostationary, bottom_geostationary",
    [
        (0, 0, -1198161.000, -1071143.875, 4573614.500, 4446597.500),
        (10, 0, -1198161.000, -1071143.875, 4573614.500, 4446597.500),
        (30, 0, -1198161.000, -1071143.875, 4573614.500, 4446597.500),
        (1000, 0, -1197160.875, -1070143.750, 4573614.500, 4446597.500),
        (0, 1000, -1198161.000, -1071143.875, 4574614.500, 4447597.500),
        (1000, 1000, -1197160.875, -1070143.750, 4574614.500, 4447597.500),
        (2000, 2000, -1196160.625, -1069143.625, 4574614.500, 4447597.500),
        (2000, 1000, -1196160.625, -1069143.625, 4574614.500, 4447597.500),
        (2001, 2001, -1196160.625, -1069143.625, 4574614.500, 4447597.500),
    ],
)
def test_hrv_get_example(
    hrv_sat_data_source,
    x_center_osgb,
    y_center_osgb,
    left_geostationary,
    right_geostationary,
    top_geostationary,
    bottom_geostationary,
):  # noqa: D103
    _test_get_example(
        data_source=hrv_sat_data_source,
        x_center_osgb=x_center_osgb,
        y_center_osgb=y_center_osgb,
        left_geostationary=left_geostationary,
        right_geostationary=right_geostationary,
        top_geostationary=top_geostationary,
        bottom_geostationary=bottom_geostationary,
    )


def test_hrv_geospatial_border(hrv_sat_data_source):  # noqa: D103
    border = hrv_sat_data_source.geospatial_border()
    correct_border = [
        [-636130.9, -339340.97],
        [-1123853.4, 1853731.9],
        [1265433.1, -457073.28],
        [998077.7, 1192428.8],
    ]
    np.testing.assert_array_almost_equal(border, correct_border, decimal=1)


def test_geospatial_border(sat_data_source):  # noqa: D103
    border = sat_data_source.geospatial_border()
    correct_border = [
        [-530300.1, -127087.12],
        [-807881.6, 1179602.2],
        [1097792.5, -247369.67],
        [920786.25, 833248.0],
    ]
    np.testing.assert_array_almost_equal(border, correct_border, decimal=1)


def test_wrong_sample_period(sat_filename):
    """Test that a error is raise when the time_resolution_minutes is not divisible by 5"""
    with pytest.raises(Exception):
        _ = SatelliteDataSource(
            image_size_pixels=pytest.IMAGE_SIZE_PIXELS,
            zarr_path=sat_filename,
            history_minutes=0,
            forecast_minutes=15,
            channels=("IR_016",),
            meters_per_pixel=6000,
            time_resolution_minutes=27,
        )
