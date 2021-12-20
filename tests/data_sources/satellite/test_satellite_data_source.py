"""Test SatelliteDataSource."""
import numpy as np
import pandas as pd
import pytest


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


@pytest.mark.parametrize(
    "x, y, left, right, top, bottom",
    [
        (0, 0, -221121.39078455, 208335.1546766134, 384697.57834117045, -355285.4325789859),
        (10, 0, -221121.39078455, 208335.1546766134, 384697.57834117045, -355285.4325789859),
        (30, 0, -221121.39078455, 208335.1546766134, 384697.57834117045, -355285.4325789859),
        (1000, 0, -221121.39078455, 208335.1546766134, 384697.57834117045, -355285.4325789859),
        (0, 1000, -221121.39078455, 208335.1546766134, 384697.57834117045, -355285.4325789859),
        (1000, 1000, -221121.39078455, 208335.1546766134, 384697.57834117045, -355285.4325789859),
        (
            2000,
            2000,
            -217660.91342640377,
            211646.9832259717,
            384697.57834117045,
            -355285.4325789859,
        ),
        (
            2000,
            1000,
            -217660.91342640377,
            211646.9832259717,
            384697.57834117045,
            -355285.4325789859,
        ),
        (
            10001,
            10001,
            -210744.23604969133,
            218267.83605829155,
            391200.6059295471,
            -349991.70123797783,
        ),
    ],
)
def test_get_example(sat_data_source, x, y, left, right, top, bottom):  # noqa: D103
    sat_data_source.open()
    t0_dt = pd.Timestamp("2020-04-01T13:00")
    sat_data = sat_data_source.get_example(t0_dt=t0_dt, x_meters_center=x, y_meters_center=y)
    assert np.isclose(left, sat_data.x_osgb.values[0])
    assert np.isclose(right, sat_data.x_osgb.values[-1])
    # sat_data.y is top-to-bottom.
    assert np.isclose(bottom, sat_data.y_osgb.values[0])
    assert np.isclose(top, sat_data.y_osgb.values[-1])
    assert len(sat_data.x_osgb) == pytest.IMAGE_SIZE_PIXELS
    assert len(sat_data.y_osgb) == pytest.IMAGE_SIZE_PIXELS


@pytest.mark.parametrize(
    "x, y, left, right, top, bottom",
    [
        (0, 0, -72420.55623992952, 70584.73313047731, 122797.71873342531, -122767.5525824396),
        (10, 0, -72420.55623992952, 70584.73313047731, 122797.71873342531, -122767.5525824396),
        (30, 0, -72420.55623992952, 70584.73313047731, 122797.71873342531, -122767.5525824396),
        (1000, 0, -72420.55623992952, 70584.73313047731, 122797.71873342531, -122767.5525824396),
        (0, 1000, -72420.55623992952, 70584.73313047731, 124799.65960250192, -120896.5028911124),
        (1000, 1000, -72420.55623992952, 70584.73313047731, 124799.65960250192, -120896.5028911124),
        (2000, 2000, -71286.2044391025, 71702.68621321098, 124799.65960250192, -120896.5028911124),
        (2000, 1000, -71286.2044391025, 71702.68621321098, 124799.65960250192, -120896.5028911124),
        (2001, 2001, -71286.2044391025, 71702.68621321098, 124799.65960250192, -120896.5028911124),
    ],
)
def test_hrv_get_example(hrv_sat_data_source, x, y, left, right, top, bottom):  # noqa: D103
    hrv_sat_data_source.open()
    t0_dt = pd.Timestamp("2020-04-01T13:00")
    sat_data = hrv_sat_data_source.get_example(t0_dt=t0_dt, x_meters_center=x, y_meters_center=y)
    assert np.isclose(left, sat_data.x_osgb.values[0])
    assert np.isclose(right, sat_data.x_osgb.values[-1])
    # sat_data.y is top-to-bottom.
    assert np.isclose(bottom, sat_data.y_osgb.values[0])
    assert np.isclose(top, sat_data.y_osgb.values[-1])
    assert len(sat_data.x_osgb) == pytest.IMAGE_SIZE_PIXELS
    assert len(sat_data.y_osgb) == pytest.IMAGE_SIZE_PIXELS


def test_hrv_geospatial_border(hrv_sat_data_source):  # noqa: D103
    border = hrv_sat_data_source.geospatial_border()
    correct_border = [
        [-612835.047242, -447550.008416],
        [-612835.047242, 1204307.57632],
        [1281085.856779, -447550.008416],
        [1281085.856779, 1204307.57632],
    ]
    np.testing.assert_array_almost_equal(border, correct_border)


def test_geospatial_border(sat_data_source):  # noqa: D103
    border = sat_data_source.geospatial_border()
    correct_border = [
        [-460044.797056, -220812.66369],
        [-460044.797056, 862710.688863],
        [1147550.338664, -220812.66369],
        [1147550.338664, 862710.688863],
    ]
    np.testing.assert_array_almost_equal(border, correct_border)
