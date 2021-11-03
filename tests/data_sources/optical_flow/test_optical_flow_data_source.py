"""Test OpticalFlowDataSource."""
import numpy as np
import pandas as pd
import pytest


def test_satellite_data_source_init(sat_data_source):  # noqa: D103
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
        (0, 0, -128_000, 126_000, 128_000, -126_000),
        (10, 0, -126_000, 128_000, 128_000, -126_000),
        (30, 0, -126_000, 128_000, 128_000, -126_000),
        (1000, 0, -126_000, 128_000, 128_000, -126_000),
        (0, 1000, -128_000, 126_000, 128_000, -126_000),
        (1000, 1000, -126_000, 128_000, 128_000, -126_000),
        (2000, 2000, -126_000, 128_000, 130_000, -124_000),
        (2000, 1000, -126_000, 128_000, 128_000, -126_000),
        (2001, 2001, -124_000, 130_000, 130_000, -124_000),
    ],
)
def test_get_example(sat_data_source, x, y, left, right, top, bottom):  # noqa: D103
    sat_data_source.open()
    t0_dt = pd.Timestamp("2019-01-01T13:00")
    sat_data = sat_data_source.get_example(t0_dt=t0_dt, x_meters_center=x, y_meters_center=y)

    assert left == sat_data.x.values[0]
    assert right == sat_data.x.values[-1]
    # sat_data.y is top-to-bottom.
    assert top == sat_data.y.values[0]
    assert bottom == sat_data.y.values[-1]
    assert len(sat_data.x) == pytest.IMAGE_SIZE_PIXELS
    assert len(sat_data.y) == pytest.IMAGE_SIZE_PIXELS


def test_geospatial_border(sat_data_source):  # noqa: D103
    border = sat_data_source.geospatial_border()
    correct_border = [(-110000, 1094000), (-110000, -58000), (730000, 1094000), (730000, -58000)]
    np.testing.assert_array_equal(border, correct_border)
