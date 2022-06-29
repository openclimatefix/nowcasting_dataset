""" Test for geospatial functions """
import numpy as np
import pandas as pd

from nowcasting_dataset import geospatial


def _get_random_osgb_coords_for_tests() -> tuple[np.ndarray, np.ndarray]:
    x_osgb = np.random.randint(0, 100, 10)
    y_osgb = np.random.randint(0, 100, 10)
    return x_osgb, y_osgb


def test_osgb_to_lat_lon():
    """
    Test to check transform function is working correctly
    """

    try:
        osgb_coords = geospatial.osgb_to_lat_lon(x=0, y=0)
        np.testing.assert_allclose(osgb_coords, (49.76680681317516, -7.557207277153569))
    except Exception as e:  # noqa F841
        # Sometimes this test fails, then run the following code.
        # This forces a fresh 'grid' to be downloaded.
        geospatial.download_grids()

    finally:
        osgb_coords = geospatial.osgb_to_lat_lon(x=0, y=0)
        np.testing.assert_allclose(osgb_coords, (49.76680681317516, -7.557207277153569))


def test_calculate_azimuth_and_elevation_angle():
    """Test calculate_azimuth_and_elevation_angle"""
    datestamps = pd.date_range("2021-06-22 12:00:00", "2021-06-23", freq="5T", tz="UTC")

    s = geospatial.calculate_azimuth_and_elevation_angle(
        longitude=0, latitude=51, datestamps=datestamps.to_pydatetime()
    )

    assert len(s) == len(datestamps)
    assert "azimuth" in s.columns
    assert "elevation" in s.columns

    # midday sun at 12 oclock on mid summer, middle of the sky, and in london at around 62 degrees
    # https://diamondgeezer.blogspot.com/2017/12/solar-elevation.html
    assert 170 < s["azimuth"][0] < 190
    assert 60 < s["elevation"][0] < 65


def test_get_osgb_center_from_osgb():
    """Test get OSGB center"""
    x_osgb, y_osgb = _get_random_osgb_coords_for_tests()

    x, y = geospatial.get_osgb_center_from_list_of_x_and_y_osgb(x_osgb=x_osgb, y_osgb=y_osgb)

    assert x == np.mean(x_osgb)
    assert y == np.mean(y_osgb)


def test_get_lat_lon_center_from_osgb():
    """Test get lat lon center"""
    x_osgb, y_osgb = _get_random_osgb_coords_for_tests()

    x, y = geospatial.get_lat_lon_center_from_list_of_x_and_y_osgb(x_osgb=x_osgb, y_osgb=y_osgb)

    assert 49 < x < 50
    assert -8 < y < -7


def test_osgb_to_geostationary():  # noqa: D103
    x_osgb, y_osgb = _get_random_osgb_coords_for_tests()
    x_geos, y_geos = geospatial.osgb_to_geostationary(x=x_osgb, y=y_osgb)
    assert (-1133332 <= x_geos).all() and (x_geos <= -1133226).all()
    assert (4511163 <= y_geos).all() and (x_geos <= 4511222).all()
