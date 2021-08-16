from nowcasting_dataset import geospatial
import numpy as np
# import pyproj
import pandas as pd


def test_osgb_to_lat_lon():
    """
    Test to check transform function is working correctly
    """

    try:
        osgb_coords = geospatial.osgb_to_lat_lon(x=0, y=0)
        np.testing.assert_allclose(
            osgb_coords,
            (49.76680681317516, -7.557207277153569))

    except:
        # Sometimes this test fails, then run the following code.
        # This forces a fresh 'grid' to be downloaded.
        geospatial.download_grids()

    finally:
        osgb_coords = geospatial.osgb_to_lat_lon(x=0, y=0)
        np.testing.assert_allclose(
            osgb_coords,
            (49.76680681317516, -7.557207277153569))


def test_calculate_azimuth_and_elevation_angle():
    datestamps = pd.date_range('2021-01-01 00:00:00', '2021-01-05', freq='5T', tz='UTC')

    s = geospatial.calculate_azimuth_and_elevation_angle(location_x=529600.18758,
                                                         location_y=136150.294751,
                                                         datestamps=datestamps.to_pydatetime())

    assert len(s) == len(datestamps)
    assert 'azimuth' in s.columns
    assert 'elevation' in s.columns
