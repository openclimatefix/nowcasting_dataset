from nowcasting_dataset import geospatial
import numpy as np
import pyproj


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
        pyproj.transformer.\
            TransformerGroup(crs_from=geospatial.OSGB, crs_to=geospatial.WGS84).download_grids(verbose=True)

    finally:
        osgb_coords = geospatial.osgb_to_lat_lon(x=0, y=0)
        np.testing.assert_allclose(
            osgb_coords,
            (49.76680681317516, -7.557207277153569))

