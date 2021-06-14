from nowcasting_dataset import geospatial
import numpy as np


def test_osgb_to_lat_lon():
    osgb_coords = geospatial.osgb_to_lat_lon(x=0, y=0)
    np.testing.assert_allclose(
        osgb_coords,
        (49.76680681317516, -7.557207277153569))
