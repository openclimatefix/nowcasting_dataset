""" Functions to make fake coordinates """

import numpy as np

from nowcasting_dataset.geospatial import lat_lon_to_osgb


def create_random_point_coordinates_osgb(size: int):
    """Make random coords [OSGB] for pv site, or gsp"""
    # this is about 100KM
    HUNDRED_KILOMETERS = 10 ** 5
    x = np.random.randint(0, HUNDRED_KILOMETERS, size)
    y = np.random.randint(0, HUNDRED_KILOMETERS, size)

    return add_uk_centroid_osgb(x, y)


def make_random_image_coords_osgb(size: int):
    """Make random coords for image. These are ranges for the pixels"""

    ONE_KILOMETER = 10 ** 3

    # 4 kilometer spacing seemed about right for real satellite images
    x = 4 * ONE_KILOMETER * np.array((range(0, size)))
    y = 4 * ONE_KILOMETER * np.array((range(size, 0, -1)))

    return add_uk_centroid_osgb(x, y)


def add_uk_centroid_osgb(x, y):
    """
    Add an OSGB value to make in center of UK

    Args:
        x: random values, OSGB
        y: random values, OSGB

    Returns: X,Y random coordinates [OSGB]
    """

    # get random OSGB center in the UK
    lat = np.random.uniform(51, 55)
    lon = np.random.uniform(-2.5, 1)
    x_center, y_center = lat_lon_to_osgb(lat=lat, lon=lon)

    # make average 0
    x = x - x.mean()
    y = y - y.mean()

    # put in the uk
    x = x + x_center
    y = y + y_center

    return x, y
