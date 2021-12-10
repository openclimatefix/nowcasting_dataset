""" Functions to make fake coordinates """

from typing import List, Optional

import numpy as np

from nowcasting_dataset.geospatial import lat_lon_to_osgb


def create_random_point_coordinates_osgb(
    size: int, x_center_osgb: Optional = None, y_center_osgb: Optional = None
):
    """Make random coords [OSGB] for pv site, or gsp"""
    # this is about 100KM
    HUNDRED_KILOMETERS = 10 ** 5
    x = np.random.randint(0, HUNDRED_KILOMETERS, size)
    y = np.random.randint(0, HUNDRED_KILOMETERS, size)

    return add_uk_centroid_osgb(x, y, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb)


def make_random_image_coords_osgb(
    size: int, x_center_osgb: Optional = None, y_center_osgb: Optional = None
):
    """Make random coords for image. These are ranges for the pixels"""

    ONE_KILOMETER = 10 ** 3

    # 4 kilometer spacing seemed about right for real satellite images
    x = 4 * ONE_KILOMETER * np.array((range(0, size)))
    y = 4 * ONE_KILOMETER * np.array((range(size, 0, -1)))

    return add_uk_centroid_osgb(
        x, y, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb, first_value_center=False
    )


def make_random_x_and_y_osgb_centers(batch_size: int) -> (List[int], List[int]):
    """Make X and Y OSGB centers"""
    lat = np.random.uniform(51, 55, batch_size)
    lon = np.random.uniform(-2.5, 0.5, batch_size)
    x_centers_osgb, y_centers_osgb = lat_lon_to_osgb(lat=lat, lon=lon)

    return x_centers_osgb, y_centers_osgb


def add_uk_centroid_osgb(
    x,
    y,
    x_center_osgb: Optional = None,
    y_center_osgb: Optional = None,
    first_value_center: bool = True,
):
    """
    Add an OSGB value to make in center of UK

    Args:
        x: random values, OSGB
        y: random values, OSGB
        y_center_osgb: TODO
        x_center_osgb: TODO
        first_value_center: TODO

    Returns: X,Y random coordinates [OSGB]
    """

    if (x_center_osgb is None) and (y_center_osgb is None):
        x_centers_osgb, y_centers_osgb = make_random_x_and_y_osgb_centers(batch_size=1)
        x_center_osgb = x_centers_osgb[0]
        y_center_osgb = y_centers_osgb[0]

    # normalize
    x = x - x.mean()
    y = y - y.mean()

    # put in the uk
    x = x + x_center_osgb
    y = y + y_center_osgb

    # make first value
    if first_value_center:
        x[0] = x_center_osgb
        y[0] = y_center_osgb

    return x, y
