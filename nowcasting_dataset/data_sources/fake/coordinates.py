""" Functions to make fake coordinates """

from numbers import Number
from typing import Optional

import numpy as np

from nowcasting_dataset.geospatial import lat_lon_to_osgb


def create_random_point_coordinates_osgb(
    size: int, x_center_osgb: Optional[Number] = None, y_center_osgb: Optional[Number] = None
):
    """Make random coords [OSGB] for pv site, or gsp."""
    # this is about 100KM
    HUNDRED_KILOMETERS = 10**5
    x = np.random.randint(0, HUNDRED_KILOMETERS, size)
    y = np.random.randint(0, HUNDRED_KILOMETERS, size)

    return add_uk_centroid_osgb(x, y, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb)


def make_image_coords_osgb(
    size_y: int,
    size_x: int,
    x_center_osgb: Optional[Number] = None,
    y_center_osgb: Optional[Number] = None,
    km_spacing: int = 4,
    two_dimensional: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make coords for image. These are ranges for the pixels.

    Args:
        size_y: The size of the coordinates to make in height
        size_x: The size of the coordinates to make in width
        x_center_osgb: center coordinates for x (in osgb)
        y_center_osgb: center coordinates for y (in osgb)
        km_spacing: the km spacing between the coordinates.
        two_dimensional: If True, then each coord array will be 2D.
            This is useful because the real satellite data has 2D OSGB coords.

    Returns: X,Y coordinates [OSGB]
        These are each 1D if `two_dimensional` is False.
        If `two_dimensional` is True then both are of shape (size_y, size_x)
    """
    ONE_KILOMETER = 10**3

    # 4 kilometer spacing seemed about right for real satellite images
    x = km_spacing * ONE_KILOMETER * np.array((range(0, size_x)))
    y = km_spacing * ONE_KILOMETER * np.array((range(size_y, 0, -1)))

    if two_dimensional:
        x = np.tile(x, (size_y, 1))

        # By default, `np.tile` prepends an axis. We want to append an axis:
        y = np.expand_dims(y, axis=1)
        y = np.tile(y, (1, size_x))

    return add_uk_centroid_osgb(
        x, y, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb, first_value_center=False
    )


def make_random_x_and_y_osgb_centers(batch_size: int) -> tuple[list[int], list[int]]:
    """Make X and Y OSGB centers."""
    lat = np.random.uniform(51, 55, batch_size)
    lon = np.random.uniform(-2.5, 0.5, batch_size)
    x_centers_osgb, y_centers_osgb = lat_lon_to_osgb(lat=lat, lon=lon)

    return x_centers_osgb, y_centers_osgb


def add_uk_centroid_osgb(
    x: np.ndarray,
    y: np.ndarray,
    x_center_osgb: Optional[Number] = None,
    y_center_osgb: Optional[Number] = None,
    first_value_center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add an OSGB value to make in center of UK

    Args:
        x: OSGB coord values
        y: OSGB coord values
        y_center_osgb: TODO
        x_center_osgb: TODO
        first_value_center: TODO

    Returns: X,Y coordinates [OSGB]
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
