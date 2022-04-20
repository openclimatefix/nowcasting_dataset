""" Square objects """
from numbers import Number
from typing import NamedTuple, Union

import pandas as pd

from nowcasting_dataset.consts import Array


class BoundingBox(NamedTuple):
    """Bounding box tuple"""

    top: Union[Number, float]
    bottom: Union[Number, float]
    left: Union[Number, float]
    right: Union[Number, float]


class Square:
    """Class for computing bounding box for satellite imagery."""

    def __init__(self, size_pixels: int, meters_per_pixel: Number):
        """
        Init

        Args:
            size_pixels: number of pixels
            meters_per_pixel: how many meters for each pixel
        """
        self.size_pixels = size_pixels
        size_meters = size_pixels * meters_per_pixel
        self._half_size_meters = size_meters / 2

    def bounding_box_centered_on(self, x_center_osgb: Number, y_center_osgb: Number) -> BoundingBox:
        """
        Get bounding box from a centre

        Args:
            x_center_osgb: x center of the bounding box
            y_center_osgb: y center of the bounding box

        Returns: Bounding box

        """
        return BoundingBox(
            top=y_center_osgb + self._half_size_meters,
            bottom=y_center_osgb - self._half_size_meters,
            left=x_center_osgb - self._half_size_meters,
            right=x_center_osgb + self._half_size_meters,
        )


class Rectangle:
    """Class for computing bounding box for satellite imagery."""

    def __init__(self, size_pixels_height: int, size_pixels_width: int, meters_per_pixel: Number):
        """
        Init

        Args:
            size_pixels_height: number of pixels for height
            size_pixels_width: number of pixels for width
            meters_per_pixel: how many meters for each pixel
        """
        self.size_pixels_height = size_pixels_height
        self.size_pixels_width = size_pixels_width
        size_meters_height = size_pixels_height * meters_per_pixel
        self._half_size_meters_height = size_meters_height / 2
        size_meters_width = size_pixels_width * meters_per_pixel
        self._half_size_meters_width = size_meters_width / 2

    def bounding_box_centered_on(self, x_center_osgb: Number, y_center_osgb: Number) -> BoundingBox:
        """
        Get bounding box from a centre

        Args:
            x_center_osgb: x center of the bounding box
            y_center_osgb: y center of the bounding box

        Returns: Bounding box

        """
        return BoundingBox(
            top=y_center_osgb + self._half_size_meters_height,
            bottom=y_center_osgb - self._half_size_meters_height,
            left=x_center_osgb - self._half_size_meters_width,
            right=x_center_osgb + self._half_size_meters_width,
        )


def get_bounding_box_mask(bounding_box: BoundingBox, x: Array, y: Array) -> Array:
    """
    Get boundary box mask from x and y locations. I.e are the x,y coords in the boundaring box

    Args:
        bounding_box: Bounding box
        x: x coordinates
        y: y coordinates

    Returns: list of booleans if the x and y coordinates are in the bounding box

    """
    mask = (
        (x >= bounding_box.left)
        & (x <= bounding_box.right)
        & (y >= bounding_box.bottom)
        & (y <= bounding_box.top)
    )
    return mask


def get_closest_coordinate_order(
    x_center: Number, y_center: Number, x: pd.Series, y: pd.Series
) -> pd.Series:
    """
    Get an order for the coordinates that are closes to the center

    Args:
        x_center: the center x coordinate
        y_center: the center y coordinate
        x: list of x coordinates
        y: list of y coordinates

    Returns: list of index, 0 being the closes, 1 being the next closes to the center.

    """

    assert len(x) == len(y)

    d = ((x - x_center) ** 2 + (y - y_center) ** 2) ** 0.5

    return d.argsort()
