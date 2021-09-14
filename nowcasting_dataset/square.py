from typing import NamedTuple, Union
from numbers import Number

from nowcasting_dataset.consts import Array


class BoundingBox(NamedTuple):
    top: Union[Number, float]
    bottom: Union[Number, float]
    left: Union[Number, float]
    right: Union[Number, float]


class Square:
    """"Class for computing bounding box for satellite imagery."""

    def __init__(self, size_pixels: int, meters_per_pixel: Number):
        self.size_pixels = size_pixels
        size_meters = size_pixels * meters_per_pixel
        self._half_size_meters = size_meters / 2

    def bounding_box_centered_on(
        self, x_meters_center: Number, y_meters_center: Number
    ) -> BoundingBox:
        return BoundingBox(
            top=y_meters_center + self._half_size_meters,
            bottom=y_meters_center - self._half_size_meters,
            left=x_meters_center - self._half_size_meters,
            right=x_meters_center + self._half_size_meters,
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
