from typing import NamedTuple
from numbers import Number


class BoundingBox(NamedTuple):
    top: Number
    bottom: Number
    left: Number
    right: Number


class Square:
    """"Class for computing bounding box for satellite imagery."""

    def __init__(self, size_pixels: int, meters_per_pixel: Number):
        self.size_pixels = size_pixels
        size_meters = size_pixels * meters_per_pixel
        self._half_size_meters = size_meters / 2

    def bounding_box_centered_on(
            self,
            x_meters_center: Number,
            y_meters_center: Number
    ) -> BoundingBox:
        return BoundingBox(
            top=y_meters_center + self._half_size_meters,
            bottom=y_meters_center - self._half_size_meters,
            left=x_meters_center - self._half_size_meters,
            right=x_meters_center + self._half_size_meters)
