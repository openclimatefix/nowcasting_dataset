""" Function to check Square"""
import pandas as pd

from nowcasting_dataset.square import BoundingBox, Square, get_closest_coordinate_order


def test_init():
    """Set up square"""
    square = Square(size_pixels=128, meters_per_pixel=1000)
    assert square._half_size_meters == 64_000


def test_bounding_box_centered_on():
    """Get bounding box"""
    square = Square(size_pixels=50, meters_per_pixel=1000)
    bounding_box = square.bounding_box_centered_on(x_center_osgb=20_000, y_center_osgb=40_000)
    assert bounding_box == BoundingBox(top=65_000, bottom=15_000, left=-5_000, right=45_000)


def test_get_closest_coordinate_order():
    """Test get_closest_coordinate_order"""
    x_center = 5
    y_center = 5

    true_order = [1, 0, 2]
    x = pd.Series(data=[4, 5, 10])
    y = pd.Series(data=[6, 5, 0])

    order = get_closest_coordinate_order(x_center=x_center, y_center=y_center, x=x, y=y)

    assert list(order.values) == true_order
