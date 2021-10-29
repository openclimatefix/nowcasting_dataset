from nowcasting_dataset.square import BoundingBox, Square


def test_init():
    square = Square(size_pixels=128, meters_per_pixel=1000)
    assert square._half_size_meters == 64_000


def test_bounding_box_centered_on():
    square = Square(size_pixels=50, meters_per_pixel=1000)
    bounding_box = square.bounding_box_centered_on(x_meters_center=20_000, y_meters_center=40_000)
    assert bounding_box == BoundingBox(top=65_000, bottom=15_000, left=-5_000, right=45_000)
