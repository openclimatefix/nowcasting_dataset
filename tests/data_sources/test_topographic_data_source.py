import pytest
import pandas as pd
from nowcasting_dataset.data_sources import TopographicDataSource


@pytest.mark.parametrize(
    "x, y, left, right, top, bottom",
    [
        (0, 0, -128_000, 126_000, 128_000, -126_000),
        (10, 0, -126_000, 128_000, 128_000, -126_000),
        (30, 0, -126_000, 128_000, 128_000, -126_000),
        (1000, 0, -126_000, 128_000, 128_000, -126_000),
        (0, 1000, -128_000, 126_000, 128_000, -126_000),
        (1000, 1000, -126_000, 128_000, 128_000, -126_000),
        (2000, 2000, -126_000, 128_000, 130_000, -124_000),
        (2000, 1000, -126_000, 128_000, 128_000, -126_000),
        (2001, 2001, -124_000, 130_000, 130_000, -124_000),
    ],
)
def test_get_example(x, y, left, right, top, bottom):
    topo_source = TopographicDataSource(
        filename="/home/jacob/Development/nowcasting_dataset/tests/data/europe_dem_1km_osgb.tif",
        image_size_pixels=128,
        meters_per_pixel=2000,
        normalize=True,
        convert_to_numpy=True,
        forecast_minutes=300,
        history_minutes=10,
    )
    t0_dt = pd.Timestamp("2019-01-01T13:00")
    example = topo_source.get_example(t0_dt=t0_dt, x_meters_center=x, y_meters_center=y)
    sat_data = example["topo_data"]
    print(f"Top: {top} Bottom: {bottom} Example:")
    print(example["topo_y_coords"])
    print(f"Left: {left} Right: {right} Example:")
    print(example["topo_x_coords"])
    # Topo data has issues with exactly being correct
    assert left == sat_data.x.values[0]
    assert right == sat_data.x.values[-1]
    # sat_data.y is top-to-bottom.
    assert top == sat_data.y.values[0]
    assert bottom == sat_data.y.values[-1]
    assert len(sat_data.x) == pytest.IMAGE_SIZE_PIXELS
    assert len(sat_data.y) == pytest.IMAGE_SIZE_PIXELS
