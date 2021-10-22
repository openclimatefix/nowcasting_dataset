from nowcasting_dataset.data_sources.data_source import ImageDataSource


def test_image_data_source():
    _ = ImageDataSource(
        image_size_pixels=64,
        meters_per_pixel=2000,
        history_minutes=30,
        forecast_minutes=60,
    )
