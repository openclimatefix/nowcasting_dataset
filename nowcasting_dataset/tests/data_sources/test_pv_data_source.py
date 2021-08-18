from nowcasting_dataset.data_sources.pv_data_source import PVDataSource
from pathlib import Path
from datetime import datetime
import pytest


@pytest.mark.skip("CD does not have access to GCS - TODO")
def test_get_example_and_batch():
    BUCKET = Path("solar-pv-nowcasting-data")

    # Solar PV data
    PV_PATH = BUCKET / "PV/PVOutput.org"
    PV_DATA_FILENAME = PV_PATH / "UK_PV_timeseries_batch.nc"
    PV_METADATA_FILENAME = PV_PATH / "UK_PV_metadata.csv"

    pv_data_source = PVDataSource(
        history_len=6,
        forecast_len=12,
        convert_to_numpy=True,
        image_size_pixels=64,
        meters_per_pixel=2000,
        filename=PV_DATA_FILENAME,
        metadata_filename=f"gs://{PV_METADATA_FILENAME}",
        start_dt=datetime.fromisoformat("2019-01-01 00:00:00.000+00:00"),
        end_dt=datetime.fromisoformat("2019-01-02 00:00:00.000+00:00"),
        load_azimuth_and_elevation=True,
    )

    x_locations, y_locations = pv_data_source.get_locations_for_batch(pv_data_source.pv_power.index)

    example = pv_data_source.get_example(pv_data_source.pv_power.index[0], x_locations[0], y_locations[0])
    assert 'pv_yield' in example.keys()

    batch = pv_data_source.get_batch(pv_data_source.pv_power.index[0:5], x_locations[0:10], y_locations[0:10])
    assert len(batch) == 5
