"""Test Manager."""
from datetime import datetime
from pathlib import Path

import nowcasting_dataset
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.manager import Manager


def test_sample_spatial_and_temporal_locations_for_examples():  # noqa: D103
    local_path = Path(nowcasting_dataset.__file__).parent.parent

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_dt=datetime(2019, 1, 1),
        end_dt=datetime(2019, 1, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    manager = Manager()
    manager.data_sources = {"gsp": gsp}
    manager.data_source_which_defines_geospatial_locations = gsp
    t0_datetimes = manager.get_t0_datetimes_across_all_data_sources(freq="30T")
    locations = manager.sample_spatial_and_temporal_locations_for_examples(
        t0_datetimes=t0_datetimes, n_examples=10
    )

    assert locations.columns.to_list() == ["t0_datetime_UTC", "x_center_OSGB", "y_center_OSGB"]
    assert len(locations) == 10
    assert (t0_datetimes[0] <= locations["t0_datetime_UTC"]).all()
    assert (t0_datetimes[-1] >= locations["t0_datetime_UTC"]).all()


def test_load_yaml_configuration():  # noqa: D103
    manager = Manager()
    local_path = Path(nowcasting_dataset.__file__).parent.parent
    filename = local_path / "tests" / "config" / "test.yaml"
    manager.load_yaml_configuration(filename=filename)
    manager.initialise_data_sources()
    assert len(manager.data_sources) == 6
    assert isinstance(manager.data_source_which_defines_geospatial_locations, GSPDataSource)


# TODO: Issue #322: Test the other Manager methods!
