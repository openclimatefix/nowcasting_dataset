from nowcasting_dataset.data_sources.nwp_data_source import NWPDataSource
import nowcasting_dataset
import os


def test_nwp_data_source_init():

    path = os.path.dirname(nowcasting_dataset.__file__)

    # Solar PV data (test data)
    NWP_FILENAME = f"{path}/../tests/data/nwp_data/test.zarr"

    _ = NWPDataSource(
        filename=NWP_FILENAME,
        history_minutes=30,
        forecast_minutes=60,
        convert_to_numpy=True,
        n_timesteps_per_batch=8,
    )


def test_nwp_data_source_open():

    path = os.path.dirname(nowcasting_dataset.__file__)

    # Solar PV data (test data)
    NWP_FILENAME = f"{path}/../tests/data/nwp_data/test.zarr"

    nwp = NWPDataSource(
        filename=NWP_FILENAME,
        history_minutes=30,
        forecast_minutes=60,
        convert_to_numpy=True,
        n_timesteps_per_batch=8,
    )

    nwp.open()


def test_nwp_data_source_batch():

    path = os.path.dirname(nowcasting_dataset.__file__)

    # Solar PV data (test data)
    NWP_FILENAME = f"{path}/../tests/data/nwp_data/test.zarr"

    nwp = NWPDataSource(
        filename=NWP_FILENAME,
        history_minutes=30,
        forecast_minutes=60,
        convert_to_numpy=True,
        n_timesteps_per_batch=8,
    )

    nwp.open()

    t0_datetimes = nwp._data.init_time[2:10].values
    x = nwp._data.x[0:4].values
    y = nwp._data.x[0:4].values

    batch = nwp.get_batch(t0_datetimes=t0_datetimes, x_locations=x, y_locations=y)

    assert len(batch) == 4
