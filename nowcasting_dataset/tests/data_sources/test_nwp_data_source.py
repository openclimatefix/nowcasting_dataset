from nowcasting_dataset.data_sources.nwp_data_source import NWPDataSource
import nowcasting_dataset
import os


def test_nwp_data_source_init():

    path = os.path.dirname(nowcasting_dataset.__file__)

    # Solar PV data (test data)
    NWP_FILENAME = f"{path}/tests/data/nwp_data/test.zarr"

    _ = NWPDataSource(
        filename=NWP_FILENAME,
        history_len=6,
        forecast_len=12,
        convert_to_numpy=True,
        n_timesteps_per_batch=8,
    )
