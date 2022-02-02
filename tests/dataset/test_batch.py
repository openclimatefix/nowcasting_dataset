"""Test Batch."""
import os
import tempfile

import pytest

from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch, join_two_batches


@pytest.fixture
def configuration():  # noqa: D103
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    return con


def test_model(configuration):  # noqa: D103
    _ = Batch.fake(configuration=configuration)


def test_model_align_in_time(configuration):  # noqa: D103
    batch = Batch.fake(configuration=configuration, temporally_align_examples=True)

    assert batch.metadata.t0_datetimes_utc[0] == batch.metadata.t0_datetimes_utc[1]


def test_model_nwp_channels(configuration):  # noqa: D103

    configuration.input_data = configuration.input_data.set_all_to_defaults()
    configuration.process.batch_size = 4
    configuration.input_data.nwp.nwp_channels = ["dlwrf"]

    batch = Batch.fake(configuration=configuration)

    assert batch.nwp.channels[0] == ["dlwrf"]


def test_model_save_to_netcdf(configuration):  # noqa: D103
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/000000.nc")


def test_model_load_from_netcdf(configuration):  # noqa: D103
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

        batch = Batch.load_netcdf(batch_idx=0, local_netcdf_path=dirpath)

        assert batch.satellite is not None


def test_model_download_and_load_from_netcdf(configuration):  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp_path, tempfile.TemporaryDirectory() as src_path:
        Batch.fake(configuration=configuration).save_netcdf(path=src_path, batch_i=0)

        batch = Batch.download_batch_and_load_batch(
            batch_idx=0, tmp_path=tmp_path, src_path=src_path
        )

        assert batch.satellite is not None
        assert os.path.exists(f"{tmp_path}/satellite/000000.nc")


def test_model_load_partial_from_netcdf(configuration):  # noqa: D103
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

        batch = Batch.load_netcdf(
            batch_idx=0, local_netcdf_path=dirpath, data_sources_names=["pv", "hrvsatellite"]
        )

        assert batch.satellite is None
        assert batch.pv is not None
        assert batch.hrvsatellite is not None


def test_model_load_partial_from_netcdf_error(configuration):  # noqa: D103
    with tempfile.TemporaryDirectory() as dirpath:
        batch = Batch.fake(configuration=configuration)
        batch.pv = batch.pv.rename({"time_index": "fake_time_index", "time": "fake_time"})
        batch.save_netcdf(path=dirpath, batch_i=0)

        with pytest.raises(Exception):
            batch = Batch.load_netcdf(
                batch_idx=0, local_netcdf_path=dirpath, data_sources_names=["pv", "hrvsatellite"]
            )


def test_join_two_batches(configuration):
    """
    Test to check batches join together
    """
    batch_1 = Batch.fake(configuration=configuration)
    batch_2 = Batch.fake(configuration=configuration)

    first_batch_examples = [2, 3]
    joined_batch = join_two_batches(
        batches=[batch_1, batch_2], first_batch_examples=first_batch_examples
    )

    for index in first_batch_examples:
        assert (joined_batch.satellite.data[index] == batch_1.satellite.data[index]).all()

    # random join
    _ = join_two_batches(batches=[batch_1, batch_2])


def test_join_random_two_batches(configuration):
    """
    Test to check batches join together randomly
    """
    batch_1 = Batch.fake(configuration=configuration)
    batch_2 = Batch.fake(configuration=configuration)

    # random join
    _ = join_two_batches(batches=[batch_1, batch_2])


def test_join_one_batches(configuration):
    """
    Test to check join together one batch
    """
    batch_1 = Batch.fake(configuration=configuration)

    joined_batch = join_two_batches(batches=[batch_1])

    assert (joined_batch.satellite.data == batch_1.satellite.data).all()
