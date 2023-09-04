""" Test for Metadata class"""
import tempfile

import pandas as pd
import pytest

from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.data_sources.fake.batch import metadata_fake
from nowcasting_dataset.data_sources.metadata.metadata_model import load_from_csv


def test_metadata_fake():
    """Test fake"""
    _ = metadata_fake(10)


def test_metadata_save():
    """Test save"""
    metadata = metadata_fake(10)

    with tempfile.TemporaryDirectory() as local_temp_path:
        metadata.save_to_csv(path=local_temp_path)


def test_metadata_save_twice():
    """Test save twice"""
    metadata1 = metadata_fake(10)
    metadata2 = metadata_fake(10)

    with tempfile.TemporaryDirectory() as local_temp_path:
        metadata1.save_to_csv(path=local_temp_path)
        metadata2.save_to_csv(path=local_temp_path)

        locations = pd.read_csv(
            f"{local_temp_path}/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"
        )

        assert len(locations) == 2 * metadata1.batch_size


def test_metadata_save_twice_and_load():
    """Test save twice"""
    batch_size = 10
    metadata1 = metadata_fake(batch_size)
    metadata2 = metadata_fake(batch_size)

    with tempfile.TemporaryDirectory() as local_temp_path:
        metadata1.save_to_csv(path=local_temp_path)
        metadata2.save_to_csv(path=local_temp_path)

        _ = load_from_csv(path=local_temp_path, batch_idx=0, batch_size=batch_size)
        _ = load_from_csv(path=local_temp_path, batch_idx=1, batch_size=batch_size)
        with pytest.raises(Exception):
            _ = load_from_csv(path=local_temp_path, batch_idx=2, batch_size=batch_size)
