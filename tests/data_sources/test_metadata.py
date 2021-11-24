""" Test for Metadata class"""
import tempfile

import pandas as pd

from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.data_sources.fake import metadata_fake


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
