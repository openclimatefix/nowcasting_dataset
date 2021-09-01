from nowcasting_dataset.data_sources.gsp_data_source import get_gsp_metadata_from_eso
import pandas as pd


def test_get_gsp_metadata_from_eso():
    """
    Test to get the gsp metadata from eso. This should take ~1 second.
    @return:
    """
    metadata = get_gsp_metadata_from_eso()

    assert isinstance(metadata, pd.DataFrame)
    assert len(metadata) > 100
    assert "gnode_name" in metadata.columns
    assert "gnode_lat" in metadata.columns
    assert "gnode_lon" in metadata.columns
