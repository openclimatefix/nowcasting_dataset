""" Test for ESO GSP metadata """
import geopandas as gpd
import pandas as pd
import pytest

from nowcasting_dataset.data_sources.gsp.eso import (
    get_gsp_metadata_from_eso,
    get_gsp_shape_from_eso,
)


def test_get_gsp_metadata_from_eso():
    """
    Test to get the gsp metadata from eso. This should take ~1 second.
    """
    metadata = get_gsp_metadata_from_eso()

    assert metadata["gsp_id"].is_unique == 1

    assert isinstance(metadata, pd.DataFrame)
    assert len(metadata) > 100
    assert "gsp_id" in metadata.columns
    # assert "gnode_lat" in metadata.columns
    # assert "gnode_lon" in metadata.columns


def test_get_pv_gsp_shape():
    """
    Test to get the gsp metadata from eso. This should take ~1 second.
    """

    gsp_shapes = get_gsp_shape_from_eso()

    assert gsp_shapes["RegionID"].is_unique

    assert isinstance(gsp_shapes, gpd.GeoDataFrame)
    assert "RegionID" in gsp_shapes.columns
    # assert "RegionName" in gsp_shapes.columns
    assert "geometry" in gsp_shapes.columns


@pytest.mark.skip("Skip this for the moment as not needed for CI")
def test_get_pv_gsp_shape_from_eso():
    """
    Test to get the gsp metadata from eso. This should take ~1 second.
    """

    gsp_shapes = get_gsp_shape_from_eso(load_local_file=False)

    assert gsp_shapes["RegionID"].is_unique

    assert isinstance(gsp_shapes, gpd.GeoDataFrame)
    assert "RegionID" in gsp_shapes.columns
    assert "RegionName" in gsp_shapes.columns
    assert "geometry" in gsp_shapes.columns
