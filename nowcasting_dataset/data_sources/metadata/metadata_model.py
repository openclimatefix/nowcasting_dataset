""" Model for output of general/metadata data, useful for a batch """

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, validator

from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.filesystem.utils import check_path_exists
from nowcasting_dataset.utils import get_start_and_end_example_index

logger = logging.getLogger(__name__)


class SpaceTimeLocation(BaseModel):
    """Location of the example"""

    t0_datetime_utc: pd.Timestamp = Field(
        ...,
        description="The t0 of one example ",
    )

    x_center_osgb: float = Field(
        ...,
        description="The x center of one example in OSGB coordinates",
    )

    y_center_osgb: float = Field(
        ...,
        description="The y center of one example in OSGB coordinates",
    )

    id: Optional[int] = Field(
        None,
        description="The id of the GSP or the PV system. This is optional so can be None",
    )

    id_type: Optional[str] = Field(
        None,
        description="The type of the id. Should be either None, 'gsp' or 'pv_system'",
    )

    @validator("t0_datetime_utc")
    def v_t0_datetime_utc(cls, t0_datetime_utc):
        """Make sure t0_datetime_utc is pandas Timestamp"""
        return pd.Timestamp(t0_datetime_utc)

    @validator("id_type")
    def v_id_type(cls, id_type):
        """Make sure id_type is either None, 'gsp' or 'pv_system'"""

        if id_type == "None":
            id_type = None

        assert id_type in [
            None,
            "gsp",
            "pv_system",
        ], f"{id_type=} should be None, 'gsp' or 'pv_system'"
        return id_type


class Metadata(BaseModel):
    """Class to store metadata data"""

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    space_time_locations: List[SpaceTimeLocation]

    @property
    def t0_datetimes_utc(self) -> list:
        """Return all the t0"""
        return [location.t0_datetime_utc for location in self.space_time_locations]

    @property
    def x_centers_osgb(self) -> List[float]:
        """List of all the x centers from all the locations"""
        return [location.x_center_osgb for location in self.space_time_locations]

    @property
    def y_centers_osgb(self) -> List[float]:
        """List of all the x centers from all the locations"""
        return [location.y_center_osgb for location in self.space_time_locations]

    @property
    def ids(self) -> List[float]:
        """List of all the ids from all the locations"""
        return [location.id for location in self.space_time_locations]

    def save_to_csv(self, path):
        """
        Save metadata to a csv file

        Args:
            path: the path where the file should be save

        """

        filename = f"{path}/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"
        metadata_dict = [location.dict() for location in self.space_time_locations]
        # metadata_dict.pop("batch_size")

        # if file exists, add to it
        try:
            check_path_exists(filename)
        except FileNotFoundError:
            metadata_df = pd.DataFrame(metadata_dict)

        else:

            metadata_df = pd.read_csv(filename)

            metadata_df_extra = pd.DataFrame(metadata_dict)
            metadata_df = metadata_df.append(metadata_df_extra)

        metadata_df.to_csv(filename, index=False)


def load_from_csv(
    path: Union[str, Path], batch_size: int, batch_idx: Optional[int] = None
) -> Metadata:
    """
    Load metadata from csv

    Args:
        path: the path which stores the metadata file
        batch_idx: the batch index
        batch_size: how many examples in each batch

    Returns: Metadata class
    """
    filename = f"{path}/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"

    # get start and end example index
    if batch_idx is not None:
        start_example_idx, end_example_idx = get_start_and_end_example_index(
            batch_idx=batch_idx, batch_size=batch_size
        )
        skiprows = start_example_idx + 1  # p+1 is to ignore header
        nrows = batch_size
    else:
        skiprows = 1  # ignore header
        nrows = None

    names = list(SpaceTimeLocation.__fields__)

    # read the file
    # kswargs = {}
    # if (start_example_idx is not None) and (end_example_idx is not None):
    #     kswargs['nrows'] = batch_size
    metadata_df = pd.read_csv(
        filename,
        skiprows=skiprows,
        nrows=nrows,
        names=names,
    )
    # If the CSV doesn't contain the optional columns, then drop those missing columns.
    # (Otherwise Pandas creates a column with NaNs, which confuses Pydantic!)
    metadata_df = metadata_df.dropna(axis="columns", how="all")

    assert (
        len(metadata_df) > 0
    ), f"Could not load metadata for {batch_size=} {batch_idx=} {filename=}"

    if "id_type" in metadata_df.columns:
        metadata_df["id_type"].fillna("None", inplace=True)

    # add batch_size
    locations_dict = metadata_df.to_dict("records")
    metadata_dict = {"space_time_locations": locations_dict, "batch_size": batch_size}

    try:
        metadata = Metadata(**metadata_dict)
    except Exception as e:
        logger.error(f"Try to make Metadata object from {metadata_dict}")
        raise e

    return metadata
