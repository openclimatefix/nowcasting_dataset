""" Model for output of general/metadata data, useful for a batch """

from typing import List

import pandas as pd
from pydantic import BaseModel, Field

from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.filesystem.utils import check_path_exists
from nowcasting_dataset.utils import get_start_and_end_example_index


class Metadata(BaseModel):
    """Class to store metadata data"""

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    t0_datetime_utc: List[pd.Timestamp] = Field(
        ...,
        description="The t0s of each example ",
    )

    x_center_osgb: List[int] = Field(
        ...,
        description="The x centers of each example in OSGB coordinates",
    )

    y_center_osgb: List[int] = Field(
        ...,
        description="The y centers of each example in OSGB coordinates",
    )

    def save_to_csv(self, path):
        """
        Save metadata to a csv file

        Args:
            path: the path where the file shold be save

        """

        filename = f"{path}/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"
        metadata_dict = self.dict()
        metadata_dict.pop("batch_size")

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


def load_from_csv(path, batch_idx, batch_size) -> Metadata:
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
    start_example_idx, end_example_idx = get_start_and_end_example_index(
        batch_idx=batch_idx, batch_size=batch_size
    )

    names = ["t0_datetime_utc", "x_center_osgb", "y_center_osgb"]

    # read the file
    metadata_df = pd.read_csv(
        filename,
        skiprows=start_example_idx + 1,  # p+1 is to ignore header
        nrows=batch_size,
        names=names,
    )

    assert (
        len(metadata_df) > 0
    ), f"Could not load metadata for {batch_size=} {batch_idx=} {filename=}"

    # add batch_size
    metadata_dict = metadata_df.to_dict("list")
    metadata_dict["batch_size"] = batch_size

    return Metadata(**metadata_dict)
