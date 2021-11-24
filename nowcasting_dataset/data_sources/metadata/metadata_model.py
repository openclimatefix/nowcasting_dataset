""" Model for output of general/metadata data, useful for a batch """

from datetime import datetime
from typing import List

import pandas as pd
from pydantic import BaseModel, Field

from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.utils import get_start_and_end_example_index


class Metadata(BaseModel):
    """Class to store metedata data"""

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    t0_datetime_utc: List[datetime] = Field(
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

        # if file exists, add to it

        filename = f"{path}/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"
        metadata_dict = self.dict()
        metadata_dict.pop("batch_size")

        metadata_df = pd.DataFrame(metadata_dict)
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

    # read whole file
    metadata_df = pd.read_csv(filename)

    # get start and end example index
    start_example_idx, end_example_idx = get_start_and_end_example_index(
        batch_idx=batch_idx, batch_size=batch_size
    )

    # only select metadata we need
    metadata_df = metadata_df.iloc[start_example_idx:end_example_idx]

    # add batch_size
    metadata_dict = metadata_df.to_dict("list")
    metadata_dict["batch_size"] = batch_size

    return Metadata(**metadata_dict)
