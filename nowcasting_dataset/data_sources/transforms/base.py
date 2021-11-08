"""Generic Transform class"""

from dataclasses import dataclass
from typing import List

from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.dataset.batch import Batch


@dataclass
class Transform:
    """Abstract base class.

    Attributes:
      data_sources: List of data sources that this transform will be applied to
    """

    data_sources: List[DataSource]

    def apply_transform(self, batch: Batch) -> Batch:
        """
        Apply transform to the Batch, returning the Batch with added/transformed data

        Args:
            batch: Batch consisting of the data to transform

        Returns:
            Batch with the transformed data
        """
        return batch
