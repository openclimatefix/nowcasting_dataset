"""Generic Transform class"""

from dataclasses import dataclass
from typing import List

from nowcasting_dataset.data_sources.data_source import DataSource, DataSourceOutput


@dataclass
class Transform:
    """Abstract base class.

    Attributes:
      data_sources: Data source that this transform will use
    """

    data_sources: List[DataSource]

    def apply_transforms(self, batch: DataSourceOutput) -> DataSourceOutput:
        """
        Apply transform to the Batch, returning the Batch with added/transformed data

        Args:
            batch: Batch consisting of the data to transform

        Returns:
            Datasource with the transformed data
        """
        return NotImplementedError


class Compose(Transform):
    """Applies list of transforms in order"""

    transforms: List[Transform]

    def apply_transforms(self, batch: DataSourceOutput) -> DataSourceOutput:
        """
        Apply list of transforms

        Args:
            batch: Batch containing data to be transformed

        Returns:
            Transformed data
        """

        for transform in self.transforms:
            batch = transform.apply_transforms(batch)

        return batch
