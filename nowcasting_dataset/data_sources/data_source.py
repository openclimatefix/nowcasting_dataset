from numbers import Number
import pandas as pd
from nowcasting_dataset.example import Example
from nowcasting_dataset.square import Square
from dataclasses import dataclass


@dataclass
class DataSource:
    """Abstract base class."""

    image_size: Square  #: Defines the image size of each example.

    def open(self):
        """Open the data source, if necessary.

        Called from each worker process.  Useful for data sources where the
        underlying data source cannot be forked (like Zarr on GCP!).

        Data sources which can be forked safely will open
        the underlying data source in __init__().
        """
        pass

    def available_timestamps(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available timesteps"""
        raise NotImplementedError()

    def get_example(
            self,
            start: pd.Timestamp,
            end: pd.Timestamp,
            t0: pd.Timestamp,
            x_meters: Number,  #: Centre, in OSGB coordinates.
            y_meters: Number  #: Centre, in OSGB coordinates.
    ) -> Example:
        """Must be overridden by child classes."""
        raise NotImplementedError()
