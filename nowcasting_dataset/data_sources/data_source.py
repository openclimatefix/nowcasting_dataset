from typing import Optional
import pandas as pd
from nowcasting_dataset.example import Example
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataSource():
    """Abstract base class."""
    image_size_pixels: Optional[int] = 128

    def open(self, filename: Path):
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
            x: float,  #: Centre, in OSGB coordinates.
            y: float  #: Centre, in OSGB coordinates.
    ) -> Example:
        """Must be overridden by child classes."""
        raise NotImplementedError()
