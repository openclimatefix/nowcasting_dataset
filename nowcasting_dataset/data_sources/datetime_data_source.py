""" Datetime DataSource - add hour and year features """
from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset import time as nd_time
from dataclasses import dataclass
import pandas as pd
from numbers import Number
from typing import List, Tuple


@dataclass
class DatetimeDataSource(DataSource):
    """ Add hour_of_day_{sin, cos} and day_of_year_{sin, cos} features. """

    def __post_init__(self):
        """ Post init """
        super().__post_init__()

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Example:
        """
        Get example data

        Args:
            t0_dt: list of timestamps
            x_meters_center: x center of patches - not needed
            y_meters_center: y center of patches - not needed

        Returns: batch data of datetime features

        """
        del x_meters_center, y_meters_center
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        index = pd.date_range(start_dt, end_dt, freq="5T")
        return nd_time.datetime_features_in_example(index)

    def get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        """ This method is not needed for DatetimeDataSource """
        raise NotImplementedError()

    def datetime_index(self) -> pd.DatetimeIndex:
        """ This method is not needed for DatetimeDataSource """
        raise NotImplementedError()
