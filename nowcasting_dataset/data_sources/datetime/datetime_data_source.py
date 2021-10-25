""" Datetime DataSource - add hour and year features """
from dataclasses import dataclass
from numbers import Number
from typing import List, Tuple

import pandas as pd

from nowcasting_dataset import time as nd_time
from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.data_sources.datetime.datetime_model import Datetime
from nowcasting_dataset.dataset.xr_utils import make_dim_index


@dataclass
class DatetimeDataSource(DataSource):
    """ Add hour_of_day_{sin, cos} and day_of_year_{sin, cos} features. """

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Datetime:
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

        datetime_xr_dataset = nd_time.datetime_features_in_example(index).rename({"index": "time"})

        # make sure time is indexes in the correct way
        datetime_xr_dataset = make_dim_index(datetime_xr_dataset)

        return Datetime(datetime_xr_dataset)
