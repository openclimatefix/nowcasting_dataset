"""DataSourceList class."""

import numpy as np
import pandas as pd
import logging

import nowcasting_dataset.time as nd_time
from nowcasting_dataset.dataset.split.split import SplitMethod, split_data, SplitName

logger = logging.getLogger(__name__)


class DataSourceList(list):
    """Hold a list of DataSource objects.

    The first DataSource in the list is used to compute the geospatial locations of each example.
    """

    def get_t0_datetimes_across_all_data_sources(self, freq: str) -> pd.DatetimeIndex:
        """
        Compute the intersection of the t0 datetimes available across all DataSources.

        Args:
            freq: The Pandas frequency string. The returned DatetimeIndex will be at this frequency,
                and every datetime will be aligned to this frequency.  For example, if
                freq='5 minutes' then every datetime will be at 00, 05, ..., 55 minutes
                past the hour.

        Returns:  Valid t0 datetimes, taking into consideration all DataSources,
            filtered by daylight hours (SatelliteDataSource.datetime_index() removes the night
            datetimes).
        """
        logger.debug("Get the intersection of time periods across all DataSources.")

        # Get the intersection of t0 time periods from all data sources.
        t0_time_periods_for_all_data_sources = []
        for data_source in self:
            logger.debug(f"Getting t0 time periods for {type(data_source).__name__}")
            try:
                t0_time_periods = data_source.get_contiguous_t0_time_periods()
            except NotImplementedError:
                pass  # Skip data_sources with no concept of time.
            else:
                t0_time_periods_for_all_data_sources.append(t0_time_periods)

        intersection_of_t0_time_periods = nd_time.intersection_of_multiple_dataframes_of_periods(
            t0_time_periods_for_all_data_sources
        )

        t0_datetimes = nd_time.time_periods_to_datetime_index(
            time_periods=intersection_of_t0_time_periods, freq=freq
        )

        return t0_datetimes

    def sample_position_of_every_example_of_every_split(
        self,
        t0_datetimes: pd.DatetimeIndex,
        split_method: SplitMethod,
        n_examples_per_split: dict[SplitName, int],
    ) -> dict[SplitName, pd.DataFrame]:
        """
        Computes the geospatial and temporal position of each training example.

        The first data_source in this DataSourceList defines the geospatial locations of
        each example.

        Args:
            t0_datetimes: All available t0 datetimes.  Can be computed with
                `DataSourceList.get_t0_datetimes_across_all_data_sources()`
            split_method: The method used to split data into train, validation, and test.
            n_examples_per_split: The number of examples requested for each split.

        Returns:
            A dict where the keys are a SplitName, and the values are a pd.DataFrame.
            Each row of each DataFrame specifies the position of each example, using
            columns: 't0_datetime_UTC', 'x_center_OSGB', 'y_center_OSGB'.
        """
        # Split t0_datetimes into train, test and validation sets.
        t0_datetimes_per_split = split_data(datetimes=t0_datetimes, method=split_method)
        t0_datetimes_per_split = t0_datetimes_per_split._asdict()

        data_source_which_defines_geo_position = self[0]

        positions_per_split: dict[SplitName, pd.DataFrame] = {}
        for split_name, t0_datetimes_for_split in t0_datetimes_per_split.items():
            n_examples = n_examples_per_split[split_name]
            shuffled_t0_datetimes = np.random.choice(t0_datetimes_for_split, shape=n_examples)
            x_locations, y_locations = data_source_which_defines_geo_position.get_locations(
                shuffled_t0_datetimes
            )
            positions_per_split[split_name] = pd.DataFrame(
                {
                    "t0_datetime_UTC": shuffled_t0_datetimes,
                    "x_center_OSGB": x_locations,
                    "y_center_OSGB": y_locations,
                }
            )

        return positions_per_split
