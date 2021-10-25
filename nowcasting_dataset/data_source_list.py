"""DataSourceList class."""

import pandas as pd
import logging

import nowcasting_dataset.time as nd_time

logger = logging.getLogger(__name__)


class DataSourceList(list):
    """Hold a list of DataSource objects."""

    def get_t0_datetimes_across_all_data_sources(self, freq: str) -> pd.DatetimeIndex:
        """
        Compute the intersection of the t0 datetimes available across all `data_sources`.

        Returns the valid t0 datetimes, taking into consideration all DataSources,
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

    """
    def compute_and_save_positions_of_each_example_of_each_split(
            self,
            split_method: SplitMethod,
            n_examples_per_split: dict[SplitMethod, int],
            dst_path: Path
    ) -> None:
        Computes the geospatial and temporal position of each training example.

        Finds the time periods available across all data_sources.

        Args:
            data_sources: A list of DataSources.  The first data_source is used to define the geospatial
                location of each example.
            split_method: The method used to split the available data into train, validation, and test.
            n_examples_per_split: The number of examples requested for each split.
            dst_path: The destination path.  This is where the CSV files will be saved into.
                CSV files will be saved into dst_path / split_method / 'positions_of_each_example.csv'.

        # Get intersection of all available t0_datetimes.  Current done by NowcastingDataModule._get_datetimes():
        # github.com/openclimatefix/nowcasting_dataset/blob/main/nowcasting_dataset/dataset/datamodule.py#L364
        t0_datetimes_for_all_data_sources = [data_source.get_t0_datetimes() for data_source in data_sources]
        intersection_of_t0_datetimes = nd_time.intersection_of_datetimeindexes(t0_datetimes_for_all_data_sources)

        # Split t0_datetimes into train, test and validation sets (being careful to ensure each group is
        # at least `total_seq_duration` apart).  Currently done by NowcastingDataModule._split_data():
        # github.com/openclimatefix/nowcasting_dataset/blob/main/nowcasting_dataset/dataset/datamodule.py#L315
        t0_datetimes_per_split: dict[SplitName, pd.DatetimeIndex] = split_datetimes(
            intersection_of_t0_datetimes, method=split_method)

        for split_name, t0_datetimes_for_split in t0_datetimes_per_split.items():
            n_examples = n_examples_per_split[split_name]
            positions = compute_positions_of_each_example(t0_datetimes_for_split, data_source, n_examples)
            filename = dst_path / split_name / 'positions_of_each_example.csv'
            positions.to_csv(filename)
        """
