"""DataSourceList class."""

import logging

import numpy as np
import pandas as pd

import nowcasting_dataset.time as nd_time
import nowcasting_dataset.utils as nd_utils
from nowcasting_dataset.config import model
from nowcasting_dataset import data_sources
logger = logging.getLogger(__name__)


class DataSourceList(list):
    """Hold a list of DataSource objects.

    Attrs:
      data_source_which_defines_geospatial_locations: The DataSource used to compute the
        geospatial locations of each example.
    """

    @classmethod
    def from_config(cls, config_for_all_data_sources: model.InputData):
        """Create a DataSource List from an InputData configuration object.

        For each key in each DataSource's configuration object, the string `<data_source_name>_`
        is removed from the key before passing to the DataSource constructor.  This allows us to
        have verbose field names in the configuration YAML files, whilst also using standard
        constructor arguments for DataSources.
        """
        data_source_name_to_class = {
            "pv": data_sources.PVDataSource,
            "satellite": data_sources.SatelliteDataSource,
            "nwp": data_sources.NWPDataSource,
            "gsp": data_sources.GSPDataSource,
            "topographic": data_sources.TopographicDataSource,
            "sun": data_sources.SunDataSource,
        }
        data_source_list = cls([])
        for data_source_name, data_source_class in data_source_name_to_class.items():
            logger.debug(f"Creating {data_source_name} DataSource object.")
            config_for_data_source = getattr(config_for_all_data_sources, data_source_name)
            if config_for_data_source is None:
                logger.info(f"No configuration found for {data_source_name}.")
                continue
            config_for_data_source = config_for_data_source.dict()

            # Strip `<data_source_name>_` from the config option field names.
            config_for_data_source = nd_utils.remove_regex_pattern_from_keys(
                config_for_data_source, pattern_to_remove=f"^{data_source_name}_"
            )

            try:
                data_source = data_source_class(**config_for_data_source)
            except Exception:
                logger.exception(f"Exception whilst instantiating {data_source_name}!")
                raise
            data_source_list.append(data_source)
            if (
                data_source_name
                == config_for_all_data_sources.data_source_which_defines_geospatial_locations
            ):
                data_source_list.data_source_which_defines_geospatial_locations = data_source
                logger.info(
                    f"DataSource {data_source_name} set as"
                    " data_source_which_defines_geospatial_locations"
                )

        try:
            _ = data_source_list.data_source_which_defines_geospatial_locations
        except AttributeError:
            logger.warning(
                "No DataSource configured as data_source_which_defines_geospatial_locations!"
            )
        return data_source_list

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

    def sample_spatial_and_temporal_locations_for_examples(
        self, t0_datetimes: pd.DatetimeIndex, n_examples: int
    ) -> pd.DataFrame:
        """
        Computes the geospatial and temporal locations for each training example.

        The first data_source in this DataSourceList defines the geospatial locations of
        each example.

        Args:
            t0_datetimes: All available t0 datetimes.  Can be computed with
                `DataSourceList.get_t0_datetimes_across_all_data_sources()`
            n_examples: The number of examples requested.

        Returns:
            Each row of each the DataFrame specifies the position of each example, using
            columns: 't0_datetime_UTC', 'x_center_OSGB', 'y_center_OSGB'.
        """
        # This code is for backwards-compatibility with code which expects the first DataSource
        # in the list to be used to define which DataSource defines the spatial location.
        # TODO: Remove this try block after implementing issue #213.
        try:
            data_source_which_defines_geospatial_locations = (
                self.data_source_which_defines_geospatial_locations
            )
        except AttributeError:
            data_source_which_defines_geospatial_locations = self[0]

        shuffled_t0_datetimes = np.random.choice(t0_datetimes, size=n_examples)
        x_locations, y_locations = data_source_which_defines_geospatial_locations.get_locations(
            shuffled_t0_datetimes
        )
        return pd.DataFrame(
            {
                "t0_datetime_UTC": shuffled_t0_datetimes,
                "x_center_OSGB": x_locations,
                "y_center_OSGB": y_locations,
            }
        )
