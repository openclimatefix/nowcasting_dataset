""" PV Data Source """

import datetime
import functools
import io
import logging
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset import geospatial
from nowcasting_dataset.consts import DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE
from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.square import get_bounding_box_mask

logger = logging.getLogger(__name__)


@dataclass
class PVDataSource(ImageDataSource):
    """PV Data Source.

    This inherits from ImageDataSource so PVDataSource can select a geospatial region of interest
    defined by image_size_pixels and meters_per_pixel.
    """

    filename: Union[str, Path]
    metadata_filename: Union[str, Path]
    # TODO: Issue #425: Use config to set start_dt and end_dt.
    start_datetime: Optional[datetime.datetime] = None
    end_datetime: Optional[datetime.datetime] = None
    random_pv_system_for_given_location: Optional[bool] = True
    #: Each example will always have this many PV systems.
    #: If less than this number exist in the data then pad with NaNs.
    n_pv_systems_per_example: int = DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE
    load_azimuth_and_elevation: bool = False
    load_from_gcs: bool = True  # option to load data from gcs, or local file
    get_center: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """Post Init"""
        super().__post_init__(image_size_pixels, meters_per_pixel)

        self.rng = np.random.default_rng()
        self.load()

    def check_input_paths_exist(self) -> None:
        """Check input paths exist.  If not, raise a FileNotFoundError."""
        for filename in [self.filename, self.metadata_filename]:
            nd_fs_utils.check_path_exists(filename)

    def load(self):
        """
        Load metadata and pv power
        """
        self._load_metadata()
        self._load_pv_power()
        self.pv_metadata, self.pv_power = align_pv_system_ids(self.pv_metadata, self.pv_power)

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
        return PV

    def _load_metadata(self):

        logger.debug(f"Loading PV metadata from {self.metadata_filename}")

        pv_metadata = pd.read_csv(self.metadata_filename, index_col="system_id")
        pv_metadata.dropna(subset=["longitude", "latitude"], how="any", inplace=True)

        pv_metadata["location_x"], pv_metadata["location_y"] = geospatial.lat_lon_to_osgb(
            pv_metadata["latitude"], pv_metadata["longitude"]
        )

        # Remove PV systems outside the geospatial boundary of the
        # satellite data:
        GEO_BOUNDARY_OSGB = {
            "WEST": -238_000,
            "EAST": 856_000,
            "NORTH": 1_222_000,
            "SOUTH": -184_000,
        }
        self.pv_metadata = pv_metadata[
            (pv_metadata.location_x >= GEO_BOUNDARY_OSGB["WEST"])
            & (pv_metadata.location_x <= GEO_BOUNDARY_OSGB["EAST"])
            & (pv_metadata.location_y <= GEO_BOUNDARY_OSGB["NORTH"])
            & (pv_metadata.location_y >= GEO_BOUNDARY_OSGB["SOUTH"])
        ]

    def _load_pv_power(self):

        logger.debug(f"Loading PV Power data from {self.filename}")

        pv_power = load_solar_pv_data(
            self.filename, start_dt=self.start_datetime, end_dt=self.end_datetime
        )

        # A bit of hand-crafted cleaning
        if 30248 in pv_power.columns:
            pv_power[30248]["2018-10-29":"2019-01-03"] = np.NaN

        # Drop columns and rows with all NaNs.
        pv_power.dropna(axis="columns", how="all", inplace=True)
        pv_power.dropna(axis="index", how="all", inplace=True)

        # drop systems with over night power
        pv_power = drop_pv_systems_which_produce_overnight(pv_power)

        # Resample to 5-minutely and interpolate up to 15 minutes ahead.
        # TODO: Issue #301: Give users the option to NOT resample (because Perceiver IO
        # doesn't need all the data to be perfectly aligned).
        pv_power = pv_power.resample("5T").interpolate(method="time", limit=3)
        pv_power.dropna(axis="index", how="all", inplace=True)
        # self.pv_power = dd.from_pandas(pv_power, npartitions=3)
        print("pv_power = {:,.1f} MB".format(pv_power.values.nbytes / 1e6))
        self.pv_power = pv_power

        # get the max generation / capacity for each system
        self.pv_capacity = pv_power.max()

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> [pd.DataFrame]:
        # TODO: Cache this?
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        del t0_dt  # t0 is not used in the rest of this method!
        selected_pv_power = self.pv_power.loc[start_dt:end_dt].dropna(axis="columns", how="any")
        selected_pv_capacity = self.pv_capacity[selected_pv_power.columns]

        pv_power_zero_or_above_flag = selected_pv_power.ge(0).all()

        if pv_power_zero_or_above_flag.sum() != len(selected_pv_power.columns):
            n = len(selected_pv_power.columns) - pv_power_zero_or_above_flag.sum()
            logger.debug(f"Will be removing {n} pv systems as they have negative values")

        selected_pv_power = selected_pv_power.loc[:, pv_power_zero_or_above_flag]
        selected_pv_capacity = selected_pv_capacity.loc[pv_power_zero_or_above_flag]

        return selected_pv_power, selected_pv_capacity

    def _get_central_pv_system_id(
        self,
        x_meters_center: Number,
        y_meters_center: Number,
        pv_system_ids_with_data_for_timeslice: pd.Int64Index,
    ) -> int:
        # If x_meters_center and y_meters_center have been chosen
        # by PVDataSource.pick_locations_for_batch() then we just have
        # to find the pv_system_ids at that exact location.  This is
        # super-fast (a few hundred microseconds).  We use np.isclose
        # instead of the equality operator because floats.
        pv_system_ids = self.pv_metadata.index[
            np.isclose(self.pv_metadata.location_x, x_meters_center)
            & np.isclose(self.pv_metadata.location_y, y_meters_center)
        ]
        if len(pv_system_ids) == 0:
            # TODO: Implement finding PV systems closest to x_meters_center,
            # y_meters_center.  This will probably be quite slow, so always
            # try finding an exact match first (which is super-fast).
            raise NotImplementedError(
                "Not yet implemented the ability to find PV systems *nearest*"
                " (but not at the identical location to) x_meters_center and"
                " y_meters_center."
            )

        pv_system_ids = pv_system_ids_with_data_for_timeslice.intersection(pv_system_ids)
        assert len(pv_system_ids) > 0

        # Select just one PV system (the locations in PVOutput.org are quite
        # approximate, so it's quite common to have multiple PV systems
        # at the same nominal lat lon).
        if self.random_pv_system_for_given_location:
            pv_system_id = self.rng.choice(pv_system_ids)
        else:
            pv_system_id = pv_system_ids[0]

        return pv_system_id

    def _get_all_pv_system_ids_in_roi(
        self,
        x_meters_center: Number,
        y_meters_center: Number,
        pv_system_ids_with_data_for_timeslice: pd.Int64Index,
    ) -> pd.Int64Index:
        """
        Find the PV system IDs. for all the PV systems within the geospatial

        This is for all the PV systems within the geospatial
        region of interest, defined by self.square.
        """
        logger.debug(f"Getting PV example data for {x_meters_center} and {y_meters_center}")

        bounding_box = self._square.bounding_box_centered_on(
            x_meters_center=x_meters_center, y_meters_center=y_meters_center
        )
        x = self.pv_metadata.location_x
        y = self.pv_metadata.location_y

        # make mask of pv system ids
        mask = get_bounding_box_mask(bounding_box, x, y)
        pv_system_ids = self.pv_metadata.index[mask]

        pv_system_ids = pv_system_ids_with_data_for_timeslice.intersection(pv_system_ids)

        # there may not be any pv systems in a GSP region
        # assert len(pv_system_ids) > 0

        return pv_system_ids

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> xr.Dataset:
        """
        Get Example data for PV data

        Args:
            t0_dt: list of timestamps for the datetime of the batches. The batch will also include
                data for historic and future depending on 'history_minutes' and 'future_minutes'.
            x_meters_center: x center batch locations
            y_meters_center: y center batch locations

        Returns: Example data

        """
        logger.debug("Getting PV example data")

        selected_pv_power, selected_pv_capacity = self._get_time_slice(t0_dt)
        all_pv_system_ids = self._get_all_pv_system_ids_in_roi(
            x_meters_center, y_meters_center, selected_pv_power.columns
        )
        if self.get_center:
            central_pv_system_id = self._get_central_pv_system_id(
                x_meters_center, y_meters_center, selected_pv_power.columns
            )

            # By convention, the 'target' PV system ID (the one in the center
            # of the image) must be in the first position of the returned arrays.
            all_pv_system_ids = all_pv_system_ids.drop(central_pv_system_id)
            all_pv_system_ids = all_pv_system_ids.insert(loc=0, item=central_pv_system_id)

        all_pv_system_ids = all_pv_system_ids[: self.n_pv_systems_per_example]

        selected_pv_power = selected_pv_power[all_pv_system_ids]
        selected_pv_capacity = selected_pv_capacity[all_pv_system_ids]

        pv_system_row_number = np.flatnonzero(self.pv_metadata.index.isin(all_pv_system_ids))
        pv_system_x_coords = self.pv_metadata.location_x[all_pv_system_ids]
        pv_system_y_coords = self.pv_metadata.location_y[all_pv_system_ids]

        # Save data into the PV object...

        # convert to data array
        da = xr.DataArray(
            data=selected_pv_power.values,
            dims=["time", "id"],
            coords=dict(
                id=all_pv_system_ids.values.astype(int),
                time=selected_pv_power.index.values,
            ),
        )

        capacity = xr.DataArray(
            data=selected_pv_capacity.values,
            dims=["id"],
            coords=dict(
                id=all_pv_system_ids.values.astype(int),
            ),
        )

        # convert to dataset
        pv = da.to_dataset(name="power_mw")
        pv["capacity_mwp"] = capacity

        # add pv x coords
        x_coords = xr.DataArray(
            data=pv_system_x_coords.values,
            dims=["id"],
        )

        y_coords = xr.DataArray(
            data=pv_system_y_coords.values,
            dims=["id"],
        )
        pv_system_row_number = xr.DataArray(
            data=pv_system_row_number,
            dims=["id"],
        )
        pv["x_osgb"] = x_coords
        pv["y_osgb"] = y_coords
        pv["pv_system_row_number"] = pv_system_row_number

        # pad out so that there are always n_pv_systems_per_example, pad with zeros
        pad_n = self.n_pv_systems_per_example - len(pv.id)
        pv = pv.pad(id=(0, pad_n), power_mw=((0, 0), (0, pad_n)), constant_values=0)

        return pv

    def get_locations(self, t0_datetimes: pd.DatetimeIndex) -> Tuple[List[Number], List[Number]]:
        """Find a valid geographical location for each t0_datetime.

        Returns:  x_locations, y_locations. Each has one entry per t0_datetime.
            Locations are in OSGB coordinates.
        """
        # Set this up as a separate function, so we can cache the result!
        @functools.cache  # functools.cache requires Python >= 3.9
        def _get_pv_system_ids(t0_datetime: pd.Timestamp) -> pd.Int64Index:
            start_dt = self._get_start_dt(t0_datetime)
            end_dt = self._get_end_dt(t0_datetime)
            available_pv_data = self.pv_power.loc[start_dt:end_dt]
            columns_mask = available_pv_data.notna().all().values
            pv_system_ids = available_pv_data.columns[columns_mask]
            assert len(pv_system_ids) > 0
            return pv_system_ids

        # Pick a random PV system for each t0_datetime, and then grab
        # their geographical location.
        x_locations = []
        y_locations = []
        for t0_datetime in t0_datetimes:
            pv_system_ids = _get_pv_system_ids(t0_datetime)
            pv_system_id = self.rng.choice(pv_system_ids)

            # Get metadata for PV system
            metadata_for_pv_system = self.pv_metadata.loc[pv_system_id]
            x_locations.append(metadata_for_pv_system.location_x)
            y_locations.append(metadata_for_pv_system.location_y)

        return x_locations, y_locations

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes."""
        return self.pv_power.index


def load_solar_pv_data(
    filename: Union[str, Path],
    start_dt: Optional[datetime.datetime] = None,
    end_dt: Optional[datetime.datetime] = None,
) -> pd.DataFrame:
    """
    Load solar pv data from any compute environment.

    Args:
        filename: filename of file to be loaded
        start_dt: the start datetime, which to trim the data to
        end_dt: the end datetime, which to trim the data to

    Returns: Solar PV data
    """
    logger.debug(f"Loading Solar PV Data from {filename} from {start_dt} to {end_dt}.")

    # It is possible to simplify the code below and do
    # xr.open_dataset(file, engine='h5netcdf')
    # in the first 'with' block, and delete the second 'with' block.
    # But that takes 1 minute to load the data, where as loading into memory
    # first and then loading from memory takes 23 seconds!
    with fsspec.open(filename, mode="rb") as file:
        file_bytes = file.read()

    with io.BytesIO(file_bytes) as file:
        pv_power = xr.open_dataset(file, engine="h5netcdf")
        pv_power = pv_power.sel(datetime=slice(start_dt, end_dt))
        pv_power_df = pv_power.to_dataframe()

    # Save memory
    del file_bytes
    del pv_power

    # Process the data a little
    pv_power_df = pv_power_df.dropna(axis="columns", how="all")
    pv_power_df = pv_power_df.clip(lower=0, upper=5e7)
    # Convert the pv_system_id column names from strings to ints:
    pv_power_df.columns = [np.int32(col) for col in pv_power_df.columns]

    if "passiv" not in filename:
        pv_power_df = pv_power_df.tz_localize("Europe/London").tz_convert("UTC").tz_convert(None)

    logger.debug("Loading Solar PV Data: done")

    return pv_power_df


def align_pv_system_ids(
    pv_metadata: pd.DataFrame, pv_power: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Only pick PV systems for which we have metadata."""
    pv_system_ids = pv_metadata.index.intersection(pv_power.columns)
    pv_system_ids = np.sort(pv_system_ids)
    pv_power = pv_power[pv_system_ids]
    pv_metadata = pv_metadata.loc[pv_system_ids]
    return pv_metadata, pv_power


def drop_pv_systems_which_produce_overnight(pv_power: pd.DataFrame) -> pd.DataFrame:
    """Drop systems which produce power over night."""
    # TODO: Of these bad systems, 24647, 42656, 42807, 43081, 51247, 59919
    # might have some salvagable data?
    NIGHT_YIELD_THRESHOLD = 0.4
    night_hours = [22, 23, 0, 1, 2]
    pv_data_at_night = pv_power.loc[pv_power.index.hour.isin(night_hours)]
    pv_above_threshold_at_night = (pv_data_at_night > NIGHT_YIELD_THRESHOLD).any()
    bad_systems = pv_power.columns[pv_above_threshold_at_night]
    print(len(bad_systems), "bad PV systems found and removed!")
    return pv_power.drop(columns=bad_systems)
