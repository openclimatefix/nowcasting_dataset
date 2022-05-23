""" PV Data Source """

import datetime
import functools
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
from nowcasting_dataset.config.model import PVFiles
from nowcasting_dataset.consts import DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE
from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation
from nowcasting_dataset.data_sources.pv.live import (
    get_metadata_from_database,
    get_pv_power_from_database,
)
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.pv.utils import encode_label
from nowcasting_dataset.square import get_bounding_box_mask, get_closest_coordinate_order

logger = logging.getLogger(__name__)


@dataclass
class PVDataSource(ImageDataSource):
    """PV Data Source.

    This inherits from ImageDataSource so PVDataSource can select a geospatial region of interest
    defined by image_size_pixels and meters_per_pixel.
    """

    files_groups: List[Union[PVFiles, dict]]
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
    is_live: bool = False
    live_interpolate_minutes: int = 30
    live_load_extra_minutes: int = 60

    def __post_init__(
        self, image_size_pixels_height: int, image_size_pixels_width: int, meters_per_pixel: int
    ):
        """Post Init"""

        if type(self.files_groups[0]) == dict:
            self.files_groups = [PVFiles(**files) for files in self.files_groups]

        super().__post_init__(image_size_pixels_height, image_size_pixels_width, meters_per_pixel)

        self.rng = np.random.default_rng()
        self.load()

    def check_input_paths_exist(self) -> None:
        """Check input paths exist.

        If using live data, dont check Files, as we are loading data from the database
        If not, raise a FileNotFoundError.
        """
        if self.is_live:
            return None

        for pv_files in self.files_groups:
            for filename in [pv_files.pv_filename, pv_files.pv_metadata_filename]:
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

        logger.debug(f"Loading PV metadata from {self.files_groups}")

        # collect all metadata together
        if not self.is_live:
            pv_metadata = []
            for pv_files in self.files_groups:
                metadata_filename = pv_files.pv_metadata_filename

                # read metadata file
                metadata = pd.read_csv(metadata_filename, index_col="system_id")

                # encode index, to make sure the indexes are unique
                metadata.index = encode_label(indexes=metadata.index, label=pv_files.label)

                # filter for zero capacity
                if pv_files.label == "pvoutput":
                    metadata = metadata[metadata["system_size_watts"] > 0]
                elif pv_files.label == "passiv":
                    metadata = metadata[metadata["kwp"] > 0]

                pv_metadata.append(metadata)
            pv_metadata = pd.concat(pv_metadata)

        else:
            pv_metadata = get_metadata_from_database()

            logger.debug(f"Found {len(pv_metadata)} pv systems from database")

        # drop any systems with no lon or lat
        pv_metadata.dropna(subset=["longitude", "latitude"], how="any", inplace=True)

        logger.debug(f"Found {len(pv_metadata)} pv systems with locations")

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

        logger.debug(f"Found {len(pv_metadata)} pv systems")

    def _load_pv_power(self):

        logger.debug(f"Loading PV Power data from {self.files_groups}")

        if not self.is_live:
            # collect all PV power timeseries together
            pv_power_all = []
            for pv_files in self.files_groups:
                filename = pv_files.pv_filename

                # get pv power data
                pv_power = load_solar_pv_data(
                    filename, start_dt=self.start_datetime, end_dt=self.end_datetime
                )

                # encode index, to make sure the columns are unique
                new_columns = encode_label(indexes=pv_power.columns, label=pv_files.label)
                pv_power.columns = new_columns

                pv_power_all.append(pv_power)

            pv_power = pd.concat(pv_power_all, axis="columns")

        else:
            pv_power = get_pv_power_from_database(
                history_duration=self.history_duration,
                interpolate_minutes=self.live_interpolate_minutes,
                load_extra_minutes=self.live_load_extra_minutes,
            )
            logger.debug(f"Found {len(pv_power)} pv power datetimes from database ")
            logger.debug(f"Found {len(pv_power.columns)} pv power pv system ids from database")

        assert not pv_power.columns.duplicated().any()

        # A bit of hand-crafted cleaning
        bad_pvputput_indexes = [30248]
        bad_pvputput_indexes = encode_label(bad_pvputput_indexes, label="pvoutput")
        for bad_index in bad_pvputput_indexes:
            if bad_index in pv_power.columns:
                pv_power[bad_index]["2018-10-29":"2019-01-03"] = np.NaN

        # Drop columns and rows with all NaNs.
        pv_power.dropna(axis="columns", how="all", inplace=True)
        pv_power.dropna(axis="index", how="all", inplace=True)

        logger.debug(f"Found {len(pv_power)} pv power datetimes from database ")
        logger.debug(f"Found {len(pv_power.columns)} pv power pv system ids from database")

        # drop systems with over night power
        # TODO: Fix bug!
        # drop_pv_systems_which_produce_overnight expects normalised data,
        # but it doesn't receive normalised data. See Issue #664.
        pv_power = drop_pv_systems_which_produce_overnight(pv_power)

        logger.debug(
            f"Found {len(pv_power)} pv power datetimes " f"after filtering for over night power  "
        )
        logger.debug(
            f"Found {len(pv_power.columns)} pv power pv system ids "
            f"after filtering for over night power"
        )

        # Resample to 5-minutely and interpolate up to 15 minutes ahead.
        # TODO: Issue #301: Give users the option to NOT resample (because Perceiver IO
        # doesn't need all the data to be perfectly aligned).
        if len(pv_power) > 0:
            pv_power = pv_power.resample("5T").interpolate(method="time", limit=3)
            pv_power.dropna(axis="index", how="all", inplace=True)
        # self.pv_power = dd.from_pandas(pv_power, npartitions=3)
        print("pv_power = {:,.1f} MB".format(pv_power.values.nbytes / 1e6))
        self.pv_power = pv_power

        logger.debug(f"Found {len(self.pv_power)} pv power datetimes")
        logger.debug(f"Found {len(self.pv_power.columns)} pv power pv system ids")

        # get the max generation / capacity for each system
        if not self.is_live:
            self.pv_capacity = pv_power.max()
        else:
            self.pv_capacity = self.pv_metadata["installed_capacity_kw"]

    def _get_time_slice(self, t0_datetime_utc: pd.Timestamp) -> tuple[pd.DataFrame]:
        # TODO: Cache this?
        start_dt = self._get_start_dt(t0_datetime_utc)
        end_dt = self._get_end_dt(t0_datetime_utc)

        selected_pv_power = self.pv_power.loc[start_dt:end_dt].dropna(axis="columns", how="any")
        selected_pv_capacity = self.pv_capacity[selected_pv_power.columns]

        pv_power_zero_or_above_flag = selected_pv_power.ge(0).all()

        if pv_power_zero_or_above_flag.sum() != len(selected_pv_power.columns):
            n = len(selected_pv_power.columns) - pv_power_zero_or_above_flag.sum()
            logger.debug(f"Will be removing {n} pv systems as they have negative values")

        selected_pv_power = selected_pv_power.loc[:, pv_power_zero_or_above_flag]
        selected_pv_capacity = selected_pv_capacity.loc[pv_power_zero_or_above_flag]

        # is live, and no values, add a row of nans
        if (len(selected_pv_power) == 0) and self.is_live:
            selected_pv_power = selected_pv_power.append([pd.DataFrame(index=[t0_datetime_utc])])
            selected_pv_capacity = selected_pv_capacity.append([pd.Series(index=[t0_datetime_utc])])

        return selected_pv_power, selected_pv_capacity

    def _get_central_pv_system_id(
        self,
        x_center_osgb: Number,
        y_center_osgb: Number,
        pv_system_ids_with_data_for_timeslice: pd.Index,
    ) -> int:
        # If x_center_osgb and y_center_osgb have been chosen
        # by PVDataSource.pick_locations_for_batch() then we just have
        # to find the pv_system_ids at that exact location.  This is
        # super-fast (a few hundred microseconds).  We use np.isclose
        # instead of the equality operator because floats.
        pv_system_ids = self.pv_metadata.index[
            np.isclose(self.pv_metadata.location_x, x_center_osgb)
            & np.isclose(self.pv_metadata.location_y, y_center_osgb)
        ]
        if len(pv_system_ids) == 0:
            # TODO: Implement finding PV systems closest to x_center_osgb,
            # y_center_osgb.  This will probably be quite slow, so always
            # try finding an exact match first (which is super-fast).
            raise NotImplementedError(
                "Not yet implemented the ability to find PV systems *nearest*"
                " (but not at the identical location to) x_center_osgb and"
                " y_center_osgb."
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
        x_center_osgb: Number,
        y_center_osgb: Number,
        pv_system_ids_with_data_for_timeslice: pd.Index,
    ) -> pd.Index:
        """
        Find the PV system IDs. for all the PV systems within the geospatial

        This is for all the PV systems within the geospatial
        region of interest, defined by self.square.
        """
        logger.debug(f"Getting PV example data for {x_center_osgb} and {y_center_osgb}")

        bounding_box = self._rectangle.bounding_box_centered_on(
            x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb
        )
        x = self.pv_metadata.location_x
        y = self.pv_metadata.location_y

        # make mask of pv system ids
        mask = get_bounding_box_mask(bounding_box, x, y)
        pv_system_ids = self.pv_metadata.index[mask]
        x = self.pv_metadata.location_x[mask]
        y = self.pv_metadata.location_y[mask]

        # order the pv systems
        mask_order = get_closest_coordinate_order(
            x_center=x_center_osgb, y_center=y_center_osgb, x=x, y=y
        )
        mask_order = mask_order.sort_values()
        pv_system_ids = mask_order.index

        pv_system_ids = pv_system_ids_with_data_for_timeslice.intersection(pv_system_ids)

        # there may not be any pv systems in a GSP region
        # assert len(pv_system_ids) > 0

        return pv_system_ids

    def get_example(self, location: SpaceTimeLocation) -> xr.Dataset:
        """
        Get Example data for PV data

        Args:
            location: A location object of the example which contains
                - a timestamp of the example (t0_datetime_utc),
                - the x center location of the example (x_location_osgb)
                - the y center location of the example(y_location_osgb)

        Returns: Example data

        """
        logger.debug("Getting PV example data")

        t0_datetime_utc = location.t0_datetime_utc
        x_center_osgb = location.x_center_osgb
        y_center_osgb = location.y_center_osgb

        selected_pv_power, selected_pv_capacity = self._get_time_slice(t0_datetime_utc)
        all_pv_system_ids = self._get_all_pv_system_ids_in_roi(
            x_center_osgb, y_center_osgb, selected_pv_power.columns
        )
        if self.get_center:
            central_pv_system_id = self._get_central_pv_system_id(
                x_center_osgb, y_center_osgb, selected_pv_power.columns
            )

            # By convention, the 'target' PV system ID (the one in the center
            # of the image) must be in the first position of the returned arrays.
            all_pv_system_ids = all_pv_system_ids.drop(central_pv_system_id)
            all_pv_system_ids = all_pv_system_ids.insert(loc=0, item=central_pv_system_id)

        all_pv_system_ids = all_pv_system_ids[: self.n_pv_systems_per_example]

        selected_pv_power = selected_pv_power[all_pv_system_ids]
        selected_pv_capacity = selected_pv_capacity[all_pv_system_ids]

        # this provides an index of what pv systesm are in the examples
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
        pv["x_osgb"] = x_coords.astype("float32")
        pv["y_osgb"] = y_coords.astype("float32")
        pv["pv_system_row_number"] = pv_system_row_number

        # pad out so that there are always n_pv_systems_per_example, pad with zeros
        pad_n = self.n_pv_systems_per_example - len(pv.id)
        pv = pv.pad(id=(0, pad_n), power_mw=((0, 0), (0, pad_n)), constant_values=0)

        # format id
        pv.__setitem__("id", pv.id.astype("int32"))

        return pv

    def get_locations(self, t0_datetimes_utc: pd.DatetimeIndex) -> List[SpaceTimeLocation]:
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
        locations = []
        for t0_datetime in t0_datetimes_utc:
            pv_system_ids = _get_pv_system_ids(t0_datetime)
            pv_system_id = self.rng.choice(pv_system_ids)

            # Get metadata for PV system
            metadata_for_pv_system = self.pv_metadata.loc[pv_system_id]

            locations.append(
                SpaceTimeLocation(
                    t0_datetime_utc=t0_datetime,
                    x_center_osgb=metadata_for_pv_system.location_x,
                    y_center_osgb=metadata_for_pv_system.location_y,
                    id=pv_system_id,
                    id_type="pv_system",
                )
            )

        return locations

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

    with fsspec.open(filename, mode="rb") as file:
        pv_power = xr.open_dataset(file, engine="h5netcdf")
        pv_power = pv_power.sel(datetime=slice(start_dt, end_dt))
        pv_power_df = pv_power.to_dataframe()

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
    """Drop systems which produce power over night.

    Args:
        pv_power: Normalised to [0, 1].
    """

    if len(pv_power) == 0:
        return pv_power

    # TODO: Of these bad systems, 24647, 42656, 42807, 43081, 51247, 59919
    # might have some salvagable data?
    NIGHT_YIELD_THRESHOLD = 0.4
    night_hours = [22, 23, 0, 1, 2]
    pv_data_at_night = pv_power.loc[pv_power.index.hour.isin(night_hours)]
    pv_above_threshold_at_night = (pv_data_at_night > NIGHT_YIELD_THRESHOLD).any()
    bad_systems = pv_power.columns[pv_above_threshold_at_night]
    logger.debug(f"{len(bad_systems)} bad PV systems found and removed!")
    return pv_power.drop(columns=bad_systems)
