from nowcasting_dataset.consts import (
    PV_SYSTEM_ID,
    PV_SYSTEM_ROW_NUMBER,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_Y_COORDS,
    PV_AZIMUTH_ANGLE,
    PV_ELEVATION_ANGLE,
    PV_YIELD,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    OBJECT_AT_CENTER,
)
from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset import geospatial, utils
from nowcasting_dataset.square import get_bounding_box_mask
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from numbers import Number
from typing import List, Tuple, Union, Optional
import datetime
from pathlib import Path
import io
import gcsfs
import xarray as xr
import functools
import logging
import time
from concurrent import futures

logger = logging.getLogger(__name__)


@dataclass
class PVDataSource(ImageDataSource):
    filename: Union[str, Path]
    metadata_filename: Union[str, Path]
    start_dt: Optional[datetime.datetime] = None
    end_dt: Optional[datetime.datetime] = None
    random_pv_system_for_given_location: Optional[bool] = True
    #: Each example will always have this many PV systems.
    #: If less than this number exist in the data then pad with NaNs.
    n_pv_systems_per_example: int = DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE
    load_azimuth_and_elevation: bool = False
    load_from_gcs: bool = True  # option to load data from gcs, or local file
    get_center: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)
        self.load()

    def load(self):
        self._load_metadata()
        self._load_pv_power()
        if self.load_azimuth_and_elevation:
            self._calculate_azimuth_and_elevation()
        self.pv_metadata, self.pv_power = align_pv_system_ids(self.pv_metadata, self.pv_power)

    def _load_metadata(self):

        logger.debug("Loading Metadata")

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

        logger.debug("Loading PV Power data")

        if "gs://" not in str(self.filename):
            self.load_from_gcs = False

        pv_power = load_solar_pv_data_from_gcs(
            self.filename, start_dt=self.start_dt, end_dt=self.end_dt, from_gcs=self.load_from_gcs
        )

        # A bit of hand-crafted cleaning
        if 30248 in pv_power.columns:
            pv_power[30248]["2018-10-29":"2019-01-03"] = np.NaN

        # Drop columns and rows with all NaNs.
        pv_power.dropna(axis="columns", how="all", inplace=True)
        pv_power.dropna(axis="index", how="all", inplace=True)

        pv_power = utils.scale_to_0_to_1(pv_power)
        pv_power = drop_pv_systems_which_produce_overnight(pv_power)

        # Resample to 5-minutely and interpolate up to 15 minutes ahead.
        # TODO: Cubic interpolation?
        pv_power = pv_power.resample("5T").interpolate(method="time", limit=3)
        pv_power.dropna(axis="index", how="all", inplace=True)
        # self.pv_power = dd.from_pandas(pv_power, npartitions=3)
        print("pv_power = {:,.1f} MB".format(pv_power.values.nbytes / 1e6))
        self.pv_power = pv_power

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> [pd.DataFrame]:
        # TODO: Cache this?
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        del t0_dt  # t0 is not used in the rest of this method!
        selected_pv_power = self.pv_power.loc[start_dt:end_dt].dropna(axis="columns", how="any")

        if self.load_azimuth_and_elevation:
            selected_pv_azimuth_angle = self.pv_azimuth.loc[start_dt:end_dt].dropna(
                axis="columns", how="any"
            )
            selected_pv_elevation_angle = self.pv_elevation.loc[start_dt:end_dt].dropna(
                axis="columns", how="any"
            )
        else:
            selected_pv_azimuth_angle = None
            selected_pv_elevation_angle = None

        return selected_pv_power, selected_pv_azimuth_angle, selected_pv_elevation_angle

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
        """Find the PV system IDs for all the PV systems within the geospatial
        region of interest, defined by self.square."""

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
    ) -> Example:

        logger.debug("Getting PV example data")

        (
            selected_pv_power,
            selected_pv_azimuth_angle,
            selected_pv_elevation_angle,
        ) = self._get_time_slice(t0_dt)
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
        if self.load_azimuth_and_elevation:
            selected_pv_azimuth_angle = selected_pv_azimuth_angle[all_pv_system_ids]
            selected_pv_elevation_angle = selected_pv_elevation_angle[all_pv_system_ids]

        pv_system_row_number = np.flatnonzero(self.pv_metadata.index.isin(all_pv_system_ids))
        pv_system_x_coords = self.pv_metadata.location_x[all_pv_system_ids]
        pv_system_y_coords = self.pv_metadata.location_y[all_pv_system_ids]
        # Save data into the Example dict...
        example = Example(
            t0_dt=t0_dt,
            pv_system_id=all_pv_system_ids,
            pv_system_row_number=pv_system_row_number,
            pv_yield=selected_pv_power,
            x_meters_center=x_meters_center,
            y_meters_center=y_meters_center,
            pv_system_x_coords=pv_system_x_coords,
            pv_system_y_coords=pv_system_y_coords,
            pv_datetime_index=selected_pv_power.index,
        )

        if self.load_azimuth_and_elevation:
            example[PV_AZIMUTH_ANGLE] = selected_pv_azimuth_angle
            example[PV_ELEVATION_ANGLE] = selected_pv_elevation_angle

        if self.get_center:
            example[OBJECT_AT_CENTER] = "pv"

        # Pad (if necessary) so returned arrays are always of size n_pv_systems_per_example.
        pad_size = self.n_pv_systems_per_example - len(all_pv_system_ids)

        one_dimensional_arrays = [
            PV_SYSTEM_ID,
            PV_SYSTEM_ROW_NUMBER,
            PV_SYSTEM_X_COORDS,
            PV_SYSTEM_Y_COORDS,
        ]

        pad_nans_variables = [PV_YIELD]
        if self.load_azimuth_and_elevation:
            pad_nans_variables.append(PV_AZIMUTH_ANGLE)
            pad_nans_variables.append(PV_ELEVATION_ANGLE)

        example = utils.pad_data(
            data=example,
            one_dimensional_arrays=one_dimensional_arrays,
            two_dimensional_arrays=pad_nans_variables,
            pad_size=pad_size,
        )

        return example

    def get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
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

    def _calculate_azimuth_and_elevation(self):
        """
        Calculate the azimuth and elevation angles for each datestamp, for each pv system.
        """

        logger.debug("Calculating azimuth and elevation angles")

        self.pv_azimuth, self.pv_elevation = calculate_azimuth_and_elevation_all_pv_systems(
            self.datetime_index().to_pydatetime(), self.pv_metadata
        )


def calculate_azimuth_and_elevation_all_pv_systems(
    datestamps: List[datetime.datetime], pv_metadata: pd.DataFrame
) -> (pd.Series, pd.Series):
    """
    Calculate the azimuth and elevation angles for each datestamp, for each pv system.
    """

    logger.debug(
        f"Will be calculating for {len(datestamps)} datestamps and {len(pv_metadata)} pv systems"
    )

    # create array of index datetime, columns of system_id for both azimuth and elevation
    pv_azimuth = []
    pv_elevation = []

    t = time.time()
    # loop over all metadata and fine azimuth and elevation angles,
    # not sure this is the best method to use, as currently this step takes ~2 minute for 745 pv systems,
    # and 235 datestamps (~100,000 point). But this only needs to be done once.
    with futures.ThreadPoolExecutor(max_workers=len(pv_metadata)) as executor:

        logger.debug("Setting up jobs")

        # Submit tasks to the executor.
        future_azimuth_and_elevation_per_pv_system = []
        for i in tqdm(range(len(pv_metadata))):
            future_azimuth_and_elevation = executor.submit(
                geospatial.calculate_azimuth_and_elevation_angle,
                latitude=pv_metadata.iloc[i].latitude,
                longitude=pv_metadata.iloc[i].longitude,
                datestamps=datestamps,
            )
            future_azimuth_and_elevation_per_pv_system.append(
                [future_azimuth_and_elevation, pv_metadata.iloc[i].name]
            )

        logger.debug(f"Getting results")

        # Collect results from each thread.
        for i in tqdm(range(len(future_azimuth_and_elevation_per_pv_system))):
            future_azimuth_and_elevation, name = future_azimuth_and_elevation_per_pv_system[i]
            azimuth_and_elevation = future_azimuth_and_elevation.result()

            azimuth = azimuth_and_elevation.loc[:, "azimuth"].rename(name)
            elevation = azimuth_and_elevation.loc[:, "elevation"].rename(name)

            pv_azimuth.append(azimuth)
            pv_elevation.append(elevation)

    pv_azimuth = pd.concat(pv_azimuth, axis=1)
    pv_elevation = pd.concat(pv_elevation, axis=1)

    logger.debug(f"Calculated Azimuth and Elevation angles in {time.time() - t} seconds")

    return pv_azimuth, pv_elevation


def load_solar_pv_data_from_gcs(
    filename: Union[str, Path],
    start_dt: Optional[datetime.datetime] = None,
    end_dt: Optional[datetime.datetime] = None,
    from_gcs: bool = True,
) -> pd.DataFrame:
    """
    Load solar pv data from gcs (althought there is an option to load from loca - for testing)
    @param filename: filename of file to be loaded
    @param start_dt: the start datetime, which to trim the data to
    @param end_dt: the end datetime, which to trim the data to
    @param from_gcs: option to laod from gcs, or form local file
    @return: dataframe of pv data
    """
    gcs = gcsfs.GCSFileSystem(access="read_only")

    logger.debug("Loading Solar PV Data from GCS")

    # It is possible to simplify the code below and do
    # xr.open_dataset(file, engine='h5netcdf')
    # in the first 'with' block, and delete the second 'with' block.
    # But that takes 1 minute to load the data, where as loading into memory
    # first and then loading from memory takes 23 seconds!
    if from_gcs:
        with gcs.open(filename, mode="rb") as file:
            file_bytes = file.read()
    else:
        with open(filename, mode="rb") as file:
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

    return pv_power_df.tz_localize("Europe/London").tz_convert("UTC").tz_convert(None)


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
