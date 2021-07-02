from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.example import Example
from nowcasting_dataset import geospatial, utils
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from numbers import Number
from typing import List, Tuple, Union, Optional
import datetime
from pathlib import Path
import io
import gcsfs
import xarray as xr
import functools


@dataclass
class PVDataSource(DataSource):
    metadata_filename: Union[str, Path]
    start_dt: Optional[datetime.datetime] = None
    end_dt: Optional[datetime.datetime] = None

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)
        self.load()

    def load(self):
        self._load_metadata()
        self._load_pv_power()
        self.pv_metadata, self.pv_power = align_pv_system_ids(
            self.pv_metadata, self.pv_power)

    def _load_metadata(self):
        pv_metadata = pd.read_csv(
            self.metadata_filename, index_col='system_id')
        pv_metadata.dropna(
            subset=['longitude', 'latitude'], how='any', inplace=True)

        pv_metadata['location_x'], pv_metadata['location_y'] = (
            geospatial.lat_lon_to_osgb(
                pv_metadata['latitude'], pv_metadata['longitude']))

        # Remove PV systems outside the geospatial boundary of the
        # satellite data:
        GEO_BOUNDARY_OSGB = {
            'WEST': -238_000, 'EAST': 856_000,
            'NORTH': 1_222_000, 'SOUTH': -184_000}
        self.pv_metadata = pv_metadata[
            (pv_metadata.location_x >= GEO_BOUNDARY_OSGB['WEST']) &
            (pv_metadata.location_x <= GEO_BOUNDARY_OSGB['EAST']) &
            (pv_metadata.location_y <= GEO_BOUNDARY_OSGB['NORTH']) &
            (pv_metadata.location_y >= GEO_BOUNDARY_OSGB['SOUTH'])]

    def _load_pv_power(self):
        pv_power = load_solar_pv_data_from_gcs(
            self.filename, start_dt=self.start_dt, end_dt=self.end_dt)

        # A bit of hand-crafted cleaning
        pv_power[30248]['2018-10-29':'2019-01-03'] = np.NaN

        # Drop columns and rows with all NaNs.
        pv_power.dropna(axis='columns', how='all', inplace=True)
        pv_power.dropna(axis='index', how='all', inplace=True)

        pv_power = utils.scale_to_0_to_1(pv_power)
        pv_power = drop_pv_systems_which_produce_overnight(pv_power)

        # Resample to 5-minutely and interpolate up to 15 minutes ahead.
        # TODO: Cubic interpolation?
        pv_power = pv_power.resample('5T').interpolate(method='time', limit=3)
        pv_power.dropna(axis='index', how='all', inplace=True)
        # self.pv_power = dd.from_pandas(pv_power, npartitions=3)
        print('pv_power = {:,.1f} MB'.format(pv_power.values.nbytes / 1E6))
        self.pv_power = pv_power
        
    def _get_timestep(self, t0_dt: pd.Timestamp) -> pd.DataFrame:
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        del t0_dt  # t0 is not used in the rest of this method!
        selected_pv_power = self.pv_power.loc[start_dt:end_dt]
        return selected_pv_power.dropna(axis='columns', how='any')

    def get_sample(
            self,
            x_meters_center: Number,
            y_meters_center: Number,
            t0_dt: pd.Timestamp) -> Example:

        # If x_meters_center and y_meters_center have been chosen
        # by PVDataSource.pick_locations_for_batch() then we just have
        # to find the pv_system_ids at that exact location.  This is
        # super-fast (a few hundred microseconds).  We use np.isclose
        # instead of the equality operator because floats.
        pv_system_ids = self.pv_metadata.index[
            np.isclose(self.pv_metadata.location_x, x_meters_center) &
            np.isclose(self.pv_metadata.location_y, y_meters_center)]
        if len(pv_system_ids) == 0:
            # TODO: Implement finding PV systems closest to x_meters_center,
            # y_meters_center.  This will probably be quite slow, so always
            # try finding an exact match first (which is super-fast).
            raise NotImplementedError(
                "Not yet implemented the ability to find PV systems *nearest*"
                " (but not at the identical location to) x_meters_center and"
                " y_meters_center.")

        selected_pv_power = self._get_timestep_with_cache(t0_dt)
        pv_system_ids = selected_pv_power.columns.intersection(pv_system_ids)

        # Select just one PV system (the locations in PVOutput.org are quite
        # approximate, so it's quite common to have multiple PV systems
        # at the same nominal lat lon.
        pv_system_id = self.rng.choice(pv_system_ids)
        selected_pv_power = selected_pv_power[pv_system_id]

        # Save data into the Example dict...
        return Example(
            pv_system_id=pv_system_id,
            #pv_system_row_number=self.pv_metadata.index.get_loc(pv_system_id),
            pv_yield=selected_pv_power)

    def pick_locations_for_batch(
            self,
            t0_datetimes: pd.DatetimeIndex) -> Tuple[List[Number], List[Number]]:
        """Find a valid geographical location for each t0_datetime.

        Returns:  x_locations, y_locations. Each has one entry per t0_datetime.
            Locations are in OSGB coordinates.
        """
        
        # Set this up as a separate function, so we can cache the result!
        @functools.cache
        def _get_pv_system_ids(t0_datetime: pd.Timestamp) -> pd.Int64Index:
            start_dt = self._get_start_dt(t0_datetime)
            end_dt = self._get_end_dt(t0_datetime)
            available_pv_data = self.pv_power.loc[start_dt:end_dt]
            columns_mask = available_pv_data.notna().all().values
            pv_system_ids = available_pv_data.columns[columns_mask]
            assert len(pv_system_ids) > 0
            return pv_system_ids
        
        # Pick a random PV system for each t0_datetime, and then grab their geographical location.
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


def load_solar_pv_data_from_gcs(
        filename: Union[str, Path],
        start_dt: Optional[datetime.datetime] = None,
        end_dt: Optional[datetime.datetime] = None) -> pd.DataFrame:
    gcs = gcsfs.GCSFileSystem(access='read_only')

    # It is possible to simplify the code below and do
    # xr.open_dataset(file, engine='h5netcdf')
    # in the first 'with' block, and delete the second 'with' block.
    # But that takes 1 minute to load the data, where as loading into memory
    # first and then loading from memory takes 23 seconds!
    with gcs.open(filename, mode='rb') as file:
        file_bytes = file.read()

    with io.BytesIO(file_bytes) as file:
        pv_power = xr.open_dataset(file, engine='h5netcdf')
        pv_power = pv_power.sel(datetime=slice(start_dt, end_dt))
        pv_power_df = pv_power.to_dataframe()

    # Save memory
    del file_bytes
    del pv_power

    # Process the data a little
    pv_power_df = pv_power_df.dropna(axis='columns', how='all')
    pv_power_df = pv_power_df.clip(lower=0, upper=5E7)
    # Convert the pv_system_id column names from strings to ints:
    pv_power_df.columns = [np.int32(col) for col in pv_power_df.columns]

    return (
        pv_power_df
        .tz_localize('Europe/London')
        .tz_convert('UTC')
        .tz_convert(None))


def align_pv_system_ids(
        pv_metadata: pd.DataFrame,
        pv_power: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Only pick PV systems for which we have metadata."""
    pv_system_ids = pv_metadata.index.intersection(pv_power.columns)
    pv_system_ids = np.sort(pv_system_ids)
    pv_power = pv_power[pv_system_ids]
    pv_metadata = pv_metadata.loc[pv_system_ids]
    return pv_metadata, pv_power


def drop_pv_systems_which_produce_overnight(
        pv_power: pd.DataFrame) -> pd.DataFrame:
    """Drop systems which produce power over night."""
    # TODO: Of these bad systems, 24647, 42656, 42807, 43081, 51247, 59919
    # might have some salvagable data?
    NIGHT_YIELD_THRESHOLD = 0.4
    night_hours = [22, 23, 0, 1, 2]
    pv_data_at_night = pv_power.loc[pv_power.index.hour.isin(night_hours)]
    pv_above_threshold_at_night = (
        pv_data_at_night > NIGHT_YIELD_THRESHOLD).any()
    bad_systems = pv_power.columns[pv_above_threshold_at_night]
    print(len(bad_systems), 'bad PV systems found and removed!')
    return pv_power.drop(columns=bad_systems)
