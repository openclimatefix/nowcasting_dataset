import logging

import xarray as xr

from typing import Union, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from numbers import Number
import torch
import numpy as np
import pandas as pd

from nowcasting_dataset.utils import scale_to_0_to_1, pad_data
from nowcasting_dataset.square import get_bounding_box_mask
from nowcasting_dataset.geospatial import lat_lon_to_osgb
from nowcasting_dataset.example import Example
from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso

from nowcasting_dataset.data_sources.constants import GSP_YIELD, GSP_SYSTEM_ID, GSP_SYSTEM_X_COORDS, \
    GSP_SYSTEM_Y_COORDS, DEFAULT_N_GSP_PER_EXAMPLE, CENTROID_TYPE


logger = logging.getLogger(__name__)


@dataclass
class GSPDataSource(ImageDataSource):
    """
    Data source for GSP PV Data

    30 mins data is taken from 'pvline' from https://www.solar.sheffield.ac.uk/pvlive/
    meta data is taken from ESO
    """

    filename: Union[str, Path]
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    threshold: int = 20
    minute_delta: int = 30
    get_centroid: bool = True
    n_gsp_systems_per_example: int = DEFAULT_N_GSP_PER_EXAMPLE

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """
        Set random seed and load data
        """
        super().__post_init__(image_size_pixels, meters_per_pixel)
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)
        self.load()

    def load(self):
        """
        Load the meta data and load the gsp power data
        """

        # load metadata
        self.metadata = get_gsp_metadata_from_eso()

        # make location x,y in osgb
        self.metadata["location_x"], self.metadata["location_y"] = lat_lon_to_osgb(
            self.metadata["gsp_lat"], self.metadata["gsp_lon"]
        )

        # load gsp data from file / gcp
        self.gsp_power = load_solar_gsp_data(self.filename, start_dt=self.start_dt, end_dt=self.end_dt)

        # drop any gsp below 20 MW (or set threshold)
        self.gsp_power, self.metadata = drop_gsp_system_by_threshold(
            self.gsp_power, self.metadata, threshold_mw=self.threshold
        )

        # scale from 0 to 1
        self.gsp_power = scale_to_0_to_1(self.gsp_power)

    def datetime_index(self):
        """
        Return the datetime that are available
        """
        return self.gsp_power.index

    def get_locations_for_batch(self, t0_datetimes: pd.DatetimeIndex) -> Tuple[List[Number], List[Number]]:
        """
        Get x and locations for a batch. Assume that all data is available for all GSP.
        Random GSP are taken, and the locations of them are returned. This is useful as other datasources need to know
        which x,y locations to get
        Returns: list of x and y locations
        """

        logger.debug("Getting locations for the batch")

        # Pick a random GSP system for each t0_datetime, and then grab
        # their geographical location.
        x_locations = []
        y_locations = []

        # assume that all gsp data is there for all timestamps
        for _ in t0_datetimes:

            # get random index
            random_index = self.rng.choice(self.gsp_power.columns)
            meta_data = self.metadata[(self.metadata["gsp_id"] == random_index)]

            # make sure there is only one
            metadata_for_gsp_system = meta_data.iloc[0]

            # Get metadata for GSP
            x_locations.append(metadata_for_gsp_system.location_x)
            y_locations.append(metadata_for_gsp_system.location_y)

            logger.debug(
                f"Found locations for gsp id {random_index} of {metadata_for_gsp_system.location_x} and"
                f"{metadata_for_gsp_system.location_y}"
            )

        return x_locations, y_locations

    def get_example(self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number) -> Example:
        """
        Get data example from one time point (t0_dt) and for x and y coords (x_meters_center), (y_meters_center).

        Get data at the location of x,y and get surrounding gsp power data also.

        Args:
            t0_dt: datetime of data. History and forecast are also returned
            x_meters_center: x location of centroid gsp.
            y_meters_center: y location of centroid gsp.

        Returns: Dictionary with GSP data in it

        """
        logger.debug("Getting example data")

        # get the power for the on time stamp, including history and forecaster
        selected_gsp_power = self._get_time_slice(t0_dt)

        # get the main gsp id, and the ids of the gsp in the bounding box
        all_gsp_system_ids = self._get_all_gsp_system_ids_in_roi(
            x_meters_center, y_meters_center, selected_gsp_power.columns
        )
        if self.get_centroid:
            central_gsp_system_id = self._get_central_gsp_system_id(
                x_meters_center, y_meters_center, selected_gsp_power.columns
            )
            assert central_gsp_system_id in all_gsp_system_ids

            # By convention, the 'target' GSP system ID (the one in the center
            # of the image) must be in the first position of the returned arrays.
            all_gsp_system_ids = all_gsp_system_ids.drop(central_gsp_system_id)
            all_gsp_system_ids = all_gsp_system_ids.insert(loc=0, item=central_gsp_system_id)

        # only select at most {n_gsp_systems_per_example} at most
        all_gsp_system_ids = all_gsp_system_ids[: self.n_gsp_systems_per_example]

        # select the GSP power output for the selected systems
        selected_gsp_power = selected_gsp_power[all_gsp_system_ids]

        gsp_system_x_coords = self.metadata[self.metadata["gsp_id"].isin(all_gsp_system_ids)].location_x
        gsp_system_y_coords = self.metadata[self.metadata["gsp_id"].isin(all_gsp_system_ids)].location_y

        # Save data into the Example dict...
        example = Example(
            gsp_system_id=all_gsp_system_ids,
            gsp_yield=selected_gsp_power,
            x_meters_center=x_meters_center,
            y_meters_center=y_meters_center,
            gsp_system_x_coords=gsp_system_x_coords,
            gsp_system_y_coords=gsp_system_y_coords,
            gsp_datetime_index=selected_gsp_power.index,
        )

        if self.get_centroid:
            example[CENTROID_TYPE] = 'gsp'

        # Pad (if necessary) so returned arrays are always of size n_gsp_systems_per_example.
        pad_size = self.n_gsp_systems_per_example - len(all_gsp_system_ids)
        example = pad_data(data=example,
                           one_dimensional_arrays=[GSP_SYSTEM_ID, GSP_SYSTEM_X_COORDS, GSP_SYSTEM_Y_COORDS],
                           two_dimensional_arrays=[GSP_YIELD],
                           pad_size=pad_size)

        return example

    def _get_central_gsp_system_id(
        self, x_meters_center: Number, y_meters_center: Number, gsp_system_ids_with_data_for_timeslice: pd.Int64Index
    ) -> int:
        """
        Get the system id of the central system from coordinates
        Args:
            x_meters_center: the location of the gsp (x)
            y_meters_center: the location of the gsp (y)
            gsp_system_ids_with_data_for_timeslice: List of gsp ids that are available for a certain timeslice

        Returns: gsp id
        """

        logger.debug("Getting Central gsp")

        # If x_meters_center and y_meters_center have been chosen
        # by {}.get_locations_for_batch() then we just have
        # to find the gsp_ids at that exact location.  This is
        # super-fast (a few hundred microseconds).  We use np.isclose
        # instead of the equality operator because floats.
        meta_data_index = self.metadata.index[
            np.isclose(self.metadata.location_x, x_meters_center, rtol=1E-05, atol=1E-05)
            & np.isclose(self.metadata.location_y, y_meters_center, rtol=1E-05, atol=1E-05)
        ]
        gsp_system_ids = self.metadata.loc[meta_data_index].gsp_id.values

        if len(gsp_system_ids) == 0:
            # TODO: Implement finding GSP systems closest to x_meters_center,
            # y_meters_center.  This will probably be quite slow, so always
            # try finding an exact match first (which is super-fast).
            raise NotImplementedError(
                "Not yet implemented the ability to find GSP systems *nearest*"
                " (but not at the identical location to) x_meters_center and"
                " y_meters_center."
            )

        gsp_system_ids = gsp_system_ids_with_data_for_timeslice.intersection(gsp_system_ids)

        if len(gsp_system_ids) == 0:
            raise NotImplementedError(
                f"Could not find gsp system id for {x_meters_center}, {y_meters_center} "
                f"({gsp_system_ids}) and {gsp_system_ids_with_data_for_timeslice}"
            )

        return int(gsp_system_ids[0])

    def _get_all_gsp_system_ids_in_roi(
        self, x_meters_center: Number, y_meters_center: Number, gsp_system_ids_with_data_for_timeslice: pd.Int64Index
    ) -> pd.Int64Index:
        """
        Find the GSP IDs for all the GSP within the geospatial region of interest, defined by self.square.
        Args:
            x_meters_center: centroid of area of interest (x coords)
            y_meters_center: centroid of area of interest (y coords)
            gsp_system_ids_with_data_for_timeslice: ids that are avialble for a specific time slice

        Returns: list of gsp ids that are in area of interest

        """

        logger.debug("Getting all gsp in ROI")

        # creating bounding box
        bounding_box = self._square.bounding_box_centered_on(
            x_meters_center=x_meters_center, y_meters_center=y_meters_center
        )

        # get all x and y locations of gsp
        x = self.metadata.location_x
        y = self.metadata.location_y

        # make mask of gsp_ids
        mask = get_bounding_box_mask(bounding_box, x, y)

        gsp_system_ids = self.metadata[mask].gsp_id
        gsp_system_ids = gsp_system_ids_with_data_for_timeslice.intersection(gsp_system_ids)
        assert len(gsp_system_ids) > 0
        return gsp_system_ids

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> [pd.DataFrame]:
        """
        Get time slice of gsp power data for give time.
        Note the time is extended backwards by history lenght and forward by prediction time
        Args:
            t0_dt: timestamp of interest

        Returns: pandas data frame of gsp power data
        """

        # get start and end datetime, takening into account history and forecast length.
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        # select power for certain times
        power = self.gsp_power.loc[start_dt:end_dt]

        # remove any nans
        power = power.dropna(axis="columns", how="any")

        return power


def drop_gsp_system_by_threshold(gsp_power: pd.DataFrame, meta_data: pd.DataFrame, threshold_mw: int = 20):
    """
    Drop gsp system where the max power is below a certain threshold
    Args:
        gsp_power: gsp power data
        meta_data: the gsp meta data
        threshold_mw: the threshold where we only taken gsp system with a maximum power, above this value.

    Returns: power data and metadata
    """
    maximum_gsp = gsp_power.max()

    keep_index = maximum_gsp >= threshold_mw

    logger.debug(f"Dropping {sum(~keep_index)} GSPs as maximum is not greater {threshold_mw} MW")
    logger.debug(f"Keeping {sum(keep_index)} GSPs as maximum is greater {threshold_mw} MW")

    gsp_power = gsp_power[keep_index.index]
    gsp_ids = gsp_power.columns
    meta_data = meta_data[meta_data["gsp_id"].isin(gsp_ids)]

    return gsp_power[keep_index.index], meta_data


def load_solar_gsp_data(
    filename: Union[str, Path], start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load solar pv gsp data from gcs (although there is an option to load from local - for testing)
    @param filename: filename of file to be loaded, can put 'gs://' files in here too
    @param start_dt: the start datetime, which to trim the data to
    @param end_dt: the end datetime, which to trim the data to
    @return: dataframe of pv data
    """
    logger.debug(f"Loading Solar GSP Data from GCS {filename} from {start_dt} to {end_dt}")
    # Open data - it maye be quicker to open byte file first, but decided just to keep it like this at the moment
    gsp_power = xr.open_zarr(filename)
    gsp_power = gsp_power.sel(datetime_gmt=slice(start_dt, end_dt))
    gsp_power_df = gsp_power.to_dataframe()

    # Save memory
    del gsp_power

    # Process the data a little
    gsp_power_df = gsp_power_df.dropna(axis="columns", how="all")
    gsp_power_df = gsp_power_df.clip(lower=0, upper=5e7)

    # make column names ints, not strings
    gsp_power_df.columns = [int(col) for col in gsp_power_df.columns]

    return gsp_power_df
