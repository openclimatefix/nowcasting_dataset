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
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso

from nowcasting_dataset.consts import (
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    DEFAULT_N_GSP_PER_EXAMPLE,
    OBJECT_AT_CENTER,
)

logger = logging.getLogger(__name__)


@dataclass
class GSPDataSource(ImageDataSource):
    """
    Data source for GSP PV Data

    30 mins data is taken from 'PV Live' from https://www.solar.sheffield.ac.uk/pvlive/
    meta data is taken from ESO
    """

    # filename of where the gsp data is stored
    filename: Union[str, Path]
    # start datetime, this can be None
    start_dt: Optional[datetime] = None
    # end datetime, this can be None
    end_dt: Optional[datetime] = None
    # the threshold where we only taken gsp's with a maximum power, above this value.
    threshold_mw: int = 20
    # the frequency of the data
    sample_period_minutes: int = 30
    # get the data for the gsp at the center too.
    # This can be turned off if the center of the bounding box is of a pv system
    get_center: bool = True
    # the maximum number of gsp's to be loaded for data sample
    n_gsp_per_example: int = DEFAULT_N_GSP_PER_EXAMPLE

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """
        Set random seed and load data
        """
        super().__post_init__(image_size_pixels, meters_per_pixel)
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)
        self.load()

    def _get_sample_period_minutes(self):
        """Override the default sample minutes"""
        return self.sample_period_minutes

    def load(self):
        """
        Load the meta data and load the GSP power data
        """

        # load metadata
        self.metadata = get_gsp_metadata_from_eso()

        # make location x,y in osgb
        self.metadata["location_x"], self.metadata["location_y"] = lat_lon_to_osgb(
            lat=self.metadata["centroid_y"], lon=self.metadata["centroid_x"]
        )

        # load gsp data from file / gcp
        self.gsp_power = load_solar_gsp_data(
            self.filename, start_dt=self.start_dt, end_dt=self.end_dt
        )

        # drop any gsp below 20 MW (or set threshold). This is to get rid of any small GSP where predicting the
        # solar output will be harder.
        self.gsp_power, self.metadata = drop_gsp_by_threshold(
            self.gsp_power, self.metadata, threshold_mw=self.threshold_mw
        )

        # scale from 0 to 1
        self.gsp_power = scale_to_0_to_1(self.gsp_power)

        logger.debug(f"There are {len(self.gsp_power.columns)} GSP")

    def datetime_index(self):
        """
        Return the datetimes that are available
        """
        return self.gsp_power.index

    def get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        """
        Get x and y locations for a batch. Assume that all data is available for all GSP.
        Random GSP are taken, and the locations of them are returned. This is useful as other datasources need to know
        which x,y locations to get
        Returns: list of x and y locations
        """

        logger.debug("Getting locations for the batch")

        # Pick a random GSP for each t0_datetime, and then grab
        # their geographical location.
        x_locations = []
        y_locations = []

        for t0_dt in t0_datetimes:

            # Choose start and end times
            start_dt = self._get_start_dt(t0_dt)
            end_dt = self._get_end_dt(t0_dt)

            # remove any nans
            gsp_power = self.gsp_power.loc[start_dt:end_dt].dropna(axis="columns", how="any")

            # get random index
            random_gsp_id = self.rng.choice(gsp_power.columns)
            meta_data = self.metadata[(self.metadata["gsp_id"] == random_gsp_id)]

            # Make sure there is only one. Sometimes there are multiple gsp_ids at one location e.g. 'SELL_1'.
            # Further investigation on this may be needed, but going to ignore this for now.
            #
            metadata_for_gsp = meta_data.iloc[0]

            # Get metadata for GSP
            x_locations.append(metadata_for_gsp.location_x)
            y_locations.append(metadata_for_gsp.location_y)

            logger.debug(
                f"Found locations for GSP id {random_gsp_id} of {metadata_for_gsp.location_x} and "
                f"{metadata_for_gsp.location_y}"
            )

        return x_locations, y_locations

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Example:
        """
        Get data example from one time point (t0_dt) and for x and y coords (x_meters_center), (y_meters_center).

        Get data at the location of x,y and get surrounding GSP power data also.

        Args:
            t0_dt: datetime of "now". History and forecast are also returned
            x_meters_center: x location of center GSP.
            y_meters_center: y location of center GSP.

        Returns: Dictionary with GSP data in it

        """
        logger.debug("Getting example data")

        # get the GSP power, including history and forecast
        selected_gsp_power = self._get_time_slice(t0_dt)

        # get the main gsp id, and the ids of the gsp in the bounding box
        all_gsp_ids = self._get_gsp_ids_in_roi(
            x_meters_center, y_meters_center, selected_gsp_power.columns
        )
        if self.get_center:
            central_gsp_id = self._get_central_gsp_id(
                x_meters_center, y_meters_center, selected_gsp_power.columns
            )
            assert central_gsp_id in all_gsp_ids

            # By convention, the 'target' GSP ID (the one in the center
            # of the image) must be in the first position of the returned arrays.
            all_gsp_ids = all_gsp_ids.drop(central_gsp_id)
            all_gsp_ids = all_gsp_ids.insert(loc=0, item=central_gsp_id)
        else:
            logger.warning("Not getting center GSP")

        # only select at most {n_gsp_per_example}
        all_gsp_ids = all_gsp_ids[: self.n_gsp_per_example]

        # select the GSP power output for the selected GSP IDs
        selected_gsp_power = selected_gsp_power[all_gsp_ids]

        gsp_x_coords = self.metadata[self.metadata["gsp_id"].isin(all_gsp_ids)].location_x
        gsp_y_coords = self.metadata[self.metadata["gsp_id"].isin(all_gsp_ids)].location_y

        # Save data into the Example dict...
        example = Example(
            t0_dt=t0_dt,
            gsp_id=all_gsp_ids,
            gsp_yield=selected_gsp_power,
            x_meters_center=x_meters_center,
            y_meters_center=y_meters_center,
            gsp_x_coords=gsp_x_coords,
            gsp_y_coords=gsp_y_coords,
            gsp_datetime_index=selected_gsp_power.index,
        )

        if self.get_center:
            example[OBJECT_AT_CENTER] = "gsp"

        # Pad (if necessary) so returned arrays are always of size n_gsp_per_example.
        pad_size = self.n_gsp_per_example - len(all_gsp_ids)
        example = pad_data(
            data=example,
            one_dimensional_arrays=[GSP_ID, GSP_X_COORDS, GSP_Y_COORDS],
            two_dimensional_arrays=[GSP_YIELD],
            pad_size=pad_size,
        )

        return example

    def _get_central_gsp_id(
        self,
        x_meters_center: Number,
        y_meters_center: Number,
        gsp_ids_with_data_for_timeslice: pd.Int64Index,
    ) -> int:
        """
        Get the GSP id of the central GSP from coordinates
        Args:
            x_meters_center: the location of the gsp (x)
            y_meters_center: the location of the gsp (y)
            gsp_ids_with_data_for_timeslice: List of gsp ids that are available for a certain timeslice

        Returns: GSP id
        """

        logger.debug("Getting Central GSP")

        # If x_meters_center and y_meters_center have been chosen
        # by {}.get_locations_for_batch() then we just have
        # to find the gsp_ids at that exact location.  This is
        # super-fast (a few hundred microseconds).  We use np.isclose
        # instead of the equality operator because floats.
        meta_data_index = self.metadata.index[
            np.isclose(self.metadata.location_x, x_meters_center, rtol=1e-05, atol=1e-05)
            & np.isclose(self.metadata.location_y, y_meters_center, rtol=1e-05, atol=1e-05)
        ]
        gsp_ids = self.metadata.loc[meta_data_index].gsp_id.values

        if len(gsp_ids) == 0:
            # TODO: Implement finding GSP closest to x_meters_center,
            # y_meters_center.  This will probably be quite slow, so always
            # try finding an exact match first (which is super-fast).
            raise NotImplementedError(
                "Not yet implemented the ability to find GSP *nearest*"
                " (but not at the identical location to) x_meters_center and"
                " y_meters_center."
            )

        gsp_ids = gsp_ids_with_data_for_timeslice.intersection(gsp_ids)

        if len(gsp_ids) == 0:
            raise NotImplementedError(
                f"Could not find GSP id for {x_meters_center}, {y_meters_center} "
                f"({gsp_ids}) and {gsp_ids_with_data_for_timeslice}"
            )

        return int(gsp_ids[0])

    def _get_gsp_ids_in_roi(
        self,
        x_meters_center: Number,
        y_meters_center: Number,
        gsp_ids_with_data_for_timeslice: pd.Int64Index,
    ) -> pd.Int64Index:
        """
        Find the GSP IDs for all the GSP within the geospatial region of interest, defined by self.square.
        Args:
            x_meters_center: center of area of interest (x coords)
            y_meters_center: center of area of interest (y coords)
            gsp_ids_with_data_for_timeslice: ids that are avialble for a specific time slice

        Returns: list of GSP ids that are in area of interest

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

        gsp_ids = self.metadata[mask].gsp_id
        gsp_ids = gsp_ids_with_data_for_timeslice.intersection(gsp_ids)

        assert len(gsp_ids) > 0
        return gsp_ids

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> [pd.DataFrame]:
        """
        Get time slice of GSP power data for give time.
        Note the time is extended backwards by history lenght and forward by prediction time
        Args:
            t0_dt: timestamp of interest

        Returns: pandas data frame of GSP power data
        """

        logger.debug(f"Getting power slice for {t0_dt}")

        # get start and end datetime, takening into account history and forecast length.
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        # select power for certain times
        power = self.gsp_power.loc[start_dt:end_dt]

        # remove any nans
        power = power.dropna(axis="columns", how="any")

        logger.debug(f"Found {len(power.columns)} GSP")

        return power


def drop_gsp_by_threshold(gsp_power: pd.DataFrame, meta_data: pd.DataFrame, threshold_mw: int = 20):
    """
    Drop GSP where the max power is below a certain threshold
    Args:
        gsp_power: GSP power data
        meta_data: the GSP meta data
        threshold_mw: the threshold where we only taken GSP with a maximum power, above this value.

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
    filename: Union[str, Path],
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load solar PV GSP data

    Args:
        filename:  filename of file to be loaded, can put 'gs://' files in here too
        start_dt: the start datetime, which to trim the data to
        end_dt: the end datetime, which to trim the data to

    Returns:dataframe of pv data

    """

    logger.debug(f"Loading Solar GSP Data from GCS {filename} from {start_dt} to {end_dt}")
    # Open data - it may be quicker to open byte file first, but decided just to keep it like this at the moment
    gsp_power = xr.open_dataset(filename, engine="zarr")
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
