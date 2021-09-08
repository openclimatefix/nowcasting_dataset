import logging

import pandas as pd
import xarray as xr

from typing import Union, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from numbers import Number
import torch
import numpy as np
import pandas as pd

from nowcasting_dataset.geospatial import lat_lon_to_osgb
from nowcasting_dataset.example import Example
from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.data_sources.gsp.eso import get_pv_gsp_metadata_from_eso


logger = logging.getLogger(__name__)


@dataclass
class GSPPVDataSource(ImageDataSource):
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

    n_gsp_systems_per_example: int = 32

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)
        self.load()

    def load(self):
        """
        Load the meta data and load he gsp pv power data
        """

        # load metadata
        self.metadata = get_pv_gsp_metadata_from_eso()

        # make location x,y in osgb
        self.metadata["location_x"], self.metadata["location_y"] = lat_lon_to_osgb(
            self.metadata["gsp_lat"], self.metadata["gsp_lon"]
        )

        self.gsp_pv_power = load_solar_pv_gsp_data(self.filename, start_dt=self.start_dt, end_dt=self.end_dt)

        self.gsp_pv_power, self.metadata = drop_gcp_system_by_threshold(
            self.gsp_pv_power, self.metadata, threshold=self.threshold
        )

    def datetime_index(self):
        return self.gsp_pv_power.index

    def get_locations_for_batch(self, t0_datetimes: pd.DatetimeIndex) -> Tuple[List[Number], List[Number]]:
        """

        Returns:

        """

        logger.debug("Getting locations for the batch")

        # Pick a random GSP system for each t0_datetime, and then grab
        # their geographical location.
        x_locations = []
        y_locations = []

        # assume that all gsp data is there for all timestamps
        for _ in t0_datetimes:

            # get random index
            random_index = self.rng.choice(self.gsp_pv_power.columns)
            meta_data = self.metadata[(self.metadata["gsp_id"] == random_index)]

            # make sure there is only one
            metadata_for_pv_system = meta_data.iloc[0]

            # Get metadata for PV system
            x_locations.append(metadata_for_pv_system.location_x)
            y_locations.append(metadata_for_pv_system.location_y)

            logger.debug(
                f"Found locations for gsp id {random_index} of {metadata_for_pv_system.location_x} and"
                f"{metadata_for_pv_system.location_y}"
            )

        return x_locations, y_locations

    def get_example(self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number) -> Example:

        logger.debug("Getting example data")

        # get the power for the on time stamp, including history and forecaster
        selected_gsp_power = self._get_time_slice(t0_dt)

        # get the main gsp id, and the ids of the gsp in the bounding box
        central_gsp_system_id = self._get_central_gsp_system_id(
            x_meters_center, y_meters_center, selected_gsp_power.columns
        )
        all_gsp_system_ids = self._get_all_gsp_system_ids_in_roi(
            x_meters_center, y_meters_center, selected_gsp_power.columns
        )
        assert central_gsp_system_id in all_gsp_system_ids

        # By convention, the 'target' PV system ID (the one in the center
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

        return example

    def _get_central_gsp_system_id(
        self, x_meters_center: Number, y_meters_center: Number, gcp_pv_system_ids_with_data_for_timeslice: pd.Int64Index
    ) -> int:
        """

        Args:
            x_meters_center:
            y_meters_center:
            gcp_pv_system_ids_with_data_for_timeslice:

        Returns:

        """

        logger.debug("Getting Central gcp")

        # If x_meters_center and y_meters_center have been chosen
        # by PVDataSource.pick_locations_for_batch() then we just have
        # to find the pv_system_ids at that exact location.  This is
        # super-fast (a few hundred microseconds).  We use np.isclose
        # instead of the equality operator because floats.
        meta_data_index = self.metadata.index[
            np.isclose(self.metadata.location_x, x_meters_center)
            & np.isclose(self.metadata.location_y, y_meters_center)
        ]
        gcp_system_ids = self.metadata.loc[meta_data_index].gsp_id.values

        if len(gcp_system_ids) == 0:
            # TODO: Implement finding PV systems closest to x_meters_center,
            # y_meters_center.  This will probably be quite slow, so always
            # try finding an exact match first (which is super-fast).
            raise NotImplementedError(
                "Not yet implemented the ability to find GCP PV systems *nearest*"
                " (but not at the identical location to) x_meters_center and"
                " y_meters_center."
            )

        gcp_system_ids = gcp_pv_system_ids_with_data_for_timeslice.intersection(gcp_system_ids)

        if len(gcp_system_ids) == 0:
            raise NotImplementedError(
                f"Could not find gcp system id for {x_meters_center}, {y_meters_center} "
                f"({gcp_system_ids}) and {gcp_pv_system_ids_with_data_for_timeslice}"
            )

        return int(gcp_system_ids[0])

    def _get_all_gsp_system_ids_in_roi(
        self, x_meters_center: Number, y_meters_center: Number, gsp_system_ids_with_data_for_timeslice: pd.Int64Index
    ) -> pd.Int64Index:
        """

        Args:
            x_meters_center:
            y_meters_center:
            gsp_system_ids_with_data_for_timeslice:

        Returns:

        """
        """Find the PV system IDs for all the PV systems within the geospatial
        region of interest, defined by self.square."""

        logger.debug("Getting all gcp in ROI")

        # creating bounding box
        bounding_box = self._square.bounding_box_centered_on(
            x_meters_center=x_meters_center, y_meters_center=y_meters_center
        )

        # get all x and y locations of gsp
        x = self.metadata.location_x
        y = self.metadata.location_y

        mask = (
            (x >= bounding_box.left) & (x <= bounding_box.right) & (y >= bounding_box.bottom) & (y <= bounding_box.top)
        )

        gsp_system_ids = self.metadata[mask].gsp_id
        gsp_system_ids = gsp_system_ids_with_data_for_timeslice.intersection(gsp_system_ids)
        assert len(gsp_system_ids) > 0
        return gsp_system_ids

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> [pd.DataFrame]:
        """

        Args:
            t0_dt:

        Returns:

        """

        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        power = self.gsp_pv_power.loc[start_dt:end_dt]
        logger.debug(power)

        power = power.dropna(axis="columns", how="any")
        logger.debug(power)

        return power


def drop_gcp_system_by_threshold(gsp_pv_power: pd.DataFrame, meta_data: pd.DataFrame, threshold: int = 20):

    maximum_gsp_pv = gsp_pv_power.max()

    keep_index = maximum_gsp_pv >= threshold

    logger.debug(f"Dropping {sum(~keep_index)} GCPs as maximum is not greater {threshold} MW")
    logger.debug(f"Keeping {sum(keep_index)} GCPs as maximum is greater {threshold} MW")

    gsp_pv_power = gsp_pv_power[keep_index.index]
    gsp_ids = gsp_pv_power.columns
    meta_data = meta_data[meta_data["gsp_id"].isin(gsp_ids)]

    return gsp_pv_power[keep_index.index], meta_data


def load_solar_pv_gsp_data(
    filename: Union[str, Path], start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load solar pv gsp data from gcs (although there is an option to load from local - for testing)
    @param filename: filename of file to be loaded, can put 'gs://' files in here too
    @param start_dt: the start datetime, which to trim the data to
    @param end_dt: the end datetime, which to trim the data to
    @return: dataframe of pv data
    """
    logger.debug("Loading Solar PV GCP Data from GCS")
    # Open data - it maye be quicker to open byte file first, but decided just to keep it like this at the moment
    pv_power = xr.open_zarr(filename)

    pv_power = pv_power.sel(datetime_gmt=slice(start_dt, end_dt))
    pv_power_df = pv_power.to_dataframe()

    # Save memory
    del pv_power

    # Process the data a little
    pv_power_df = pv_power_df.dropna(axis="columns", how="all")
    pv_power_df = pv_power_df.clip(lower=0, upper=5e7)

    # make column names ints, not strings
    pv_power_df.columns = [int(col) for col in pv_power_df.columns]

    return pv_power_df
