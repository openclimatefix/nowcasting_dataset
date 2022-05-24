""" GSP Data Source. GSP - Grid Supply Points

Read more https://data.nationalgrideso.com/system/gis-boundaries-for-gb-grid-supply-points
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist

import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset.consts import DEFAULT_N_GSP_PER_EXAMPLE
from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.gsp.live import get_gsp_power_from_database
from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation
from nowcasting_dataset.geospatial import lat_lon_to_osgb
from nowcasting_dataset.square import get_bounding_box_mask

logger = logging.getLogger(__name__)


@dataclass
class GSPDataSource(ImageDataSource):
    """
    Data source for GSP (Grid Supply Point) PV Data.

    30 mins data is taken from 'PV Live' from https://www.solar.sheffield.ac.uk/pvlive/
    meta data is taken from ESO.  PV Live estimates the total PV power generation for each
    Grid Supply Point region.

    Even though GSP data isn't image data, `GSPDataSource` inherits from `ImageDataSource`
    so it can select Grid Supply Point regions within the geospatial region of interest.
    The region of interest is defined by `image_size_pixels` and `meters_per_pixel`.
    """

    # zarr_path of where the gsp data is stored
    zarr_path: Union[str, Path]
    # start datetime, this can be None
    start_datetime: Optional[datetime] = None
    # end datetime, this can be None
    end_datetime: Optional[datetime] = None
    # the threshold where we only taken gsp's with a maximum power, above this value.
    threshold_mw: int = 0
    # get the data for the gsp at the center too.
    # This can be turned off if the center of the bounding box is of a pv system
    get_center: bool = True
    # the maximum number of gsp's to be loaded for data sample
    n_gsp_per_example: int = DEFAULT_N_GSP_PER_EXAMPLE
    # scale from zero to one
    do_scale_0_to_1: bool = False
    # Drop any GSPs norther of this boundary.
    # Many satellite images have a "rectangle of zeros" extending north from 1,037,047 meters
    # (OSGB "northing" coordinate).  The largest satellite regions of interest are
    # 144 km.  So set the boundary to be 1,037 km - (144 km / 2) = 1,036,975 meters.
    # This results in 2 GSPs being dropped.
    # See https://github.com/openclimatefix/Satip/issues/30
    northern_boundary_osgb: Optional[float] = 1_036_975
    # Only load metadata
    metadata_only: bool = False
    is_live: bool = False
    live_interpolate_minutes: int = 60
    live_load_extra_minutes: int = 60

    def __post_init__(
        self, image_size_pixels_height: int, image_size_pixels_width: int, meters_per_pixel: int
    ):
        """
        Set random seed and load data
        """
        super().__post_init__(image_size_pixels_height, image_size_pixels_width, meters_per_pixel)
        self.rng = np.random.default_rng()
        self.load()

    def check_input_paths_exist(self) -> None:
        """Check input paths exist.  If not, raise a FileNotFoundError."""
        if not self.is_live:
            nd_fs_utils.check_path_exists(self.zarr_path)

    @property
    def sample_period_minutes(self) -> int:
        """Override the default sample minutes"""
        return 30

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
        return GSP

    def load(self):
        """
        Load the meta data and load the GSP power data
        """
        # load metadata
        self.metadata = get_gsp_metadata_from_eso()
        self.metadata.set_index("gsp_id", drop=False, inplace=True)

        # make location x,y in osgb
        self.metadata["location_x"], self.metadata["location_y"] = lat_lon_to_osgb(
            lat=self.metadata["centroid_lat"], lon=self.metadata["centroid_lon"]
        )

        if not self.metadata_only:
            # load gsp data from file / gcp
            if self.is_live:
                self.gsp_power, self.gsp_capacity = get_gsp_power_from_database(
                    history_duration=self.history_duration,
                    interpolate_minutes=self.live_interpolate_minutes,
                    load_extra_minutes=self.live_load_extra_minutes,
                )
            else:
                self.gsp_power, self.gsp_capacity = load_solar_gsp_data(
                    self.zarr_path, start_dt=self.start_datetime, end_dt=self.end_datetime
                )

            # drop any gsp below a threshold mw. This is to get rid of any small GSP where
            # predicting the solar output will be harder.
            self.gsp_power, self.metadata = drop_gsp_by_threshold(
                self.gsp_power, self.metadata, threshold_mw=self.threshold_mw
            )

            if self.northern_boundary_osgb is not None:
                self.gsp_power, self.metadata = drop_gsp_north_of_boundary(
                    self.gsp_power,
                    self.metadata,
                    northern_boundary_osgb=self.northern_boundary_osgb,
                )

            logger.debug(f"There are {len(self.gsp_power.columns)} GSP")

    def datetime_index(self):
        """
        Return the datetimes that are available
        """
        return self.gsp_power.index

    def get_number_locations(self):
        """Get the number of GSP"""

        return len(self.metadata.location_x)

    def get_all_locations(self, t0_datetimes_utc: pd.DatetimeIndex) -> List[SpaceTimeLocation]:
        """
        Make locations for all GSP

        For some datetimes, return locations of all datetimes and all GSPs.
        This means a national forecast can then be made

        Args:
            t0_datetimes_utc: list of available t0 datetimes.

        Returns:
            List of space time locations which includes
            1. datetimes
            2. x locations
            3. y locations
            4. gsp ids

        """

        logger.info("Getting all locations for each datetime")
        total_gsp_nan_count = 0
        if not self.metadata_only:
            total_gsp_nan_count = self.gsp_power.isna().sum().sum()

        if total_gsp_nan_count > 0:
            assert Exception("There are nans in the GSP data. Can't get locations for all GSPs")
        else:

            t0_datetimes_utc.name = "t0_datetime_utc"

            # get all locations
            x_centers_osgb = self.metadata.location_x
            y_centers_osgb = self.metadata.location_y
            gsp_ids = self.metadata.index

            # make x centers
            x_centers_osgb_all_gsps = pd.DataFrame(columns=t0_datetimes_utc, index=x_centers_osgb)
            x_centers_osgb_all_gsps = x_centers_osgb_all_gsps.unstack().reset_index()

            # make y centers
            y_centers_osgb_all_gsps = pd.DataFrame(columns=t0_datetimes_utc, index=y_centers_osgb)
            y_centers_osgb_all_gsps = y_centers_osgb_all_gsps.unstack().reset_index()

            # make gsp ids
            gsp_ids = pd.DataFrame(columns=t0_datetimes_utc, index=gsp_ids)
            gsp_ids = gsp_ids.unstack().reset_index()

            t0_datetimes_utc_all_gsps = pd.DatetimeIndex(x_centers_osgb_all_gsps["t0_datetime_utc"])
            x_centers_osgb_all_gsps = list(x_centers_osgb_all_gsps["location_x"])
            y_centers_osgb_all_gsps = list(y_centers_osgb_all_gsps["location_y"])
            gsp_ids = list(gsp_ids["gsp_id"])

            assert len(x_centers_osgb_all_gsps) == len(y_centers_osgb_all_gsps)
            assert len(x_centers_osgb_all_gsps) == len(
                gsp_ids
            ), f"{len(x_centers_osgb_all_gsps)=} {len(gsp_ids)=}"
            assert len(y_centers_osgb_all_gsps) == len(gsp_ids)

            locations = []
            # TODO make dataframe -> List[dict] -> List[Locations]
            for i in range(len(t0_datetimes_utc_all_gsps)):
                locations.append(
                    SpaceTimeLocation(
                        t0_datetime_utc=t0_datetimes_utc_all_gsps[i],
                        x_center_osgb=x_centers_osgb_all_gsps[i],
                        y_center_osgb=y_centers_osgb_all_gsps[i],
                        id=gsp_ids[i],
                    )
                )

            return locations

    def get_locations(self, t0_datetimes_utc: pd.DatetimeIndex) -> List[SpaceTimeLocation]:
        """
        Get x and y locations. Assume that all data is available for all GSP.

        Random GSP are taken, and the locations of them are returned. This is useful as other
        datasources need to know which x,y locations to get.

        Args:
            t0_datetimes_utc: list of available t0 datetimes.

        Returns: list of location objects

        """

        logger.info(f"Getting one location for each datetime {self.gsp_power}")

        total_gsp_nan_count = self.gsp_power.isna().sum().sum()
        if total_gsp_nan_count == 0:

            # get random GSP metadata
            indexes = sorted(
                list(self.rng.integers(low=0, high=len(self.metadata), size=len(t0_datetimes_utc)))
            )
            metadata = self.metadata.iloc[indexes]

            # get x, y locations
            x_centers_osgb = list(metadata.location_x)
            y_centers_osgb = list(metadata.location_y)
            ids = list(metadata.index)

        else:

            logger.warning(
                "There are some nans in the gsp data, "
                "so to get x,y locations we have to do a big loop"
            )

            # Pick a random GSP for each t0_datetime, and then grab
            # their geographical location.
            x_centers_osgb = []
            y_centers_osgb = []
            ids = []

            for t0_dt in t0_datetimes_utc:

                # Choose start and end times
                start_dt = self._get_start_dt(t0_dt)
                end_dt = self._get_end_dt(t0_dt)

                # remove any nans
                gsp_power = self.gsp_power.loc[start_dt:end_dt].dropna(axis="columns", how="any")

                # get random index
                random_gsp_id = self.rng.choice(gsp_power.columns)
                meta_data = self.metadata[(self.metadata["gsp_id"] == random_gsp_id)]

                # Make sure there is only one GSP.
                # Sometimes there are multiple gsp_ids at one location e.g. 'SELL_1'.
                # TODO: Issue #272: Further investigation on multiple GSPs may be needed.
                metadata_for_gsp = meta_data.iloc[0]

                # Get metadata for GSP
                x_centers_osgb.append(metadata_for_gsp.location_x)
                y_centers_osgb.append(metadata_for_gsp.location_y)
                ids.append(meta_data.index[0])

        assert len(x_centers_osgb) == len(y_centers_osgb)
        assert len(x_centers_osgb) == len(ids)
        assert len(y_centers_osgb) == len(ids)

        locations = []
        for i in range(len(x_centers_osgb)):

            locations.append(
                SpaceTimeLocation(
                    t0_datetime_utc=t0_datetimes_utc[i],
                    x_center_osgb=x_centers_osgb[i],
                    y_center_osgb=y_centers_osgb[i],
                    id=ids[i],
                    id_type="gsp",
                )
            )

        return locations

    def get_example(self, location: SpaceTimeLocation) -> xr.Dataset:
        """
        Get data example from one time point (t0_dt) and for x and y coords.

        Get data at the location of x,y and get surrounding GSP power data also.

        Args:
            location: A location object of the example which contains
                - a timestamp of the example (t0_datetime_utc),
                - the x center location of the example (x_location_osgb)
                - the y center location of the example(y_location_osgb)

        Returns: Dictionary with GSP data in it.
        """
        logger.debug("Getting example data")

        t0_datetime_utc = location.t0_datetime_utc
        x_center_osgb = location.x_center_osgb
        y_center_osgb = location.y_center_osgb

        # get the GSP power, including history and forecast
        selected_gsp_power, selected_capacity = self._get_time_slice(t0_datetime_utc)

        # get the main gsp id, and the ids of the gsp in the bounding box
        all_gsp_ids = self._get_gsp_ids_in_roi(
            x_center_osgb, y_center_osgb, selected_gsp_power.columns
        )
        if self.get_center:
            central_gsp_id = self._get_central_gsp_id(
                x_center_osgb, y_center_osgb, selected_gsp_power.columns
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
        selected_capacity = selected_capacity[all_gsp_ids]

        # get x,y coordinates
        gsp_x_coords = self.metadata.location_x[all_gsp_ids]
        gsp_y_coords = self.metadata.location_y[all_gsp_ids]

        # convert to data array
        da = xr.DataArray(
            data=selected_gsp_power.values,
            dims=["time", "id"],
            coords=dict(
                id=all_gsp_ids.values.astype(int),
                time=selected_gsp_power.index.values,
            ),
        )

        capacity = xr.DataArray(
            data=selected_capacity.values,
            dims=["time", "id"],
            coords=dict(
                id=all_gsp_ids.values.astype(int),
                time=selected_gsp_power.index.values,
            ),
        )

        # convert to dataset
        gsp = da.to_dataset(name="power_mw")
        gsp["capacity_mwp"] = capacity

        # add gsp x coords
        gsp_x_coords = xr.DataArray(
            data=gsp_x_coords.values,
            dims=["id"],
        )

        gsp_y_coords = xr.DataArray(
            data=gsp_y_coords.values,
            dims=["id"],
        )
        gsp["x_osgb"] = gsp_x_coords
        gsp["y_osgb"] = gsp_y_coords

        # pad out so that there are always 32 gsp, fill with 0
        pad_n = self.n_gsp_per_example - len(gsp.id)
        gsp = gsp.pad(id=(0, pad_n), power_mw=((0, 0), (0, pad_n)), constant_values=0)

        return gsp

    def _get_central_gsp_id(
        self,
        x_center_osgb: Number,
        y_center_osgb: Number,
        gsp_ids_with_data_for_timeslice: pd.Index,
    ) -> int:
        """
        Get the GSP id of the central GSP from coordinates

        Args:
            x_center_osgb: the location of the gsp (x)
            y_center_osgb: the location of the gsp (y)
            gsp_ids_with_data_for_timeslice: List of gsp ids that are available for a certain
                timeslice.

        Returns: GSP id
        """
        logger.debug("Getting Central GSP")

        # If x_center_osgb and y_center_osgb have been chosen
        # by {}.get_locations() then we just have
        # to find the gsp_ids at that exact location.  This is
        # super-fast (a few hundred microseconds).  We use np.isclose
        # instead of the equality operator because floats.
        meta_data_index = self.metadata.index[
            np.isclose(self.metadata.location_x, x_center_osgb, rtol=1e-05, atol=1e-05)
            & np.isclose(self.metadata.location_y, y_center_osgb, rtol=1e-05, atol=1e-05)
        ]
        gsp_ids = self.metadata.loc[meta_data_index].gsp_id.values

        if len(gsp_ids) == 0:
            # TODO: Implement finding GSP closest to x_center_osgb,
            # y_center_osgb.  This will probably be quite slow, so always
            # try finding an exact match first (which is super-fast).

            mat = cdist(
                self.metadata[["location_x", "location_y"]],
                [[x_center_osgb, y_center_osgb]],
                metric="euclidean",
            )
            closest_gsp = np.argmin(mat[:, 0])
            # This is the closest GSP ID then
            gsp_ids = [self.metadata.iloc[closest_gsp].gsp_id]

        gsp_ids = gsp_ids_with_data_for_timeslice.intersection(gsp_ids)

        if len(gsp_ids) == 0:
            raise NotImplementedError(
                f"Could not find GSP id for {x_center_osgb}, {y_center_osgb} "
                f"({gsp_ids}) and {gsp_ids_with_data_for_timeslice}"
            )

        return int(gsp_ids[0])

    def _get_gsp_ids_in_roi(
        self,
        x_center_osgb: Number,
        y_center_osgb: Number,
        gsp_ids_with_data_for_timeslice: pd.Index,
    ) -> pd.Index:
        """
        Find the GSP IDs for all the GSP within the geospatial region of interest.

        The geospatial region of interest is defined by self.square.

        Args:
            x_center_osgb: center of area of interest (x coords)
            y_center_osgb: center of area of interest (y coords)
            gsp_ids_with_data_for_timeslice: ids that are avialble for a specific time slice

        Returns: list of GSP ids that are in area of interest
        """
        logger.debug(f"Getting all gsp in ROI ({x_center_osgb=},{y_center_osgb=})")

        # creating bounding box
        bounding_box = self._rectangle.bounding_box_centered_on(
            x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb
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

    def _get_time_slice(self, t0_datetime_utc: pd.Timestamp) -> [pd.DataFrame]:
        """
        Get time slice of GSP power data for give time.

        Note the time is extended backwards by history lenght and forward by prediction time

        Args:
            t0_datetime_utc: timestamp of interest

        Returns: pandas data frame of GSP power data
        """
        logger.debug(f"Getting power slice for {t0_datetime_utc}")

        # get start and end datetime, takening into account history and forecast length.
        start_dt = self._get_start_dt(t0_datetime_utc)
        end_dt = self._get_end_dt(t0_datetime_utc)

        # need to floor by 30 mins.
        # If t0 is 12.45 and history duration is 1 hours, then start_dt will be 11.45.
        # But we need to collect data at 11.30, 12.00, and 12.30
        start_dt = pd.to_datetime(start_dt).floor("30T")

        # select power and capacity for certain times
        power = self.gsp_power.loc[start_dt:end_dt]
        capacity = self.gsp_capacity.loc[start_dt:end_dt]

        # remove any nans
        power = power.dropna(axis="columns", how="any")
        capacity = capacity.dropna(axis="columns", how="any")

        logger.debug(f"Found {len(power.columns)} GSP valid data for {t0_datetime_utc}")

        return power, capacity


def drop_gsp_by_threshold(
    gsp_power: pd.DataFrame, meta_data: pd.DataFrame, threshold_mw: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop GSP where the max power is below a certain threshold

    Args:
        gsp_power: GSP power data
        meta_data: the GSP meta data
        threshold_mw: the threshold where we only taken GSP with a maximum power, above this value.

    Returns: power data and metadata
    """
    if len(gsp_power) == 0:
        return gsp_power, meta_data

    maximum_gsp = gsp_power.max()

    keep_index = maximum_gsp > threshold_mw

    dropped_gsp_names = meta_data.loc[keep_index[~keep_index].index].gsp_name

    logger.debug(
        f"Dropping {sum(~keep_index)} GSPs as maximum is not greater {threshold_mw} MW."
        f" Dropped GSP IDs and GSP names:\n{dropped_gsp_names}"
    )
    logger.debug(f"Keeping {sum(keep_index)} GSPs as maximum is greater {threshold_mw} MW")

    filtered_gsp_power = gsp_power.loc[:, keep_index]
    filtered_gsp_ids = filtered_gsp_power.columns
    filtered_meta_data = meta_data[meta_data["gsp_id"].isin(filtered_gsp_ids)]

    assert set(filtered_gsp_power.columns) == set(filtered_meta_data.gsp_id)
    return filtered_gsp_power, filtered_meta_data


def drop_gsp_north_of_boundary(
    gsp_power: pd.DataFrame, meta_data: pd.DataFrame, northern_boundary_osgb: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop GSPs north of northern_boundary_osgb.

    Args:
        gsp_power: GSP power data
        meta_data: the GSP meta data
        northern_boundary_osgb: The geospatial boundary.

    Returns: power data and metadata
    """

    if len(gsp_power) == 0:
        return gsp_power, meta_data

    keep_index = meta_data.location_y < northern_boundary_osgb
    filtered_meta_data = meta_data.loc[keep_index]
    filtered_gsp_ids = filtered_meta_data.gsp_id
    filtered_gsp_power = gsp_power[filtered_gsp_ids]
    pv_capacity_of_dropped_gsps = gsp_power.loc[:, ~keep_index].max()

    logger.debug(
        f"Dropping {sum(~keep_index)} GSPs because they are north of"
        f" y={northern_boundary_osgb:,d}m:\n"
        f" {meta_data.loc[~keep_index].gsp_name}\n with capacity in MW of:\n"
        f"{pv_capacity_of_dropped_gsps}"
    )
    logger.debug(
        f"Keeping {sum(keep_index)} GSPs because they are south of y={northern_boundary_osgb:,d}m."
    )

    assert set(filtered_gsp_power.columns) == set(filtered_meta_data.gsp_id)
    return filtered_gsp_power, filtered_meta_data


def load_solar_gsp_data(
    zarr_path: Union[str, Path],
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Load solar PV GSP data

    Args:
        zarr_path:  zarr_path of file to be loaded, can put 'gs://' files in here too
        start_dt: the start datetime, which to trim the data to
        end_dt: the end datetime, which to trim the data to

    Returns: dataframe of gsp data

    """
    logger.debug(f"Loading Solar GSP Data from {zarr_path} from {start_dt} to {end_dt}")
    # Open data - it may be quicker to open byte file first, but decided just to keep it
    # like this at the moment.
    gsp_power_and_capacity = xr.open_dataset(zarr_path, engine="zarr")
    gsp_power_and_capacity = gsp_power_and_capacity.sel(datetime_gmt=slice(start_dt, end_dt))

    # make dataframe with index datetime_gmt and columns og gsp_id
    gsp_power_and_capacity_df = gsp_power_and_capacity.to_dataframe()
    gsp_power_and_capacity_df.reset_index(inplace=True)
    gsp_power_df = gsp_power_and_capacity_df.pivot(
        index="datetime_gmt", columns="gsp_id", values="generation_mw"
    )

    gsp_capacity_df = gsp_power_and_capacity_df.pivot(
        index="datetime_gmt", columns="gsp_id", values="installedcapacity_mwp"
    )

    # Save memory
    del gsp_power_and_capacity

    # Process the data a little
    gsp_power_df = gsp_power_df.dropna(axis="columns", how="all")
    gsp_power_df = gsp_power_df.clip(lower=0, upper=5e7)
    gsp_capacity_df = gsp_capacity_df.dropna(axis="columns", how="all")
    gsp_capacity_df = gsp_capacity_df.clip(lower=0, upper=5e7)

    # remove whole rows of nans
    gsp_power_df = gsp_power_df.dropna(axis="columns", how="all")
    gsp_capacity_df = gsp_capacity_df.loc[:, gsp_power_df.columns]

    # make column names ints, not strings
    gsp_power_df.columns = [int(col) for col in gsp_power_df.columns]
    gsp_capacity_df.columns = [int(col) for col in gsp_capacity_df.columns]

    return gsp_power_df, gsp_capacity_df
