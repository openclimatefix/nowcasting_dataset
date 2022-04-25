""" NWP Data Source """
import io
import logging
from dataclasses import InitVar, dataclass
from typing import Iterable, Optional

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset import utils
from nowcasting_dataset.consts import NWP_VARIABLE_NAMES
from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.data_sources.nwp.nwp_model import NWP

_LOG = logging.getLogger(__name__)


@dataclass
class NWPDataSource(ZarrDataSource):
    """
    NWP Data Source (Numerical Weather Predictions)

    Attributes:
        _data: xr.DataArray of Numerical Weather Predictions, opened by open().
            x is left-to-right.
            y is top-to-bottom (after reversing the `y` index in open_nwp()).
        consolidated: Whether or not the Zarr store is consolidated.
        channels: The NWP forecast parameters to load. If None then don't filter.
            See:  http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf
            All of these params are "instant" (i.e. a snapshot at the target time,
            not accumulated over some time period).  The available params are:
                cdcb  : Height of lowest cloud base > 3 oktas, in meters above surface.
                lcc   : Low-level cloud cover in %.
                mcc   : Medium-level cloud cover in %.
                hcc   : High-level cloud cover in %.
                sde   : Snow depth in meters.
                hcct  : Height of convective cloud top, meters above surface.
                        WARNING: hcct has NaNs where there are no clouds forecast to exist!
                dswrf : Downward short-wave radiation flux in W/m^2 (irradiance) at surface.
                dlwrf : Downward long-wave radiation flux in W/m^2 (irradiance) at surface.
                h     : Geometrical height, meters.
                t     : Air temperature at 1 meter above surface in Kelvin.
                r     : Relative humidty in %.
                dpt   : Dew point temperature in Kelvin.
                vis   : Visibility in meters.
                si10  : Wind speed in meters per second, 10 meters above surface.
                wdir10: Wind direction in degrees, 10 meters above surface.
                prmsl : Pressure reduce to mean sea level in Pascals.
                prate : Precipitation rate at the surface in kg/m^2/s.
    """

    channels: Optional[Iterable[str]] = NWP_VARIABLE_NAMES
    image_size_pixels_height: InitVar[int] = 2
    image_size_pixels_width: InitVar[int] = 2
    meters_per_pixel: InitVar[int] = 2_000

    def __post_init__(
        self, image_size_pixels_height: int, image_size_pixels_width: int, meters_per_pixel: int
    ):
        """
        Post init

        Args:
            image_size_pixels_height: number of pixels in image height
            image_size_pixels_width: number of pixels in image width
            meters_per_pixel: how many meteres for each pixel

        """
        super().__post_init__(image_size_pixels_height, image_size_pixels_width, meters_per_pixel)
        n_channels = len(self.channels)
        self._shape_of_example = (
            n_channels,
            self.total_seq_length,
            image_size_pixels_height,
            image_size_pixels_width,
        )
        _LOG.info(
            f"NWPDataSource instantiated with {self.channels=},"
            f" {self._rectangle.size_pixels_height=},"
            f" {self._rectangle.size_pixels_width=},"
            f" {self.zarr_path=}"
        )

    def open(self) -> None:
        """
        Open NWP data

        We don't want to open_nwp() in __init__.
        If we did that, then we couldn't copy NWPDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        self._data = self._open_data()
        np.testing.assert_array_equal(self._data["variable"].values, self.channels)
        self._data = self._data.rename({"x": "x_osgb", "y": "y_osgb"})

    def _open_data(self) -> xr.DataArray:
        data = open_nwp(self.zarr_path, consolidated=self.consolidated)
        return data.sel(variable=list(self.channels))

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
        return NWP

    def _get_time_slice(self, t0_datetime_utc: pd.Timestamp) -> xr.DataArray:
        """
        Select the numerical weather predictions for a single time slice.

        Note that this function does *not* resample from hourly to 5 minutely.
        Resampling would be very expensive if done on the whole geographical
        extent of the NWP data!

        Args:
            t0_datetime_utc: the time slice is around t0_dt.

        Returns: Slice of data

        """
        start_dt = self._get_start_dt(t0_datetime_utc)
        end_dt = self._get_end_dt(t0_datetime_utc)

        start_hourly = start_dt.floor("H")
        end_hourly = end_dt.ceil("H")

        # TODO: Issue #398: Use NWP init time closest to t0.
        init_time_i = np.searchsorted(self.data.init_time, start_hourly.to_numpy(), side="right")
        init_time_i -= 1  # Because searchsorted() gives the index to the entry _after_.
        init_time = self.data.init_time.values[init_time_i]

        step_start = start_hourly - init_time
        step_end = end_hourly - init_time

        selected = self.data.sel(init_time=init_time, step=slice(step_start, step_end))
        selected = selected.swap_dims({"step": "target_time"})
        selected["target_time"] = init_time + selected.step
        return selected

    def _post_process_example(self, selected_data: xr.Dataset, t0_dt: pd.Timestamp) -> xr.Dataset:
        """Resamples to 5 minutely."""

        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        # if t0_dt is not on the hour, e.g. 13.05.
        # Then if the history_minutes is 1 hours,
        # so start_dt will be 12.05, but we want to the 12.00 time step too
        start_dt = start_dt.floor("H")

        selected_data = selected_data.sel(target_time=slice(start_dt, end_dt))
        selected_data = selected_data.rename({"target_time": "time", "variable": "channels"})
        selected_data.data = selected_data.data.astype(np.float32)

        return selected_data

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes"""
        if self._data is None:
            nwp = self._open_data()
        else:
            nwp = self._data

        # We need to return the `target_times` (the times the NWPs are _about_).
        # The `target_time` is the `init_time` plus the forecast horizon `step`.
        # `step` is an array of timedeltas, so we can just add `init_time` to `step`.
        target_times = nwp["init_time"] + nwp["step"]
        target_times = target_times.values.flatten()
        target_times = np.unique(target_times)
        target_times = np.sort(target_times)
        target_times = pd.DatetimeIndex(target_times)

        return target_times

    @property
    def sample_period_minutes(self) -> int:
        """Override the default sample minutes"""
        return 60


def open_nwp(zarr_path: str, consolidated: bool) -> xr.DataArray:
    """
    Open The NWP data

    Args:
        zarr_path: zarr_path must start with 'gs://' if it's on GCP.
        consolidated: Is the Zarr metadata consolidated?

    Returns: NWP data.
    """
    _LOG.debug("Opening NWP data: %s", zarr_path)
    utils.set_fsspec_for_multiprocess()

    if zarr_path.split(".")[-1] == "netcdf":
        # loading netcdf file, download bytes and then load as xarray
        with fsspec.open(zarr_path, mode="rb") as file:
            file_bytes = file.read()

        with io.BytesIO(file_bytes) as file:
            nwp = xr.load_dataset(file, engine="h5netcdf")

    else:
        nwp = xr.open_dataset(
            zarr_path,
            engine="zarr",
            consolidated=consolidated,
            mode="r",
            chunks=None,  # Reading Satellite Zarr benefits from setting chunks='auto'
            # (see issue #456) but 'auto' massively slows down reading NWPs.
        )

    # Select the "UKV" DataArray from the "nwp" Dataset.
    # "UKV" is the one and only DataArray in the Zarr Dataset.
    # "UKV" stands for "United Kingdom Variable", and it the UK Met Office's high-res deterministic
    # NWP for the UK.  All the NWP variables are represented in the `variable` dimension within
    # the UKV DataArray.
    ukv = nwp["UKV"]

    # Reverse `y` so it's top-to-bottom (so ZarrDataSource.get_example() works correctly!)
    # if necessary.  Adapted from:
    # https://stackoverflow.com/questions/54677161/xarray-reverse-an-array-along-one-coordinate
    if ukv.y[0] < ukv.y[1]:
        _LOG.warning(
            "NWP y axis runs from bottom-to-top.  Will reverse y axis so it runs top-to-bottom."
        )
        y_reversed = ukv.y[::-1]
        ukv = ukv.reindex(y=y_reversed)

    # Sanity checks.
    ukv = utils.drop_duplicate_times(data_array=ukv, class_name="NWP", time_dim="init_time")

    # If any init_times are not monotonic_increasing then drop the out-of-order init_times:
    ukv = utils.drop_non_monotonic_increasing(
        data_array=ukv, class_name="NWP", time_dim="init_time"
    )

    assert pd.DatetimeIndex(ukv["init_time"]).is_unique
    assert pd.DatetimeIndex(ukv["init_time"]).is_monotonic_increasing

    return ukv
