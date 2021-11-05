""" NWP Data Source """
import logging
from dataclasses import InitVar, dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset import utils
from nowcasting_dataset.consts import NWP_VARIABLE_NAMES
from nowcasting_dataset.data_sources.data_source import ZarrDataSource

_LOG = logging.getLogger(__name__)


@dataclass
class NWPDataSource(ZarrDataSource):
    """
    NWP Data Source (Numerical Weather Predictions)

    Attributes:
      _data: xr.DataArray of Numerical Weather Predictions, opened by open().
        x is left-to-right.
        y is top-to-bottom.
        Access using public nwp property.
      consolidated: Whether or not the Zarr store is consolidated.
      channels: The NWP forecast parameters to load. If None then don't filter.
        The available params are:
            t     : Temperature in Kelvin.
            dswrf : Downward short-wave radiation flux in W/m^2 (irradiance).
            prate : Precipitation rate in kg/m^2/s.
            r     : Relative humidty in %.
            sde   : Snow depth in meters.
            si10  : 10-meter wind speed in m/s.
            vis   : Visibility in meters.
            lcc   : Low-level cloud cover in %.
            mcc   : Medium-level cloud cover in %.
            hcc   : High-level cloud cover in %.
    """

    channels: Optional[Iterable[str]] = NWP_VARIABLE_NAMES
    image_size_pixels: InitVar[int] = 2
    meters_per_pixel: InitVar[int] = 2_000

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """
        Post init

        Args:
            image_size_pixels: number of pixels in image
            meters_per_pixel: how many meteres for each pixel

        """
        super().__post_init__(image_size_pixels, meters_per_pixel)
        n_channels = len(self.channels)
        self._shape_of_example = (
            n_channels,
            self._total_seq_length,
            image_size_pixels,
            image_size_pixels,
        )

    def open(self) -> None:
        """
        Open NWP data

        We don't want to open_sat_data in __init__.
        If we did that, then we couldn't copy NWPDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        data = self._open_data()
        self._data = data["UKV"].sel(variable=list(self.channels))

    def _open_data(self) -> xr.DataArray:
        return open_nwp(self.zarr_path, consolidated=self.consolidated)

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        """
        Select the numerical weather predictions for a single time slice.

        Note that this function does *not* resample from hourly to 5 minutely.
        Resampling would be very expensive if done on the whole geographical
        extent of the NWP data!

        Args:
            t0_dt: the time slice is around t0_dt.

        Returns: Slice of data

        """
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        start_hourly = start_dt.floor("H")
        end_hourly = end_dt.ceil("H")

        init_time_i = np.searchsorted(self.data.init_time, start_hourly.to_numpy(), side="right")
        init_time_i -= 1  # Because searchsorted() gives the index to the entry _after_.
        init_time = self.data.init_time.values[init_time_i]

        step_start = start_hourly - init_time
        step_end = end_hourly - init_time

        selected = self.data.sel(init_time=init_time, step=slice(step_start, step_end))
        selected = selected.swap_dims({"step": "target_time"})
        selected["target_time"] = init_time + selected.step
        return selected

    def _post_process_example(
        self, selected_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:
        """Resamples to 5 minutely."""
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        selected_data = selected_data.sel(target_time=slice(start_dt, end_dt))
        selected_data = selected_data.rename({"target_time": "time", "variable": "channels"})
        selected_data.data = selected_data.data.astype(np.float16)

        return selected_data

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes"""
        if self._data is None:
            nwp = self._open_data()
        else:
            nwp = self._data
        target_times = nwp["init_time"] + nwp["step"][:3]
        target_times = target_times.values.flatten()
        target_times = np.unique(target_times)
        target_times = np.sort(target_times)
        target_times = pd.DatetimeIndex(target_times)
        return target_times

    @property
    def sample_period_minutes(self) -> int:
        """Override the default sample minutes"""
        return 60


def open_nwp(zarr_path: str, consolidated: bool) -> xr.Dataset:
    """
    Open The NWP data

    Args:
        zarr_path: zarr_path must start with 'gs://' if it's on GCP.
        consolidated: consolidate the zarr file?

    Returns: nwp data

    """
    _LOG.debug("Opening NWP data: %s", zarr_path)
    utils.set_fsspec_for_multiprocess()
    nwp = xr.open_dataset(
        zarr_path, engine="zarr", consolidated=consolidated, mode="r", chunks=None
    )

    # Sanity check.
    # TODO: Replace this with
    # pandas.core.indexes.base._is_strictly_monotonic_increasing()
    assert utils.is_monotonically_increasing(nwp.init_time.astype(int))
    assert utils.is_unique(nwp.init_time)
    return nwp
