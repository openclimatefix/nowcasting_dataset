""" NWP Data Source """
import logging
from concurrent import futures
from dataclasses import dataclass, InitVar
from numbers import Number
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset import utils
from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.data_sources.nwp.nwp_model import NWP
from nowcasting_dataset.dataset.xr_utils import join_list_data_array_to_batch_dataset

_LOG = logging.getLogger(__name__)

from nowcasting_dataset.consts import NWP_VARIABLE_NAMES


@dataclass
class NWPDataSource(ZarrDataSource):
    """
    NWP Data Source (Numerical Weather Predictions)

    Args (for init):
      filename: The base path in which we find '2018_1-6', etc.

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

    filename: str = None
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

    def get_batch(
        self,
        t0_datetimes: pd.DatetimeIndex,
        x_locations: Iterable[Number],
        y_locations: Iterable[Number],
    ) -> NWP:
        """
        Get batch data

        Args:
            t0_datetimes: list of timstamps
            x_locations: list of x locations, where the batch data is for
            y_locations: list of y locations, where the batch data is for

        Returns: batch data

        """
        # Lazily select time slices.
        selections = []
        for t0_dt in t0_datetimes[: self.n_timesteps_per_batch]:
            selections.append(self._get_time_slice(t0_dt))

        # Load entire time slices from disk in multiple threads.
        data = []
        with futures.ThreadPoolExecutor(max_workers=self.n_timesteps_per_batch) as executor:
            data_futures = []
            # Submit tasks.
            for selection in selections:
                future = executor.submit(selection.load)
                data_futures.append(future)

            # Grab tasks
            for future in data_futures:
                d = future.result()
                data.append(d)

        # Select squares from pre-loaded time slices.
        examples = []
        for i, (x_meters_center, y_meters_center) in enumerate(zip(x_locations, y_locations)):
            selected_data = data[i % self.n_timesteps_per_batch]
            bounding_box = self._square.bounding_box_centered_on(
                x_meters_center=x_meters_center, y_meters_center=y_meters_center
            )
            selected_data = selected_data.sel(
                x=slice(bounding_box.left, bounding_box.right),
                y=slice(bounding_box.top, bounding_box.bottom),
            )

            # selected_sat_data is likely to have 1 too many pixels in x and y
            # because sel(x=slice(a, b)) is [a, b], not [a, b).  So trim:
            selected_data = selected_data.isel(
                x=slice(0, self._square.size_pixels), y=slice(0, self._square.size_pixels)
            )

            t0_dt = t0_datetimes[i]
            selected_data = self._post_process_example(selected_data, t0_dt)

            examples.append(selected_data)

        output = join_list_data_array_to_batch_dataset(examples)

        return NWP(output)

    def _open_data(self) -> xr.DataArray:
        return open_nwp(self.filename, consolidated=self.consolidated)

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        """
        Select the numerical weather predictions for a single time slice.

        Note that this function does *not* resample from hourly to 5 minutely.
        Resampling would be very expensive if done on the whole geographical
        extent of the NWP data!  So resampling is done in
        _post_process_example().

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
        selected_data = selected_data.resample({"target_time": "5T"})
        selected_data = selected_data.interpolate()
        selected_data = selected_data.sel(target_time=slice(start_dt, end_dt))
        selected_data = selected_data.rename({"target_time": "time"})
        selected_data = selected_data.rename({"variable": "channels"})

        selected_data.data = selected_data.data.astype(np.float32)

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
        resampler = pd.Series(0, index=target_times).resample("5T")
        return resampler.ffill(limit=11).dropna().index


def open_nwp(filename: str, consolidated: bool) -> xr.Dataset:
    """
    Open The NWP data

    Args:
        filename: filename must start with 'gs://' if it's on GCP.
        consolidated: consolidate the zarr file?

    Returns: nwp data

    """
    _LOG.debug("Opening NWP data: %s", filename)
    utils.set_fsspec_for_multiprocess()
    nwp = xr.open_dataset(filename, engine="zarr", consolidated=consolidated, mode="r", chunks=None)

    # Sanity check.
    # TODO: Replace this with
    # pandas.core.indexes.base._is_strictly_monotonic_increasing()
    assert utils.is_monotonically_increasing(nwp.init_time.astype(int))
    assert utils.is_unique(nwp.init_time)
    return nwp
