import os
import dask
from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.example import Example
from nowcasting_dataset import utils
from typing import Iterable, Optional, List
import xarray as xr
import pandas as pd
import logging
from dataclasses import dataclass, InitVar
import numpy as np
from numbers import Number

_LOG = logging.getLogger('nowcasting_dataset')

NWP_VARIABLE_NAMES = (
    't', 'dswrf', 'prate', 'r', 'sde', 'si10', 'vis', 'lcc', 'mcc', 'hcc')

# Means computed with
# nwp_ds = NWPDataSource(...)
# nwp_ds.open()
# mean = nwp_ds.data.isel(init_time=slice(0, 10)).mean(
#     dim=['step', 'x', 'init_time', 'y']).compute()
NWP_MEAN = xr.DataArray(
    data=[
        2.8041010e+02, 1.6854691e+01, 6.7529683e-05, 8.1832832e+01,
        7.1233767e-03, 8.8566933e+00, 4.3474598e+04, 4.9820110e+01,
        4.8095409e+01, 4.2833260e+01],
    dims=['variable'],
    coords={'variable': list(NWP_VARIABLE_NAMES)}).astype(np.float32)

NWP_STD = xr.DataArray(
    data=[
        2.5812180e+00, 4.1278820e+01, 2.7507244e-04, 9.0967312e+00,
        1.4110464e-01, 4.3616886e+00, 2.3853148e+04, 3.8900299e+01,
        4.2830105e+01, 4.2778091e+01],
    dims=['variable'],
    coords={'variable': list(NWP_VARIABLE_NAMES)}).astype(np.float32)


@dataclass
class NWPDataSource(ZarrDataSource):
    """
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
    channels: Optional[Iterable[str]] = NWP_VARIABLE_NAMES
    image_size_pixels: InitVar[int] = 2
    meters_per_pixel: InitVar[int] = 2_000

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        n_channels = len(self.channels)
        self._shape_of_example = (
            n_channels, self._total_seq_len, image_size_pixels,
            image_size_pixels)

    def open(self) -> None:
        # We don't want to open_sat_data in __init__.
        # If we did that, then we couldn't copy NWPDataSource
        # instances into separate processes.  Instead,
        # call open() _after_ creating separate processes.
        data = self._open_data()
        data = data['UKV'].sel(variable=list(self.channels))
        data -= NWP_MEAN
        data /= NWP_STD
        self._data = data

    def get_batch(
            self,
            t0_datetimes: pd.DatetimeIndex,
            x_locations: Iterable[Number],
            y_locations: Iterable[Number]) -> List[Example]:

        examples_delayed = super().get_batch(
            t0_datetimes=t0_datetimes,
            x_locations=x_locations,
            y_locations=y_locations)
        return dask.compute(examples_delayed)[0]

    def _open_data(self) -> xr.DataArray:
        return open_nwp(self.filename, consolidated=self.consolidated)

    def _put_data_into_example(self, selected_data: xr.DataArray) -> Example:
        return Example(nwp=selected_data)

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        """Select the numerical weather predictions for a single time slice.

        Note that this function does *not* resample from hourly to 5 minutely.
        Resampling would be very expensive if done on the whole geographical
        extent of the NWP data!  So resampling is done in
        _post_process_example().

        The NWP for each example covers a contiguous timespan running
        from `start_dt` to `end_dt`.  The first part of the timeseries
        [`start_dt`, `t0`] is the 'recent history'.  The second part of
        the timeseries (`t0`, `end_dt`] is the 'future'.

        For each timestep in the recent history [`start`, `t0`], get
        predictions produced by the freshest NWP run to each timestep.

        For the future (`t0`, `end`], use the NWP initialised most
        recently to t0.

        THIS METHOD IS NOT THREAD-SAFE!
        Both the calls to self.data.sel() break if used from multiple
        threads!  This is due to a Pandas bug
        whereby Index.is_unique() sometimes returns false when
        used in a multi-threading context.  See:
        https://github.com/pandas-dev/pandas/issues/21150
        https://github.com/openclimatefix/nowcasting_dataset/issues/42
        """
        # First we get the hourly NWPs; then we resample to `freq` at
        # the end of the function.
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        start_hourly = start_dt.floor('H')
        t0_hourly = t0_dt.ceil('H')
        end_hourly = end_dt.ceil('H')

        target_times_hourly = pd.date_range(start_hourly, end_hourly, freq='H')

        # Get the most recent NWP initialisation time for each
        # target_time_hourly.
        init_times = self.data.sel(
            init_time=target_times_hourly, method='ffill').init_time.values

        # Find the NWP init time for just the 'future' portion of the example.
        init_time_future = init_times[target_times_hourly == t0_hourly]

        # For the 'future' portion of the example, replace all the NWP
        # init times with the NWP init time most recent to t0.
        init_times[target_times_hourly > t0_hourly] = init_time_future

        # steps is the number of hourly timesteps beyond the NWP
        # initialisation time.
        steps = target_times_hourly - init_times

        def _get_data_array_indexer(index):
            # We want one timestep for each target_time_hourly
            # (obviously!)  If we simply do
            # nwp.sel(init_time=init_times, step=steps) then we'll get
            # the *product* of init_times and steps, which is not what
            # we want!  Instead, we use xarray's vectorized-indexing
            # mode by using a DataArray indexer.  See the last example here:
            # http://xarray.pydata.org/en/stable/user-guide/indexing.html#vectorized-indexing
            return xr.DataArray(
                index, dims='target_time',
                coords={'target_time': target_times_hourly})

        init_time_indexer = _get_data_array_indexer(init_times)
        step_indexer = _get_data_array_indexer(steps)
        nwp_selected = self.data.sel(
            init_time=init_time_indexer, step=step_indexer)
        return nwp_selected

    def _post_process_example(
            self,
            selected_data: xr.DataArray,
            t0_dt: pd.Timestamp) -> xr.DataArray:
        """Resamples to 5 minutely."""
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        selected_data = selected_data.resample({'target_time': '5T'})
        selected_data = selected_data.interpolate()
        selected_data = selected_data.astype(np.float32)
        return selected_data.sel(target_time=slice(start_dt, end_dt))

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes"""
        if self._data is None:
            nwp = self._open_data()
        else:
            nwp = self._data
        target_times = nwp['init_time'] + nwp['step'][:3]
        target_times = target_times.values.flatten()
        target_times = np.unique(target_times)
        target_times = np.sort(target_times)
        target_times = pd.DatetimeIndex(target_times)
        resampler = pd.Series(0, index=target_times).resample('5T')
        return resampler.ffill(limit=11).dropna().index


def open_nwp(filename: str, consolidated: bool) -> xr.Dataset:
    """
    Args:
        filename must start with 'gs://' if it's on GCP.
    """
    _LOG.debug('Opening NWP data: %s', filename)
    utils.set_fsspec_for_multiprocess()
    nwp = xr.open_dataset(
        filename, engine='zarr', consolidated=consolidated, mode='r', chunks={})
    # chunks={} loads the dataset with dask using engine preferred chunks
    # if exposed by the backend, otherwise with a single chunk for all arrays.

    # Sanity check.
    # TODO: Replace this with
    # pandas.core.indexes.base._is_strictly_monotonic_increasing()
    assert utils.is_monotonically_increasing(nwp.init_time.astype(int))
    assert utils.is_unique(nwp.init_time)
    return nwp