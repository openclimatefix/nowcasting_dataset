import os
import dask
from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.example import Example
from nowcasting_dataset import utils
from typing import Iterable, Optional
import xarray as xr
import pandas as pd
import logging
from dataclasses import dataclass, InitVar
import numpy as np

_LOG = logging.getLogger('nowcasting_dataset')


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
    channels: Optional[Iterable[str]] = (
        't', 'dswrf', 'prate', 'r', 'sde', 'si10', 'vis', 'lcc', 'mcc', 'hcc')
    max_step: int = 3  #: Max forecast timesteps to load from NWPs.
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
        data = data[list(self.channels)].to_array()
        #self._data = data.sel(
        #    step=slice(pd.Timedelta(0), pd.Timedelta(hours=self.max_step + 1)))
        self._data = data

    def _open_data(self) -> xr.DataArray:
        return open_nwp(
            base_path=self.filename, consolidated=self.consolidated)

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
        recently to t0."""
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
        return nwp_selected.load()

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
        target_times = nwp['init_time'] + nwp['step'][:self.max_step]
        target_times = target_times.values.flatten()
        target_times = np.unique(target_times)
        target_times = np.sort(target_times)
        target_times = pd.DatetimeIndex(target_times)
        resampler = pd.Series(0, index=target_times).resample('5T')
        return resampler.ffill(limit=11).dropna().index


def open_nwp(base_path: str, consolidated: bool) -> xr.Dataset:
    """
    Args:
        base_path must start with 'gs://' if it's on GCP.
    """
    _LOG.debug('Opening NWP data: %s', base_path)
    utils.set_fsspec_for_multiprocess()
    nwp_datasets = []
    # TODO: Parellise the for-loop below using ThreadPoolExecutor:
    for zarr_store in ['2018_1-6', '2018_7-12', '2019_1-6', '2019_7-12']:
        full_dir = os.path.join(base_path, zarr_store)
        ds = xr.open_dataset(
            full_dir, engine='zarr', consolidated=consolidated, mode='r',
            chunks='auto')
        ds = ds.rename({'time': 'init_time'})

        # The isobaricInhPa coordinates look messed up, especially in
        # the 2018_7-12 and 2019_7-12 Zarr stores.  So let's drop all
        # the variables with multiple vertical levels for now:
        del ds['isobaricInhPa'], ds['gh_p'], ds['r_p'], ds['t_p']
        del ds['wdir_p'], ds['ws_p']

        # There are a lot of doubled-up indicies from 2018-07-18 00:00
        # to 2018-08-27 09:00.  De-duplicate the index. Code adapted
        # from https://stackoverflow.com/a/51077784/732596
        if zarr_store == '2018_7-12':
            _, unique_index = np.unique(ds.init_time, return_index=True)
            ds = ds.isel(init_time=unique_index)

        # 2019-02-01T21 is in the wrong place! It comes after
        # 2019-02-03T15.  Oops!
        if zarr_store == '2019_1-6':
            sorted_init_time = np.sort(ds.init_time)
            ds = ds.reindex(init_time=sorted_init_time)

        nwp_datasets.append(ds)

    # Concat.
    # Silence warning about large chunks
    dask.config.set({"array.slicing.split_large_chunks": False})
    nwp_concatenated = xr.concat(nwp_datasets, dim='init_time')
    
    # Sanity check.
    assert utils.is_monotonically_increasing(nwp_concatenated.init_time.astype(int))
    assert utils.is_unique(nwp_concatenated.init_time)
    return nwp_concatenated
