from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.example import Example, to_numpy
from nowcasting_dataset import utils
from typing import Iterable, Optional, List
import xarray as xr
import pandas as pd
import logging
from dataclasses import dataclass, InitVar
import numpy as np
from numbers import Number
from concurrent import futures

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
    filename: str = None
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
        self._data = data['UKV'].sel(variable=list(self.channels))

    def get_batch(
            self,
            t0_datetimes: pd.DatetimeIndex,
            x_locations: Iterable[Number],
            y_locations: Iterable[Number]) -> List[Example]:

        # Lazily select time slices.
        selections = []
        for t0_dt in t0_datetimes[:self.n_timesteps_per_batch]:
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
                x_meters_center=x_meters_center, y_meters_center=y_meters_center)
            selected_data = selected_data.sel(
                x=slice(bounding_box.left, bounding_box.right),
                y=slice(bounding_box.top, bounding_box.bottom))

            # selected_sat_data is likely to have 1 too many pixels in x and y
            # because sel(x=slice(a, b)) is [a, b], not [a, b).  So trim:
            selected_data = selected_data.isel(
                x=slice(0, self._square.size_pixels),
                y=slice(0, self._square.size_pixels))

            t0_dt = t0_datetimes[i]
            selected_data = self._post_process_example(selected_data, t0_dt)

            example = self._put_data_into_example(selected_data)
            if self.convert_to_numpy:
                example = to_numpy(example)
            examples.append(example)
        return examples

    def _open_data(self) -> xr.DataArray:
        return open_nwp(self.filename, consolidated=self.consolidated)

    def _put_data_into_example(self, selected_data: xr.DataArray) -> Example:
        return Example(
            nwp=selected_data,
            nwp_x_coords=selected_data.x,
            nwp_y_coords=selected_data.y,
            nwp_target_time=selected_data.target_time)

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        """Select the numerical weather predictions for a single time slice.

        Note that this function does *not* resample from hourly to 5 minutely.
        Resampling would be very expensive if done on the whole geographical
        extent of the NWP data!  So resampling is done in
        _post_process_example()."""
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        start_hourly = start_dt.floor('H')
        end_hourly = end_dt.ceil('H')

        init_time_i = np.searchsorted(self.data.init_time, start_hourly.to_numpy(), side='right')
        init_time_i -= 1  # Because searchsorted() gives the index to the entry _after_.
        init_time = self.data.init_time.values[init_time_i]

        step_start = start_hourly - init_time
        step_end = end_hourly - init_time

        selected = self.data.sel(init_time=init_time, step=slice(step_start, step_end))
        selected = selected.swap_dims({'step': 'target_time'})
        selected['target_time'] = init_time + selected.step
        return selected

    def _post_process_example(
            self,
            selected_data: xr.DataArray,
            t0_dt: pd.Timestamp) -> xr.DataArray:
        """Resamples to 5 minutely."""
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        selected_data = selected_data - NWP_MEAN
        selected_data = selected_data / NWP_STD
        selected_data = selected_data.resample({'target_time': '5T'})
        selected_data = selected_data.interpolate()
        selected_data = selected_data.sel(target_time=slice(start_dt, end_dt))
        return selected_data

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
        filename, engine='zarr', consolidated=consolidated, mode='r', chunks=None)

    # Sanity check.
    # TODO: Replace this with
    # pandas.core.indexes.base._is_strictly_monotonic_increasing()
    assert utils.is_monotonically_increasing(nwp.init_time.astype(int))
    assert utils.is_unique(nwp.init_time)
    return nwp
