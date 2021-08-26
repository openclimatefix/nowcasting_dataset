from numbers import Number
import pandas as pd
import numpy as np
from nowcasting_dataset.example import Example, to_numpy
from nowcasting_dataset import square
import nowcasting_dataset.time as nd_time
from dataclasses import dataclass, InitVar
from typing import List, Tuple, Iterable
import xarray as xr
import itertools


@dataclass
class DataSource:
    """Abstract base class.

    Attributes:
      history_len: Number of timesteps of history to include in each example.
        Does NOT include t0.  That is, if history_len = 0 then the example
        will start at t0.
      forecast_len: Number of timesteps of forecast to include in each example.
        Does NOT include t0.  If forecast_len = 0 then the example will end
        at t0.  If both history_len and forecast_len are 0, then the example
        will consist of a single timestep at t0.
      convert_to_numpy: Whether or not to convert each example to numpy.
    """
    history_len: int
    forecast_len: int
    convert_to_numpy: bool

    def __post_init__(self):
        assert self.history_len >= 0
        assert self.forecast_len >= 0
        # Plus 1 because neither history_len nor forecast_len include t0.
        self._total_seq_len = self.history_len + self.forecast_len + 1
        self._history_dur = nd_time.timesteps_to_duration(self.history_len)
        self._forecast_dur = nd_time.timesteps_to_duration(self.forecast_len)

    def _get_start_dt(self, t0_dt: pd.Timestamp) -> pd.Timestamp:
        return t0_dt - self._history_dur

    def _get_end_dt(self, t0_dt: pd.Timestamp) -> pd.Timestamp:
        return t0_dt + self._forecast_dur

    # ************* METHODS THAT CAN BE OVERRIDDEN ****************************
    def open(self):
        """Open the data source, if necessary.

        Called from each worker process.  Useful for data sources where the
        underlying data source cannot be forked (like Zarr on GCP!).

        Data sources which can be forked safely should call open()
        from __init__().
        """
        pass

    def get_batch(
            self,
            t0_datetimes: pd.DatetimeIndex,
            x_locations: Iterable[Number],
            y_locations: Iterable[Number]) -> List[Example]:
        """
        Returns:
            List of Examples with data converted to Numpy data structures.
        """
        examples = []
        zipped = zip(t0_datetimes, x_locations, y_locations)
        for t0_datetime, x_location, y_location in zipped:
            example = self.get_example(t0_datetime, x_location, y_location)
            if self.convert_to_numpy:
                example = to_numpy(example)
            examples.append(example)

        return examples

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes."""
        # Leave this NotImplemented if this DataSource has no concept
        # of a list of datetimes (e.g. for DatetimeDataSource).
        raise NotImplementedError()

    def _get_time_slice(self, t0_dt: pd.Timestamp):
        """Get a single timestep of data.  Must be overridden."""
        raise NotImplementedError()

    # ****************** METHODS THAT MUST BE OVERRIDDEN **********************
    def get_locations_for_batch(
            self,
            t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        """Find a valid geographical location for each t0_datetime.

        Returns:  x_locations, y_locations. Each has one entry per t0_datetime.
            Locations are in OSGB coordinates.
        """
        # TODO: Do this properly, using PV locations!
        locations = [
            20_000, 40_000,
            500_000, 600_000,
            100_000, 100_000,
            250_000, 250_000]

        location = np.random.choice(locations, size=(len(t0_datetimes), 2))

        return location[:, 0], location[:, 1]

    def get_example(
            self,
            t0_dt: pd.Timestamp,  #: Datetime of "now": The most recent obs.
            x_meters_center: Number,  #: Centre, in OSGB coordinates.
            y_meters_center: Number  #: Centre, in OSGB coordinates.
    ) -> Example:
        """Must be overridden by child classes."""
        raise NotImplementedError()


@dataclass
class ImageDataSource(DataSource):
    """
    Args:
      image_size_pixels: Size of the width and height of the image crop
        returned by get_sample(). """
    image_size_pixels: InitVar[int]
    meters_per_pixel: InitVar[int]

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__()
        self._square = square.Square(
            size_pixels=image_size_pixels,
            meters_per_pixel=meters_per_pixel)


@dataclass
class ZarrDataSource(ImageDataSource):
    """
    Attributes:
      _data: xr.DataArray data, opened by open().
        x is left-to-right.
        y is top-to-bottom.
        Access using public data property.
      consolidated: Whether or not the Zarr store is consolidated.
      channels: The Zarr parameters to load.
    """
    channels: Iterable[str]
    #: Mustn't be None, but cannot have a non-default arg in this position :)
    n_timesteps_per_batch: int = None
    consolidated: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._data = None
        if self.n_timesteps_per_batch is None:
            raise ValueError('n_timesteps_per_batch must be set!')

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError('Please run `open()` before accessing data!')
        return self._data

    def get_example(
            self,
            t0_dt: pd.Timestamp,
            x_meters_center: Number,
            y_meters_center: Number
    ) -> Example:
        selected_data = self._get_time_slice(t0_dt)
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

        selected_data = self._post_process_example(selected_data, t0_dt)

        if selected_data.shape != self._shape_of_example:
            raise RuntimeError(
                'Example is wrong shape! '
                f'x_meters_center={x_meters_center}\n'
                f'y_meters_center={y_meters_center}\n'
                f't0_dt={t0_dt}\n'
                f'expected shape={self._shape_of_example}\n'
                f'actual shape   {selected_data.shape}')

        return self._put_data_into_example(selected_data)

    def geospatial_border(self) -> List[Tuple[Number, Number]]:
        """Get 'corner' coordinates for a rectangle within the boundary of the
        data.

        Returns List of 2-tuples of the x and y coordinates of each corner,
        in OSGB projection.
        """
        GEO_BORDER: int = 64  #: In same geo projection and units as sat_data.
        data = self._open_data()
        return [
            (data.x.values[x], data.y.values[y])
            for x, y in itertools.product(
                [GEO_BORDER, -GEO_BORDER],
                [GEO_BORDER, -GEO_BORDER])]

    # ****************** METHODS THAT CAN BE OVERRIDDEN **********************
    def _post_process_example(
            self,
            selected_data: xr.DataArray,
            t0_dt: pd.Timestamp) -> xr.DataArray:
        return selected_data

    # ****************** METHODS THAT MUST BE OVERRIDDEN **********************
    # (in addition to the DataSource methods that must be overridden)
    def open(self) -> None:
        # We don't want to _open_data() in __init__.
        # If we did that, then we couldn't copy ZarrDataSource
        # instances into separate processes.  Instead,
        # call open() _after_ creating separate processes.
        raise NotImplementedError()

    def _open_data(self) -> xr.DataArray:
        raise NotImplementedError()

    def _put_data_into_example(self, selected_data: xr.DataArray) -> Example:
        raise NotImplementedError()
