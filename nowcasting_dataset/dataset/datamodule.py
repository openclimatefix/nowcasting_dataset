from typing import Union, Optional, Iterable, Dict, Callable
from pathlib import Path
import pandas as pd
from copy import deepcopy
import torch
import logging
from nowcasting_dataset import data_sources
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset import time as nd_time
from nowcasting_dataset import utils
from nowcasting_dataset import consts
from nowcasting_dataset.dataset import datasets
from dataclasses import dataclass
from nowcasting_dataset.dataset.split.split import split_data, SplitMethod
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pytorch_lightning as pl

logger = logging.getLogger(__name__)


@dataclass
class NowcastingDataModule(pl.LightningDataModule):
    """
    Attributes (additional to the dataclass attributes):
      pv_data_source: PVDataSource
      sat_data_source: SatelliteDataSource
      data_sources: List[DataSource]
      train_t0_datetimes: pd.DatetimeIndex
      val_t0_datetimes: pd.DatetimeIndex
    """

    pv_power_filename: Optional[Union[str, Path]] = None
    pv_metadata_filename: Optional[Union[str, Path]] = None
    batch_size: int = 8
    n_training_batches_per_epoch: int = 25_000
    n_validation_batches_per_epoch: int = 1_000
    n_test_batches_per_epoch: int = 1_000
    history_minutes: int = 30  #: Number of minutes of history, not including t0.
    forecast_minutes: int = 60  #: Number of minutes of forecast, not including t0.
    sat_filename: Union[str, Path] = consts.SAT_FILENAME
    sat_channels: Iterable[str] = ("HRV",)
    normalise_sat: bool = True
    nwp_base_path: Optional[str] = None
    nwp_channels: Optional[Iterable[str]] = (
        "t",
        "dswrf",
        "prate",
        "r",
        "sde",
        "si10",
        "vis",
        "lcc",
        "mcc",
        "hcc",
    )
    satellite_image_size_pixels: int = 128  #: Passed to Data Sources.
    topographic_filename: Optional[Union[str, Path]] = None
    nwp_image_size_pixels: int = 2  #: Passed to Data Sources.
    meters_per_pixel: int = 2000  #: Passed to Data Sources.
    convert_to_numpy: bool = True  #: Passed to Data Sources.
    pin_memory: bool = True  #: Passed to DataLoader.
    num_workers: int = 16  #: Passed to DataLoader.
    prefetch_factor: int = 64  #: Passed to DataLoader.
    n_samples_per_timestep: int = 2  #: Passed to NowcastingDataset
    collate_fn: Callable = (
        torch.utils.data._utils.collate.default_collate
    )  #: Passed to NowcastingDataset
    gsp_filename: Optional[Union[str, Path]] = None
    train_validation_percentage_split: float = 20
    pv_load_azimuth_and_elevation: bool = False
    split_method: SplitMethod = SplitMethod.DAY  # which split method should be used
    seed: Optional[int] = None  # seed used to make quasi random split data

    skip_n_train_batches: int = 0  # number of train batches to skip
    skip_n_validation_batches: int = 0  # number of validation batches to skip
    skip_n_test_batches: int = 0  # number of test batches to skip

    def __post_init__(self):
        super().__init__()

        self.history_len_30_minutes = self.history_minutes // 30
        self.forecast_len_30_minutes = self.forecast_minutes // 30

        self.history_len_5_minutes = self.history_minutes // 5
        self.forecast_len_5_minutes = self.forecast_minutes // 5

        # Plus 1 because neither history_len nor forecast_len include t0.
        self._total_seq_len_5_minutes = self.history_len_5_minutes + self.forecast_len_5_minutes + 1
        self._total_seq_len_30_minutes = (
            self.history_len_30_minutes + self.forecast_len_30_minutes + 1
        )
        self.contiguous_dataset = None
        if self.num_workers == 0:
            self.prefetch_factor = 2  # Set to default when not using multiprocessing.

    def prepare_data(self) -> None:
        # Satellite data
        n_timesteps_per_batch = self.batch_size // self.n_samples_per_timestep

        self.sat_data_source = data_sources.SatelliteDataSource(
            filename=self.sat_filename,
            image_size_pixels=self.satellite_image_size_pixels,
            meters_per_pixel=self.meters_per_pixel,
            history_minutes=self.history_minutes,
            forecast_minutes=self.forecast_minutes,
            channels=self.sat_channels,
            n_timesteps_per_batch=n_timesteps_per_batch,
            convert_to_numpy=self.convert_to_numpy,
            normalise=self.normalise_sat,
        )

        self.data_sources = [self.sat_data_source]
        sat_datetimes = self.sat_data_source.datetime_index()

        # PV
        if self.pv_power_filename is not None:

            self.pv_data_source = data_sources.PVDataSource(
                filename=self.pv_power_filename,
                metadata_filename=self.pv_metadata_filename,
                start_dt=sat_datetimes[0],
                end_dt=sat_datetimes[-1],
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                convert_to_numpy=self.convert_to_numpy,
                image_size_pixels=self.satellite_image_size_pixels,
                meters_per_pixel=self.meters_per_pixel,
                get_center=False,
                load_azimuth_and_elevation=self.pv_load_azimuth_and_elevation,
            )

            self.data_sources = [self.pv_data_source, self.sat_data_source]

        if self.gsp_filename is not None:
            self.gsp_data_source = GSPDataSource(
                filename=self.gsp_filename,
                start_dt=sat_datetimes[0],
                end_dt=sat_datetimes[-1],
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                convert_to_numpy=self.convert_to_numpy,
                image_size_pixels=self.satellite_image_size_pixels,
                meters_per_pixel=self.meters_per_pixel,
                get_center=True,
            )

            # put gsp data source at the start, so data is centered around GSP. This is the current approach,
            # but in the future we may take a mix of GSP and PV data as the centroid
            self.data_sources = [self.gsp_data_source] + self.data_sources

        # NWP data
        if self.nwp_base_path is not None:
            self.nwp_data_source = data_sources.NWPDataSource(
                filename=self.nwp_base_path,
                image_size_pixels=self.nwp_image_size_pixels,
                meters_per_pixel=self.meters_per_pixel,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                channels=self.nwp_channels,
                n_timesteps_per_batch=n_timesteps_per_batch,
                convert_to_numpy=self.convert_to_numpy,
            )

            self.data_sources.append(self.nwp_data_source)

        # Topographic data
        if self.topographic_filename is not None:
            self.topo_data_source = data_sources.TopographicDataSource(
                filename=self.topographic_filename,
                image_size_pixels=self.satellite_image_size_pixels,
                meters_per_pixel=self.meters_per_pixel,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                convert_to_numpy=self.convert_to_numpy,
                normalize=self.normalise_sat,
            )

            self.data_sources.append(self.topo_data_source)

        self.datetime_data_source = data_sources.DatetimeDataSource(
            history_minutes=self.history_minutes,
            forecast_minutes=self.forecast_minutes,
            convert_to_numpy=self.convert_to_numpy,
        )
        self.data_sources.append(self.datetime_data_source)

    def setup(self, stage="fit"):
        """Split data, etc.

        Args:
          stage: {'fit', 'predict', 'test', 'validate'} This code ignores this.

        ## Selecting daytime data.

        We're interested in forecasting solar power generation, so we
        don't care about nighttime data :)

        In the UK in summer, the sun rises first in the north east, and
        sets last in the north west [1].  In summer, the north gets more
        hours of sunshine per day.

        In the UK in winter, the sun rises first in the south east, and
        sets last in the south west [2].  In winter, the south gets more
        hours of sunshine per day.

        |                        | Summer | Winter |
        |           ---:         |  :---: |  :---: |
        | Sun rises first in     | N.E.   | S.E.   |
        | Sun sets last in       | N.W.   | S.W.   |
        | Most hours of sunlight | North  | South  |

        Before training, we select timesteps which have at least some
        sunlight.  We do this by computing the clearsky global horizontal
        irradiance (GHI) for the four corners of the satellite imagery,
        and for all the timesteps in the dataset.  We only use timesteps
        where the maximum global horizontal irradiance across all four
        corners is above some threshold.

        The 'clearsky solar irradiance' is the amount of sunlight we'd
        expect on a clear day at a specific time and location. The SI unit
        of irradiance is watt per square meter.  The 'global horizontal
        irradiance' (GHI) is the total sunlight that would hit a
        horizontal surface on the surface of the Earth.  The GHI is the
        sum of the direct irradiance (sunlight which takes a direct path
        from the Sun to the Earth's surface) and the diffuse horizontal
        irradiance (the sunlight scattered from the atmosphere).  For more
        info, see: https://en.wikipedia.org/wiki/Solar_irradiance

        References:
          1. [Video of June 2019](https://www.youtube.com/watch?v=IOp-tj-IJpk)
          2. [Video of Jan 2019](https://www.youtube.com/watch?v=CJ4prUVa2nQ)
        """
        del stage  # Not used in this method!
        self._split_data()

        # Create datasets
        logger.debug("Making train dataset")
        self.train_dataset = datasets.NowcastingDataset(
            t0_datetimes=self.train_t0_datetimes,
            data_sources=self.data_sources,
            skip_batch_index=self.skip_n_train_batches,
            n_batches_per_epoch_per_worker=(
                self._n_batches_per_epoch_per_worker(self.n_training_batches_per_epoch)
            ),
            **self._common_dataset_params(),
        )
        logger.debug("Making validation dataset")
        self.val_dataset = datasets.NowcastingDataset(
            t0_datetimes=self.val_t0_datetimes,
            data_sources=self.data_sources,
            skip_batch_index=self.skip_n_validation_batches,
            n_batches_per_epoch_per_worker=(
                self._n_batches_per_epoch_per_worker(self.n_validation_batches_per_epoch)
            ),
            **self._common_dataset_params(),
        )
        logger.debug("Making validation dataset: done")

        logger.debug("Making test dataset")
        self.test_dataset = datasets.NowcastingDataset(
            t0_datetimes=self.test_t0_datetimes,
            data_sources=self.data_sources,
            skip_batch_index=self.skip_n_test_batches,
            n_batches_per_epoch_per_worker=(
                self._n_batches_per_epoch_per_worker(self.n_test_batches_per_epoch)
            ),
            **self._common_dataset_params(),
        )
        logger.debug("Making test dataset: done")

        if self.num_workers == 0:
            self.train_dataset.per_worker_init(worker_id=0)
            self.val_dataset.per_worker_init(worker_id=0)
            self.test_dataset.per_worker_init(worker_id=0)

        logger.debug("Setup: done")

    def _n_batches_per_epoch_per_worker(self, n_batches_per_epoch: int) -> int:
        if self.num_workers > 0:
            return n_batches_per_epoch // self.num_workers
        else:
            return n_batches_per_epoch

    def _split_data(self):
        """Sets self.train_t0_datetimes and self.val_t0_datetimes."""

        logger.debug("Going to split data")

        self._check_has_prepared_data()
        self.t0_datetimes = self._get_datetimes(interpolate_for_30_minute_data=True)

        logger.debug(f"Got all start times, there are {len(self.t0_datetimes)}")

        self.train_t0_datetimes, self.val_t0_datetimes, self.test_t0_datetimes = split_data(
            datetimes=self.t0_datetimes, method=self.split_method, seed=self.seed
        )

        logger.debug(
            f"Split data done, train has {len(self.train_t0_datetimes)}, "
            f"validation has {len(self.val_t0_datetimes)}"
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, **self._common_dataloader_params())

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, **self._common_dataloader_params())

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test_dataset, **self._common_dataloader_params())

    def contiguous_dataloader(self) -> torch.utils.data.DataLoader:
        if self.contiguous_dataset is None:
            pv_data_source = deepcopy(self.pv_data_source)
            pv_data_source.random_pv_system_for_given_location = False
            data_sources = [pv_data_source, self.sat_data_source]
            self.contiguous_dataset = datasets.ContiguousNowcastingDataset(
                t0_datetimes=self.val_t0_datetimes,
                data_sources=data_sources,
                n_batches_per_epoch_per_worker=(self._n_batches_per_epoch_per_worker(32)),
                **self._common_dataset_params(),
            )
            if self.num_workers == 0:
                self.contiguous_dataset.per_worker_init(worker_id=0)
        return torch.utils.data.DataLoader(
            self.contiguous_dataset, **self._common_dataloader_params()
        )

    def _common_dataset_params(self) -> Dict:
        return dict(
            batch_size=self.batch_size,
            n_samples_per_timestep=self.n_samples_per_timestep,
            collate_fn=self.collate_fn,
        )

    def _common_dataloader_params(self) -> Dict:
        return dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            worker_init_fn=datasets.worker_init_fn,
            prefetch_factor=self.prefetch_factor,
            # Disable automatic batching because NowcastingDataset.__iter__
            # returns complete batches
            batch_size=None,
            batch_sampler=None,
        )

    def _get_datetimes(
        self, interpolate_for_30_minute_data: bool = False, adjust_for_sequence_length: bool = True
    ) -> pd.DatetimeIndex:
        """Compute the datetime index.

        interpolate_for_30_minute_data: If True,
        1. all datetimes from source will be interpolated to 5 min intervals,
        2. the total intersection will be taken
        3. only 30 mins datetimes will be selected

        adjust_for_sequence_length, if true, adjust the datetimes by sequence history and length.
        This means that all the datetimes from [datetime - history_delta: datetime + forecast_delta] should be available

        This deals with a mixture of data sources that have 5 mins and 30 min datatime.

        Returns the intersection of the datetime indicies of all the
        data_sources, filtered by daylight hours."""
        logger.debug("Get the datetimes")
        self._check_has_prepared_data()

        # Get the intersection of datetimes from all data sources.
        all_datetime_indexes = []
        for data_source in self.data_sources:
            logger.debug(f"Getting datetimes for {type(data_source).__name__}")
            try:

                # get datetimes from data source
                datetime_index = data_source.datetime_index()

                if interpolate_for_30_minute_data and type(data_source).__name__ == "GSPDataSource":
                    # change 30 min data to 5 mins, only for GSP Data
                    datetime_index = nd_time.fill_30_minutes_timestamps_to_5_minutes(
                        index=datetime_index
                    )

                all_datetime_indexes.append(datetime_index)
            except NotImplementedError:
                pass
        datetimes = nd_time.intersection_of_datetimeindexes(all_datetime_indexes)
        del all_datetime_indexes  # save memory

        # Select datetimes that have at least some sunlight
        border_locations = self.sat_data_source.geospatial_border()
        dt_index = nd_time.select_daylight_datetimes(
            datetimes=datetimes, locations=border_locations
        )

        # Sanity check
        assert len(dt_index) > 2
        assert utils.is_monotonically_increasing(dt_index)

        if not adjust_for_sequence_length:
            return dt_index

        # get t0 datetime which depend on the sequence length in the dataset
        t0_datetimes = nd_time.get_t0_datetimes(
            datetimes=dt_index,
            total_seq_len=self._total_seq_len_5_minutes,
            history_len=self.history_len_5_minutes,
        )

        # only select datetimes for half hours, ignore 5 minute timestamps
        if interpolate_for_30_minute_data:
            t0_datetimes = [t0 for t0 in t0_datetimes if (t0.minute in [0, 30])]

        del dt_index

        return t0_datetimes

    def _check_has_prepared_data(self):
        if not self.has_prepared_data:
            raise RuntimeError("Must run prepare_data() first!")
