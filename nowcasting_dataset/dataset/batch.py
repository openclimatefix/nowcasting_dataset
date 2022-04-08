""" batch functions """
from __future__ import annotations

import logging
import os
from concurrent import futures
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xarray as xr
from pydantic import BaseModel

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.data_sources import MAP_DATA_SOURCE_NAME_TO_CLASS
from nowcasting_dataset.data_sources.fake.batch import make_fake_batch
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.metadata.metadata_model import Metadata, load_from_csv
from nowcasting_dataset.data_sources.nwp.nwp_model import NWP
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.satellite.satellite_model import HRVSatellite, Satellite
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.data_sources.topographic.topographic_model import Topographic
from nowcasting_dataset.filesystem.utils import download_to_local
from nowcasting_dataset.utils import get_netcdf_filename

_LOG = logging.getLogger(__name__)

data_sources = [Satellite, HRVSatellite, Topographic, PV, Sun, GSP, NWP]


class Batch(BaseModel):
    """
    Batch data object

    Contains the following data sources
    - gsp, satellite, topogrpahic, sun, pv, nwp and datetime.

    All data sources are xr.Datasets

    """

    metadata: Metadata

    satellite: Optional[Satellite]
    hrvsatellite: Optional[HRVSatellite]
    topographic: Optional[Topographic]
    opticalflow: Optional[OpticalFlow]
    pv: Optional[PV]
    sun: Optional[Sun]
    gsp: Optional[GSP]
    nwp: Optional[NWP]

    @property
    def data_sources(self):
        """The different data sources"""
        return [
            self.satellite,
            self.hrvsatellite,
            self.topographic,
            self.opticalflow,
            self.pv,
            self.sun,
            self.gsp,
            self.nwp,
        ]

    @staticmethod
    def fake(configuration: Configuration, temporally_align_examples: bool = False):
        """
        Make fake batch object

        Args:
            configuration: configuration of dataset
            temporally_align_examples: ption to align examples (within the batch) in time

        Returns: batch object
        """

        return Batch(
            **make_fake_batch(
                configuration=configuration, temporally_align_examples=temporally_align_examples
            )
        )

    def save_netcdf(self, batch_i: int, path: Path):
        """
        Save batch to netcdf file

        Args:
            batch_i: the batch id, used to make the filename
            path: the path where it will be saved. This can be local or in the cloud.

        """
        with futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the executor.
            for data_source in self.data_sources:
                if data_source is not None:
                    _ = executor.submit(
                        data_source.save_netcdf,
                        batch_i=batch_i,
                        path=path,
                    )

        # save metadata
        self.metadata.save_to_csv(path=path)

    @staticmethod
    def load_netcdf(
        local_netcdf_path: Union[Path, str],
        batch_idx: int,
        data_sources_names: Optional[list[str]] = None,
    ) -> Batch:
        """Load batch from netcdf file"""
        if data_sources_names is None:
            data_sources_names = Example.__fields__.keys()

        # set up futures executor
        batch_dict = {}
        with futures.ThreadPoolExecutor() as executor:
            future_examples_per_source = []

            # loop over data sources
            for data_source_name in data_sources_names:

                local_netcdf_filename = os.path.join(
                    local_netcdf_path, data_source_name, get_netcdf_filename(batch_idx)
                )

                # If the file exists, load it, otherwise data source isn't used
                if os.path.isfile(local_netcdf_filename):
                    # submit task
                    future_examples = executor.submit(
                        xr.load_dataset,
                        filename_or_obj=local_netcdf_filename,
                        engine="h5netcdf",
                    )
                    future_examples_per_source.append([data_source_name, future_examples])
                else:
                    _LOG.error(
                        f"{local_netcdf_filename} does not exists,"
                        f"this is for {data_source_name} data source"
                    )

        # Collect results from each thread.
        for data_source_name, future_examples in future_examples_per_source:
            xr_dataset = future_examples.result()

            # get data source model object
            data_source_class = MAP_DATA_SOURCE_NAME_TO_CLASS[data_source_name]
            data_source_model = data_source_class.get_data_model_for_batch()

            batch_dict[data_source_name] = data_source_model(xr_dataset)

        # load metadata
        batch_size = len(batch_dict[list(data_sources_names)[0]].example)
        metadata = load_from_csv(path=local_netcdf_path, batch_size=batch_size, batch_idx=batch_idx)
        batch_dict["metadata"] = metadata.dict()

        # quick options to turn on when using legacy data e.g v15
        legacy_data = False
        if legacy_data:
            # legacy GSP
            if "gsp" in batch_dict.keys():
                batch_dict["gsp"] = batch_dict["gsp"].rename(
                    {"x_coords": "x_osgb", "y_coords": "y_osgb"}
                )

            # legacy PV
            if "pv" in batch_dict.keys():
                batch_dict["pv"] = batch_dict["pv"].rename(
                    {"x_coords": "x_osgb", "y_coords": "y_osgb"}
                )

            # legacy NWP
            if "nwp" in batch_dict.keys():
                batch_dict["nwp"] = batch_dict["nwp"].rename(
                    {
                        "x_index": "x_osgb_index",
                        "y_index": "y_osgb_index",
                        "x": "x_osgb",
                        "y": "y_osgb",
                    }
                )

        try:
            batch = Batch(**batch_dict)
        except Exception as e:
            _LOG.error(
                f"Could not make batch for batch_idx {batch_idx}, "
                f"file {local_netcdf_path}, raw dict is {batch_dict}"
            )
            raise e

        return batch

    @staticmethod
    def download_batch_and_load_batch(
        batch_idx, tmp_path: str, src_path: str, data_sources_names: Optional[List[str]] = None
    ) -> Batch:
        """
        Download batch from src to temp

        Args:
            batch_idx: which batch index to download and load
            data_sources_names: list of data source names
            tmp_path: the temporary path, where files are downloaded to
            src_path: the path where files are downloaded from

        Returns: batch object

        """
        if data_sources_names is None:
            data_sources_names = list(Example.__fields__.keys())

        if batch_idx == 0:
            for data_source in data_sources_names:
                os.makedirs(f"{tmp_path}/{data_source}", exist_ok=True)

        # download all data files
        for data_source in data_sources_names:
            data_source_and_filename = f"{data_source}/{get_netcdf_filename(batch_idx)}"
            download_to_local(
                remote_filename=f"{src_path}/{data_source_and_filename}",
                local_filename=f"{tmp_path}/{data_source_and_filename}",
            )

        # download locations file
        download_to_local(
            remote_filename=f"{src_path}/"
            f"{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}",
            local_filename=f"{tmp_path}/"
            f"{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}",
        )

        return Batch.load_netcdf(
            local_netcdf_path=tmp_path, batch_idx=batch_idx, data_sources_names=data_sources_names
        )


class Example(BaseModel):
    """
    Single Data item

    Note that this is currently not really used
    """

    satellite: Optional[Satellite]
    hrvsatellite: Optional[HRVSatellite]
    topographic: Optional[Topographic]
    opticalflow: Optional[OpticalFlow]
    pv: Optional[PV]
    sun: Optional[Sun]
    gsp: Optional[GSP]
    nwp: Optional[NWP]

    @property
    def data_sources(self):
        """The different data sources"""
        return [
            self.satellite,
            self.hrvsatellite,
            self.opticalflow,
            self.topographic,
            self.pv,
            self.sun,
            self.gsp,
            self.nwp,
        ]


def join_two_batches(
    batches: List[Batch],
    data_sources_names: Optional[List[str]] = None,
    first_batch_examples: Optional[List[int]] = None,
    second_batch_examples: Optional[List[int]] = None,
) -> Batch:
    """
    Join two batches

    Args:
        batches: list of batches to be mixes
        data_sources_names: list of data source names
        first_batch_examples: list of indexes that we should use for the first batch
        second_batch_examples: list of indexes that we should use for the second batch

    Returns: batch object, mixture of two given

    """

    if len(batches) == 1:
        return batches[0]

    assert len(batches) == 2, f"Can only join list of two batches, was given {len(batches)} batches"

    if data_sources_names is None:
        data_sources_names = list(Example.__fields__.keys())

    batch = batches[0]
    batch_size = batch.metadata.batch_size

    first_batch_examples = _make_batch_examples(
        batch_examples=first_batch_examples, batch_size=batch_size, size=int(batch_size / 2)
    )
    first_size = len(first_batch_examples)

    second_batch_examples = _make_batch_examples(
        batch_examples=second_batch_examples,
        batch_size=batch_size - first_size,
        size=int(batch_size / 2),
    )
    second_size = len(second_batch_examples)

    # check the sizes are right
    assert first_size + second_size == batch_size, (
        f"number of first ({first_size}) "
        f"and second ({second_size}) batch examples "
        f"should add up to the batch size of {batch_size}"
    )

    for data_source in data_sources_names:
        first = getattr(batches[0], data_source).sel(example=first_batch_examples)
        second = getattr(batches[1], data_source).sel(example=second_batch_examples)

        # reset examples index
        first.__setitem__("example", range(0, first_size))
        second.__setitem__("example", range(first_size, batch_size))

        # join on example index
        data = xr.concat([first, second], dim="example")

        # order
        data = data.sortby("example")

        # set
        setattr(batch, data_source, data)

    # merge metadata
    metadata = batch.metadata
    # loop over metadata keys, but no 'batch_size'
    for metadata_key in Metadata.__fields__.keys():
        if metadata_key != "batch_size":
            first_data = np.array(getattr(batch.metadata, metadata_key))[first_batch_examples]
            second_data = np.array(getattr(batches[1].metadata, metadata_key))[
                second_batch_examples
            ]
            data = np.concatenate(([first_data, second_data]))
            setattr(metadata, metadata_key, data)

    return batch


def _make_batch_examples(
    size: int, batch_size: int, batch_examples: Optional[List[str]] = None
) -> List[int]:
    """
    Make random batch examples

    Args:
        size: the size of examples
        batch_size: the total batch size of the batch
        batch_examples: optional batch examples

    Returns: random batch examples

    """
    # create first random indexes, if needed
    if batch_examples is None:
        batch_examples = np.random.choice(range(0, batch_size), size=size, replace=False)

    return batch_examples
