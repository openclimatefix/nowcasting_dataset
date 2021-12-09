""" batch functions """
from __future__ import annotations

import logging
import os
from concurrent import futures
from pathlib import Path
from typing import Optional, Union

import xarray as xr
from pydantic import BaseModel

from nowcasting_dataset.config.model import Configuration
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
    def fake(configuration: Configuration):
        """Make fake batch object"""

        return Batch(**make_fake_batch(configuration=configuration))

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
    ):
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

        return Batch(**batch_dict)


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
