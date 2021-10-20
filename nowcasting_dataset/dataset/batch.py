""" batch functions """
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union
from concurrent import futures

import xarray as xr
from pydantic import BaseModel, Field

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.data_sources.datetime.datetime_model import Datetime
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.metadata.metadata_model import Metadata
from nowcasting_dataset.data_sources.nwp.nwp_model import (
    NWP,
)
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.data_sources.topographic.topographic_model import Topographic
from nowcasting_dataset.dataset.xr_utils import (
    register_xr_data_array_to_tensor,
    register_xr_data_set_to_tensor,
)
from nowcasting_dataset.data_sources.fake import (
    datetime_fake,
    metadata_fake,
    gsp_fake,
    pv_fake,
    satellite_fake,
    sun_fake,
    topographic_fake,
    nwp_fake,
)

_LOG = logging.getLogger(__name__)

register_xr_data_array_to_tensor()
register_xr_data_set_to_tensor()

data_sources = [Metadata, Satellite, Topographic, PV, Sun, GSP, NWP, Datetime]


class Batch(BaseModel):
    """
    Batch data object

    Contains the following data sources
    - gsp, satellite, topogrpahic, sun, pv, nwp and datetime.
    Also contains metadata of the class.

    All data sources are xr.Datasets

    """

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    metadata: Optional[Metadata]
    satellite: Optional[Satellite]
    topographic: Optional[Topographic]
    pv: Optional[PV]
    sun: Optional[Sun]
    gsp: Optional[GSP]
    nwp: Optional[NWP]
    datetime: Optional[Datetime]

    @property
    def data_sources(self):
        """The different data sources"""
        return [
            self.satellite,
            self.topographic,
            self.pv,
            self.sun,
            self.gsp,
            self.nwp,
            self.datetime,
            self.metadata,
        ]

    @staticmethod
    def fake(configuration: Configuration = Configuration()):
        """ Make fake batch object """
        batch_size = configuration.process.batch_size
        satellite_image_size_pixels = configuration.input_data.satellite.satellite_image_size_pixels
        nwp_image_size_pixels = configuration.input_data.nwp.nwp_image_size_pixels

        return Batch(
            batch_size=batch_size,
            satellite=satellite_fake(
                batch_size=batch_size,
                seq_length_5=configuration.input_data.satellite.seq_length_5_minutes,
                satellite_image_size_pixels=satellite_image_size_pixels,
                number_sat_channels=len(configuration.input_data.satellite.sat_channels),
            ),
            nwp=nwp_fake(
                batch_size=batch_size,
                seq_length_5=configuration.input_data.nwp.seq_length_5_minutes,
                image_size_pixels=nwp_image_size_pixels,
                number_nwp_channels=len(configuration.input_data.nwp.nwp_channels),
            ),
            metadata=metadata_fake(batch_size=batch_size),
            pv=pv_fake(
                batch_size=batch_size,
                seq_length_5=configuration.input_data.pv.seq_length_5_minutes,
                n_pv_systems_per_batch=128,
            ),
            gsp=gsp_fake(
                batch_size=batch_size,
                seq_length_30=configuration.input_data.gsp.seq_length_30_minutes,
                n_gsp_per_batch=32,
            ),
            sun=sun_fake(
                batch_size=batch_size,
                seq_length_5=configuration.input_data.sun.seq_length_5_minutes,
            ),
            topographic=topographic_fake(
                batch_size=batch_size, image_size_pixels=satellite_image_size_pixels
            ),
            datetime=datetime_fake(
                batch_size=batch_size,
                seq_length_5=configuration.input_data.default_seq_length_5_minutes,
            ),
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

    @staticmethod
    def load_netcdf(local_netcdf_path: Union[Path, str], batch_idx: int):
        """Load batch from netcdf file"""
        data_sources_names = Example.__fields__.keys()

        # set up futures executor
        batch_dict = {}
        with futures.ThreadPoolExecutor() as executor:
            future_examples_per_source = []

            # loop over data sources
            for data_source_name in data_sources_names:

                local_netcdf_filename = os.path.join(
                    local_netcdf_path, data_source_name, f"{batch_idx}.nc"
                )

                # submit task
                future_examples = executor.submit(
                    xr.load_dataset,
                    filename_or_obj=local_netcdf_filename,
                )
                future_examples_per_source.append([data_source_name, future_examples])

        # Collect results from each thread.
        for data_source_name, future_examples in future_examples_per_source:
            xr_dataset = future_examples.result()

            batch_dict[data_source_name] = xr_dataset

        batch_dict["batch_size"] = len(batch_dict["metadata"].example)

        return Batch(**batch_dict)


class Example(BaseModel):
    """
    Single Data item

    Note that this is currently not really used
    """

    metadata: Optional[Metadata]
    satellite: Optional[Satellite]
    topographic: Optional[Topographic]
    pv: Optional[PV]
    sun: Optional[Sun]
    gsp: Optional[GSP]
    nwp: Optional[NWP]
    datetime: Optional[Datetime]

    @property
    def data_sources(self):
        """The different data sources"""
        return [
            self.satellite,
            self.topographic,
            self.pv,
            self.sun,
            self.gsp,
            self.nwp,
            self.datetime,
            self.metadata,
        ]
