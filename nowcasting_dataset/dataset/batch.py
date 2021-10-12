""" batch functions """
import logging
import os
from pathlib import Path
from typing import List, Optional, Union, Dict

import xarray as xr
from pydantic import BaseModel, Field

from nowcasting_dataset.filesystem.utils import make_folder

from nowcasting_dataset.config.model import Configuration

from nowcasting_dataset.data_sources.datetime.datetime_model import Datetime
from nowcasting_dataset.data_sources.metadata.metadata_model import Metadata
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.nwp.nwp_model import NWP
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.data_sources.topographic.topographic_model import Topographic
from nowcasting_dataset.time import make_random_time_vectors
from nowcasting_dataset.utils import get_netcdf_filename

_LOG = logging.getLogger(__name__)


class Example(BaseModel):
    """Single Data item"""

    metadata: Metadata
    satellite: Optional[Satellite]
    topographic: Optional[Topographic]
    pv: Optional[PV]
    sun: Optional[Sun]
    gsp: Optional[GSP]
    nwp: Optional[NWP]
    datetime: Optional[Datetime]

    def change_type_to_numpy(self):
        """Change data to numpy"""
        for data_source in self.data_sources:
            if data_source is not None:
                data_source.to_numpy()

    @property
    def data_sources(self):
        """ The different data sources """
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


class Batch(Example):
    """
    Batch data object.

    Contains the following data sources
    - gsp, satellite, topogrpahic, sun, pv, nwp and datetime.
    Also contains metadata of the class

    """

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    def batch_to_dict_dataset(self) -> Dict[str, xr.Dataset]:
        """Change batch to xr.Dataset so it can be saved and compressed"""
        return batch_to_dict_dataset(batch=self)

    @staticmethod
    def load_batch_from_dict_dataset(xr_dataset: Dict[str, xr.Dataset]):
        """Change dictionary of xr.Datatset to Batch object"""
        # get a list of data sources
        data_sources_names = Example.__fields__.keys()

        # collect data sources
        data_sources_dict = {}
        for data_source_name in data_sources_names:
            cls = Example.__fields__[data_source_name].type_
            data_sources_dict[data_source_name] = cls.from_xr_dataset(
                xr_dataset=xr_dataset[data_source_name]
            )

        data_sources_dict["batch_size"] = data_sources_dict["metadata"].batch_size

        return Batch(**data_sources_dict)

    def split(self) -> List[Example]:
        """Split batch into list of data items"""
        # collect split data
        split_data_dict = {}
        for data_source in self.data_sources:
            if data_source is not None:
                cls = data_source.__class__.__name__.lower()
                split_data_dict[cls] = data_source.split()

        # make in to Example objects
        data_items = []
        for batch_idx in range(self.batch_size):
            split_data_one_example_dict = {k: v[batch_idx] for k, v in split_data_dict.items()}
            data_items.append(Example(**split_data_one_example_dict))

        return data_items

    @staticmethod
    def fake(configuration: Configuration = Configuration()):
        """Create fake batch"""
        process = configuration.process

        t0_dt, time_5, time_30 = make_random_time_vectors(
            batch_size=process.batch_size,
            seq_len_5_minutes=process.seq_len_5_minutes,
            seq_len_30_minutes=process.seq_len_30_minutes,
        )

        return Batch(
            batch_size=process.batch_size,
            metadata=Metadata.fake(batch_size=process.batch_size, t0_dt=t0_dt),
            satellite=Satellite.fake(
                process.batch_size,
                process.seq_len_5_minutes,
                process.satellite_image_size_pixels,
                len(process.sat_channels),
                time_5=time_5,
            ),
            topographic=Topographic.fake(
                batch_size=process.batch_size,
                satellite_image_size_pixels=process.satellite_image_size_pixels,
            ),
            pv=PV.fake(
                batch_size=process.batch_size,
                seq_length_5=process.seq_len_5_minutes,
                n_pv_systems_per_batch=128,
                time_5=time_5,
            ),
            sun=Sun.fake(batch_size=process.batch_size, seq_length_5=process.seq_len_5_minutes),
            gsp=GSP.fake(
                batch_size=process.batch_size,
                seq_length_30=process.seq_len_30_minutes,
                n_gsp_per_batch=32,
                time_30=time_30,
            ),
            nwp=NWP.fake(
                batch_size=process.batch_size,
                seq_length_5=process.seq_len_5_minutes,
                nwp_image_size_pixels=process.nwp_image_size_pixels,
                number_nwp_channels=len(process.nwp_channels),
                time_5=time_5,
            ),
            datetime=Datetime.fake(
                batch_size=process.batch_size, seq_length_5=process.seq_len_5_minutes
            ),
        )

    def save_netcdf(self, batch_i: int, path: Path):
        """
        Save batch to netcdf file

        Args:
            batch_i: the batch id, used to make the filename
            path: the path where it will be saved. This can be local or in the cloud.

        """
        batch_xr = self.batch_to_dict_dataset()

        for data_source in self.data_sources:
            xr_dataset = batch_xr[data_source.get_name()]
            data_source.save_netcdf(batch_i=batch_i, path=path, xr_dataset=xr_dataset)

    @staticmethod
    def load_netcdf(local_netcdf_path: Union[Path, str], batch_idx: int):
        """Load batch from netcdf file"""
        data_sources_names = Example.__fields__.keys()

        # collect data sources
        batch_dict = {}
        for data_source_name in data_sources_names:

            local_netcdf_filename = os.path.join(
                local_netcdf_path, data_source_name, f"{batch_idx}.nc"
            )
            xr_dataset = xr.load_dataset(local_netcdf_filename)

            batch_dict[data_source_name] = xr_dataset

        return Batch.load_batch_from_dict_dataset(batch_dict)


def batch_to_dict_dataset(batch: Batch) -> Dict[str, xr.Dataset]:
    """Concat all the individual fields in an Example into a dictionary of Datasets.

    Args:
      batch: List of Example objects, which together constitute a single batch.
    """
    individual_datasets = {}
    split_batch = batch.split()

    # loop over each data source
    for data_source in split_batch[0].data_sources:

        datasets = []
        name = data_source.get_name()

        # loop over each item in the batch
        for i, example in enumerate(split_batch):

            if data_source is not None:
                datasets.append(getattr(split_batch[i], name).to_xr_dataset(i))

        # Merge
        merged_ds = xr.concat(datasets, dim="example")
        individual_datasets[name] = merged_ds

    return individual_datasets


def write_batch_locally(batch: Union[Batch, dict], batch_i: int, path: Path):
    """
    Write a batch to a locally file

    Args:
        batch: A batch of data
        batch_i: The number of the batch
        path: The directory to write the batch into.
    """
    if type(batch):
        batch = Batch(**batch)

    batch.save_netcdf(batch_i=batch_i, path=path)
