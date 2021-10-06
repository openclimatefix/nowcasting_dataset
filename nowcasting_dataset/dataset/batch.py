""" batch functions """
import logging
from pathlib import Path
from typing import List, Optional

import xarray as xr
from pydantic import BaseModel, Field

from nowcasting_dataset.config.model import Configuration

# from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.data_sources.datetime.datetime_model import Datetime
from nowcasting_dataset.data_sources.general.general_model import General
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.data_sources.nwp.nwp_model import NWP
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.data_sources.topographic.topographic_model import Topographic
from nowcasting_dataset.time import make_time_vectors
from nowcasting_dataset.utils import get_netcdf_filename

_LOG = logging.getLogger(__name__)

# TODO
# def write_batch_locally(batch: List[Example], batch_i: int, path: Path):
#     """
#     Write a batch to a locally file
#
#     Args:
#         batch: A batch of data
#         batch_i: The number of the batch
#         path: The directory to write the batch into.
#     """
#     dataset = batch_to_dataset(batch)
#     dataset = fix_dtypes(dataset)
#     encoding = {name: {"compression": "lzf"} for name in dataset.data_vars}
#     filename = get_netcdf_filename(batch_i)
#     local_filename = path / filename
#     dataset.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)
#


# def fix_dtypes(concat_ds):
#     """
#     TODO put these into the specific models
#     """
#     ds_dtypes = {
#         "example": np.int32,
#         "sat_x_coords": np.int32,
#         "sat_y_coords": np.int32,
#         "nwp": np.float32,
#         "nwp_x_coords": np.float32,
#         "nwp_y_coords": np.float32,
#         "pv_system_id": np.float32,
#         "pv_system_row_number": np.float32,
#         "pv_system_x_coords": np.float32,
#         "pv_system_y_coords": np.float32,
#         GSP_YIELD: np.float32,
#         GSP_ID: np.float32,
#         GSP_X_COORDS: np.float32,
#         GSP_Y_COORDS: np.float32,
#         TOPOGRAPHIC_X_COORDS: np.float32,
#         TOPOGRAPHIC_Y_COORDS: np.float32,
#         TOPOGRAPHIC_DATA: np.float32,
#     }
#
#     for name, dtype in ds_dtypes.items():
#         concat_ds[name] = concat_ds[name].astype(dtype)
#
#     assert concat_ds["sat_data"].dtype == np.int16
#     return concat_ds


class DataItem(BaseModel):

    general: General
    satellite: Optional[Satellite]
    topographic: Optional[Topographic]
    pv: Optional[PV]
    sun: Optional[Sun]
    gsp: Optional[GSP]
    nwp: Optional[NWP]
    datetime: Optional[Datetime]

    def change_type_to_numpy(self):

        self.satellite.to_numpy()
        self.topographic.to_numpy()
        self.pv.to_numpy()
        self.sun.to_numpy()
        self.gsp.to_numpy()
        self.nwp.to_numpy()
        self.datetime.to_numpy()
        self.general.to_numpy()

        # other datasources


class Batch(DataItem):

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    def batch_to_dataset(self) -> xr.Dataset:
        """Change batch to xr.Dataset so it can be saved and compressed"""

        return batch_to_dataset(batch=self)

    @staticmethod
    def load_batch_from_dataset(xr_dataset: xr.Dataset):
        """Change xr.Datatset to Batch object"""

        gsp = GSP.from_xr_dataset(xr_dataset=xr_dataset)
        satellite = Satellite.from_xr_dataset(xr_dataset=xr_dataset)
        topographic = Topographic.from_xr_dataset(xr_dataset=xr_dataset)
        sun = Sun.from_xr_dataset(xr_dataset=xr_dataset)
        pv = PV.from_xr_dataset(xr_dataset=xr_dataset)
        nwp = NWP.from_xr_dataset(xr_dataset=xr_dataset)
        datetime = Datetime.from_xr_dataset(xr_dataset=xr_dataset)
        general = General.from_xr_dataset(xr_dataset=xr_dataset)

        batch_size = general.batch_size

        return Batch(
            batch_size=batch_size,
            gsp=gsp,
            satellite=satellite,
            topographic=topographic,
            sun=sun,
            pv=pv,
            nwp=nwp,
            datetime=datetime,
            general=general,
        )

    def split(self) -> List[DataItem]:

        satellite_data = self.satellite.split()
        topographic_data = self.topographic.split()
        pv_data = self.pv.split()
        sun_data = self.sun.split()
        gsp_data = self.gsp.split()
        nwp_data = self.nwp.split()
        datetime_data = self.datetime.split()
        general_data = self.general.split()

        data_items = []
        for batch_idx in range(self.batch_size):
            data_items.append(
                DataItem(
                    general=general_data[batch_idx],
                    satellite=satellite_data[batch_idx],
                    topographic=topographic_data[batch_idx],
                    pv=pv_data[batch_idx],
                    sun=sun_data[batch_idx],
                    gsp=gsp_data[batch_idx],
                    nwp=nwp_data[batch_idx],
                    datetime=datetime_data[batch_idx],
                )
            )

        return data_items

    @staticmethod
    def fake(configuration: Configuration = Configuration()):

        process = configuration.process

        t0_dt, time_5, time_30 = make_time_vectors(
            batch_size=process.batch_size,
            seq_len_5_minutes=process.seq_len_5_minutes,
            seq_len_30_minutes=process.seq_len_30_minutes,
        )

        return Batch(
            batch_size=process.batch_size,
            general=General.fake(batch_size=process.batch_size, t0_dt=t0_dt),
            satellite=Satellite.fake(
                process.batch_size,
                process.seq_len_5_minutes,
                process.satellite_image_size_pixels,
                len(process.nwp_channels),
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

        batch_xr = self.batch_to_dataset()

        encoding = {name: {"compression": "lzf"} for name in batch_xr.data_vars}
        filename = get_netcdf_filename(batch_i)
        local_filename = path / filename
        batch_xr.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)

    @staticmethod
    def load_netcdf(local_netcdf_filename: Path):

        netcdf_batch = xr.load_dataset(local_netcdf_filename)

        return Batch.load_batch_from_dataset(netcdf_batch)


def batch_to_dataset(batch: Batch) -> xr.Dataset:
    """Concat all the individual fields in an Example into a single Dataset.

    Args:
      batch: List of Example objects, which together constitute a single batch.
    """
    datasets = []

    # loop over each item in the batch
    for i, example in enumerate(batch.split()):

        individual_datasets = []

        individual_datasets.append(example.satellite.to_xr_dataset())
        individual_datasets.append(example.nwp.to_xr_dataset())
        individual_datasets.append(example.datetime.to_xr_dataset())
        individual_datasets.append(example.topographic.to_xr_dataset())
        individual_datasets.append(example.sun.to_xr_dataset())
        individual_datasets.append(example.pv.to_xr_dataset(i))
        individual_datasets.append(example.gsp.to_xr_dataset(i))
        individual_datasets.append(example.general.to_xr_dataset(i))

        # Merge
        merged_ds = xr.merge(individual_datasets)
        datasets.append(merged_ds)

    return xr.concat(datasets, dim="example")
