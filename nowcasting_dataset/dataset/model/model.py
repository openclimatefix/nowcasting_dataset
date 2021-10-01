from pydantic import BaseModel, Field
from typing import List
import xarray as xr

from nowcasting_dataset.dataset.model.satellite import Satellite
from nowcasting_dataset.dataset.model.topographic import Topographic
from nowcasting_dataset.dataset.model.pv import PV
from nowcasting_dataset.dataset.model.sun import Sun
from nowcasting_dataset.dataset.model.gsp import GSP
from nowcasting_dataset.dataset.model.nwp import NWP
from nowcasting_dataset.dataset.model.datetime import Datetime

from nowcasting_dataset.config.model import Configuration


class Batch(BaseModel):

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    satellite: Satellite
    topographic: Topographic
    pv: PV
    sun: Sun
    gsp: GSP
    nwp: NWP
    datetime: Datetime

    def change_type_to_xr_data_array(self):
        pass
        # go other datasoruces and change them to xr data arrays

    def change_type_to_numpy(self):

        self.satellite = self.satellite.to_numpy()
        self.topographic = self.topographic.to_numpy()
        self.pv = self.pv.to_numpy()
        self.sun = self.sun.to_numpy()
        self.gsp = self.gsp.to_numpy()
        self.nwp = self.nwp.to_numpy()
        self.datetime = self.datetime.to_numpy()
        # other datasources

    def batch_to_dataset(self) -> xr.Dataset:
        """Change batch to xr.Dataset so it can be saved and compressed"""
        pass

    def load_batch_from_dataset(self, xr_dataset: xr.Dataset):
        """Change xr.Datatset to Batch object"""
        return Batch()

    @staticmethod
    def fake(configuration: Configuration = Configuration()):

        process = configuration.process

        return Batch(
            batch_size=process.batch_size,
            satellite=Satellite.fake(
                process.batch_size,
                process.seq_len_5_minutes,
                process.satellite_image_size_pixels,
                len(process.nwp_channels),
            ),
            topographic=Topographic.fake(
                batch_size=process.batch_size,
                seq_length_5=process.seq_len_5_minutes,
                satellite_image_size_pixels=process.satellite_image_size_pixels,
            ),
            pv=PV.fake(
                batch_size=process.batch_size,
                seq_length_5=process.seq_len_5_minutes,
                n_pv_systems_per_batch=128,
            ),
            sun=Sun.fake(batch_size=process.batch_size, seq_length_5=process.seq_len_5_minutes),
            gsp=GSP.fake(
                batch_size=process.batch_size,
                seq_length_30=process.seq_len_30_minutes,
                n_gsp_per_batch=32,
            ),
            nwp=NWP.fake(
                batch_size=process.batch_size,
                seq_length_5=process.seq_len_5_minutes,
                nwp_image_size_pixels=process.nwp_image_size_pixels,
                number_nwp_channels=len(process.nwp_channels),
            ),
            datetime=Datetime.fake(
                batch_size=process.batch_size, seq_length_5=process.seq_len_5_minutes
            ),
        )


def join_data_to_batch(data=List[Batch]) -> Batch:
    """Join several single data items together to make a Batch"""
    pass
