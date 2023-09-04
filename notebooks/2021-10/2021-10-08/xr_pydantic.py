from typing import Union

import numpy as np
import torch
import xarray as xr
from pydantic import BaseModel, Field, validator

Array = Union[xr.DataArray, np.ndarray, torch.Tensor]


class Satellite(BaseModel):
    # Shape: [batch_size,] seq_length, width, height, channel
    image_data: xr.DataArray = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("image_data")
    def v_image_data(cls, v):
        print("validating image data")
        return v


class Batch(BaseModel):
    batch_size: int = 0
    satellite: Satellite

    @validator("batch_size")
    def v_image_data(cls, v):
        print("validating batch size")
        return v


s = Satellite(image_data=xr.DataArray())
s_dict = s.dict()

x = Satellite(**s_dict)
x = Satellite.construct(Satellite.__fields_set__, **s_dict)


batch = Batch(batch_size=5, satellite=s)

b_dict = batch.dict()

x = Batch(**b_dict)
x = Batch.construct(Batch.__fields_set__, **b_dict)


# class Satellite(BaseModel):
#
#     image_data: xr.DataArray
#
#     # validate
#
#     def to_dataset(self):
#         pass
#
#     def from_dateset(self):
#         pass
#
#     def to_numpy(self) -> SatelliteNumpy:
#         pass
#
#
# class SatelliteNumpy(BaseModel):
#
#     image_data: np.ndarray
#     x: np.ndarray
#     # more
#
#
# class Example(BaseModel):
#
#     satelllite: Satellite
#     # more
#
#
# class Batch(BaseModel):
#
#     batch_size: int = 0
#     examples: List[Example]
#
#     def to/from_netcdf():
#         pass
#
#
# class BatchNumpy(BaseModel):
#
#     batch_size: int = 0
#     satellite: SatellliteNumpy
#     # more
#
#     def from_batch(self) -> BatchNumpy:
#         """ change to Batch numpy structure """
