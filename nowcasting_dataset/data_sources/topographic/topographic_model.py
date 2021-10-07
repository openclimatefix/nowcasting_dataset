""" Model for Topogrpahic features """
from pydantic import Field, validator
import xarray as xr
import numpy as np
import torch
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.consts import Array

from nowcasting_dataset.consts import TOPOGRAPHIC_DATA, TOPOGRAPHIC_X_COORDS, TOPOGRAPHIC_Y_COORDS
from nowcasting_dataset.utils import coord_to_range


class Topographic(DataSourceOutput):
    """
    Topographic/elevation map features.
    """

    # Shape: [batch_size,] width, height
    topo_data: Array = Field(
        ...,
        description="Elevation map of the area covered by the satellite data. "
        "Shape: [batch_size,] seq_length, width, height",
    )
    topo_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the topographic images. Shape: [batch_size,] width",
    )
    topo_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the topographic images. Shape: [batch_size,] width",
    )

    @property
    def height(self):
        """ The width of the topographic image """
        return self.topo_data.shape[-1]

    @property
    def width(self):
        """ The width of the topographic image """
        return self.topo_data.shape[-2]

    @validator("topo_x_coords")
    def x_coordinates_shape(cls, v, values):
        """ Validate 'topo_x_coords' """
        assert v.shape[-1] == values["topo_data"].shape[-2]
        return v

    @validator("topo_y_coords")
    def y_coordinates_shape(cls, v, values):
        """ Validate 'topo_y_coords' """
        assert v.shape[-1] == values["topo_data"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, satellite_image_size_pixels):
        """ Create fake data """
        return Topographic(
            batch_size=batch_size,
            topo_data=torch.randn(
                batch_size,
                satellite_image_size_pixels,
                satellite_image_size_pixels,
            ),
            topo_x_coords=torch.sort(torch.randn(batch_size, satellite_image_size_pixels))[0],
            topo_y_coords=torch.sort(
                torch.randn(batch_size, satellite_image_size_pixels), descending=True
            )[0],
        )

    def to_xr_dataset(self, _):
        """ Make a xr dataset """
        data = xr.DataArray(
            self.topo_data,
            coords={
                "x": self.topo_x_coords,
                "y": self.topo_y_coords,
            },
        )

        ds = data.to_dataset(name=TOPOGRAPHIC_DATA)
        for dim in ["x", "y"]:
            ds = coord_to_range(ds, dim, prefix="topo")
        ds = ds.rename(
            {
                "x": f"topo_x",
                "y": f"topo_y",
            }
        )

        ds[TOPOGRAPHIC_DATA] = ds[TOPOGRAPHIC_DATA].astype(np.float32)
        ds[TOPOGRAPHIC_X_COORDS] = ds[TOPOGRAPHIC_X_COORDS].astype(np.float32)
        ds[TOPOGRAPHIC_Y_COORDS] = ds[TOPOGRAPHIC_Y_COORDS].astype(np.float32)

        return ds

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
        if TOPOGRAPHIC_DATA in xr_dataset.keys():
            return Topographic(
                batch_size=xr_dataset[TOPOGRAPHIC_DATA].shape[0],
                topo_data=xr_dataset[TOPOGRAPHIC_DATA],
                topo_x_coords=xr_dataset[TOPOGRAPHIC_DATA].topo_x,
                topo_y_coords=xr_dataset[TOPOGRAPHIC_DATA].topo_y,
            )
        else:
            return None
