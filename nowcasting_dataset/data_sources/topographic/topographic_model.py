""" Model for Topogrpahic features """
from pydantic import Field, validator
import xarray as xr
import numpy as np
import logging
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutputML, DataSourceOutput
from nowcasting_dataset.consts import Array

from nowcasting_dataset.consts import TOPOGRAPHIC_DATA, TOPOGRAPHIC_X_COORDS, TOPOGRAPHIC_Y_COORDS
from nowcasting_dataset.utils import coord_to_range
from nowcasting_dataset.dataset.xr_utils import from_list_data_array_to_batch_dataset
from nowcasting_dataset.dataset.pydantic_xr import PydanticXArrayDataSet

logger = logging.getLogger(__name__)


class Topographic(DataSourceOutput):
    """ Class to store topographic data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data
    __slots__ = []

    # todo add validation here

    @staticmethod
    def fake(batch_size, image_size_pixels):
        """ Create fake data """
        # make batch of arrays
        xr_arrays = [
            xr.DataArray(
                data=np.random.randn(
                    image_size_pixels,
                    image_size_pixels,
                ),
                dims=["x", "y"],
                coords=dict(
                    x=np.sort(np.random.randn(image_size_pixels)),
                    y=np.sort(np.random.randn(image_size_pixels))[::-1].copy(),
                ),
            )
            for _ in range(batch_size)
        ]

        # make dataset
        xr_dataset = from_list_data_array_to_batch_dataset(xr_arrays)

        return Topographic(xr_dataset)


class TopographicML(DataSourceOutputML):
    """
    Topographic/elevation map features.
    """

    # Shape: [batch_size,] width, height
    topo_data: Array = Field(
        ...,
        description="Elevation map of the area covered by the satellite data. "
        "Shape: [batch_size], width, height",
    )
    topo_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the topographic images. Shape: [batch_size,] width",
    )
    topo_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the topographic images. Shape: [batch_size,] height",
    )

    @property
    def height(self):
        """ The height of the topographic image """
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
    def fake(batch_size, image_size_pixels):
        """ Create fake data """
        return TopographicML(
            batch_size=batch_size,
            topo_data=np.random.randn(
                batch_size,
                image_size_pixels,
                image_size_pixels,
            ),
            topo_x_coords=np.sort(np.random.randn(batch_size, image_size_pixels)),
            topo_y_coords=np.sort(np.random.randn(batch_size, image_size_pixels))[:, ::-1].copy(),
            # copy is needed as torch doesnt not support negative strides
        )

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
        if TOPOGRAPHIC_DATA in xr_dataset.keys():
            return TopographicML(
                batch_size=xr_dataset[TOPOGRAPHIC_DATA].shape[0],
                topo_data=xr_dataset[TOPOGRAPHIC_DATA],
                topo_x_coords=xr_dataset[TOPOGRAPHIC_DATA].topo_x,
                topo_y_coords=xr_dataset[TOPOGRAPHIC_DATA].topo_y,
            )
        else:
            return None
