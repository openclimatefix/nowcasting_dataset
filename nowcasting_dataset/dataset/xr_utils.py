""" Useful functions for xarray objects

1. joining data arrays to datasets
2. pydantic exentions model of xr.Dataset
"""
from typing import Any, List

import numpy as np
import xarray as xr


# TODO: This function is only used in fake.py for testing.
# Maybe we should move this function to fake.py?
def join_list_data_array_to_batch_dataset(data_arrays: List[xr.DataArray]) -> xr.Dataset:
    """Join a list of xr.DataArrays into an xr.Dataset by concatenating on the example dim."""
    datasets = [convert_data_array_to_dataset(data_arrays[i]) for i in range(len(data_arrays))]

    return join_list_dataset_to_batch_dataset(datasets)


def join_list_dataset_to_batch_dataset(datasets: list[xr.Dataset]) -> xr.Dataset:
    """ Join a list of data sets to a dataset by expanding dims """

    new_datasets = []
    for i, dataset in enumerate(datasets):
        new_dataset = dataset.expand_dims(dim="example").assign_coords(example=("example", [i]))
        new_datasets.append(new_dataset)

    return xr.concat(new_datasets, dim="example")


# TODO: Issue #318: Maybe remove this function and, in calling code, do data_array.to_dataset()
# followed by make_dim_index, to make it more explicit what's happening?  At the moment,
# in the calling code, it's not clear that the coordinates are being changed.
def convert_data_array_to_dataset(data_xarray: xr.DataArray) -> xr.Dataset:
    """ Convert data array to dataset. Reindex dim so that it can be merged with batch"""
    data = xr.Dataset({"data": data_xarray})
    return make_dim_index(dataset=data)


# TODO: Issue #318: Maybe rename this function... maybe to coord_to_range()?
# Not sure what's best right now!  :)
def make_dim_index(dataset: xr.Dataset) -> xr.Dataset:
    """Reindex dims so that it can be merged with batch.

    For each dimension in dataset, change the coords to 0.. len(original_coords),
    and append "_index" to the dimension name.
    And save the original coordinates in `original_dim_name`.

    This is useful to align multiple examples into a single batch.
    """

    original_dim_names = dataset.dims

    for original_dim_name in original_dim_names:
        original_coords = dataset[original_dim_name]
        new_index_coords = np.arange(len(original_coords))
        new_index_dim_name = f"{original_dim_name}_index"
        dataset[original_dim_name] = new_index_coords
        dataset = dataset.rename({original_dim_name: new_index_dim_name})
        # Save the original_coords back into dataset, but this time it won't be used as
        # coords for the variables payload in the dataset.
        dataset[original_dim_name] = xr.DataArray(
            original_coords,
            coords=[new_index_coords],
            dims=[new_index_dim_name],
        )

    return dataset


class PydanticXArrayDataSet(xr.Dataset):
    """Pydantic Xarray Dataset Class

    Adapted from https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__

    """

    _expected_dimensions = ()  # Subclasses should set this.

    # xarray doesnt support sub classing at the moment: https://github.com/pydata/xarray/issues/3980
    __slots__ = ()

    @classmethod
    def model_validation(cls, v):
        """ Specific model validation, to be overwritten by class """
        return v

    @classmethod
    def __get_validators__(cls):
        """Get validators"""
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> Any:
        """Do validation"""
        v = cls.validate_dims(v)
        v = cls.validate_coords(v)
        v = cls.model_validation(v)
        return v

    @classmethod
    def validate_dims(cls, v: Any) -> Any:
        """Validate the dims"""
        assert all(
            dim.replace("_index", "") in cls._expected_dimensions
            for dim in v.dims
            if dim != "example"
        ), (
            f"{cls.__name__}.dims is wrong! "
            f"{cls.__name__}.dims is {v.dims}. "
            f"But we expected {cls._expected_dimensions}."
            " Note that '_index' is removed, and 'example' is ignored"
        )
        return v

    @classmethod
    def validate_coords(cls, v: Any) -> Any:
        """Validate the coords"""
        for dim in cls._expected_dimensions:
            coord = v.coords[f"{dim}_index"]
            assert len(coord) > 0, f"{dim}_index is empty in {cls.__name__}!"
        return v
