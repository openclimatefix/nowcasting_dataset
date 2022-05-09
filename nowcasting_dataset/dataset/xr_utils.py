""" Useful functions for xarray objects

1. joining data arrays to datasets
2. pydantic exentions model of xr.Dataset
"""
from typing import Any, List

import numpy as np
import xarray as xr


def join_list_dataset_to_batch_dataset(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Join a list of data sets to a dataset by expanding dims"""

    new_datasets = []
    for i, dataset in enumerate(datasets):
        new_dataset = dataset.expand_dims(dim="example").assign_coords(example=("example", [i]))
        new_datasets.append(new_dataset)

    joined_dataset = xr.concat(new_datasets, dim="example")

    # format example index
    joined_dataset.__setitem__("example", joined_dataset.example.astype("int32"))

    return joined_dataset


def convert_coordinates_to_indexes_for_list_datasets(
    examples: List[xr.Dataset],
) -> List[xr.Dataset]:
    """Set the coords to be indices before joining into a batch"""
    return [convert_coordinates_to_indexes(example) for example in examples]


def convert_coordinates_to_indexes(dataset: xr.Dataset) -> xr.Dataset:
    """Reindex dims so that it can be merged with batch.

    For each dimension in dataset, change the coords to 0.. len(original_coords),
    and append "_index" to the dimension name.
    And save the original coordinates in `original_dim_name`.

    This is useful to align multiple examples into a single batch.
    """

    assert type(dataset) == xr.Dataset, f" Should be xr.Dataset but found {type(dataset)}"

    original_dim_names = dataset.dims

    for original_dim_name in original_dim_names:
        original_coords = dataset[original_dim_name]
        new_index_coords = np.arange(len(original_coords)).astype("int32")
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
    _expected_data_vars = ()  # Subclasses should set this.

    # xarray doesnt support sub classing at the moment: https://github.com/pydata/xarray/issues/3980
    __slots__ = ()

    @classmethod
    def model_validation(cls, v):
        """Specific model validation, to be overwritten by class"""
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
        v = cls.validate_data_vars(v)
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
            " Note that '_index' is removed, and 'example' is ignored, and the order is ignored"
        )
        return v

    @classmethod
    def validate_coords(cls, v: Any) -> Any:
        """Validate the coords"""
        for dim in cls._expected_dimensions:
            coord = v.coords[f"{dim}_index"]
            assert len(coord) > 0, f"{dim}_index is empty in {cls.__name__}!"
        return v

    @classmethod
    def validate_data_vars(cls, v: Any) -> Any:
        """Validate the data vars"""

        data_var_names = v.data_vars
        for data_var in cls._expected_data_vars:
            assert (
                data_var in data_var_names
            ), f"{data_var} is not in all data_vars ({data_var_names}) in {cls.__name__}!"
        return v


def convert_arrays_to_uint8(*arrays: tuple[np.ndarray]) -> tuple[np.ndarray]:
    """Convert multiple arrays to uint8, using the same min and max to scale all arrays."""
    # First, stack into a single numpy array so we can work on all images at the same time:
    stacked = np.stack(arrays)

    # Convert to float64 for normalisation:
    stacked = stacked.astype(np.float64)

    # Rescale pixel values to be in the range [0, 1]:
    stacked -= stacked.min()
    stacked_max = stacked.max()
    if stacked_max > 0.0:
        # If there is still an invalid value then we want to know about it!
        # Adapted from https://stackoverflow.com/a/33701974/732596
        with np.errstate(all="raise"):
            stacked /= stacked.max()

    # Convert to uint8 (uint8 can represent integers in the range [0, 255]):
    stacked *= 255
    stacked = stacked.round()
    stacked = stacked.astype(np.uint8)

    return tuple(stacked)
