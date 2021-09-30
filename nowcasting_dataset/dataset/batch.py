from typing import List, Optional, Union
import logging

import numpy as np
import xarray as xr
from pathlib import Path

from nowcasting_dataset.consts import (
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    GSP_DATETIME_INDEX,
    DATETIME_FEATURE_NAMES,
    T0_DT,
    TOPOGRAPHIC_DATA,
    TOPOGRAPHIC_X_COORDS,
    TOPOGRAPHIC_Y_COORDS,
)

from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.utils import get_netcdf_filename

_LOG = logging.getLogger(__name__)


def write_batch_locally(batch: List[Example], batch_i: int, path: Path):
    """
    Write a batch to a locally file
    Args:
        batch: A batch of data
        batch_i: The number of the batch
        path: The directory to write the batch into.
    """
    dataset = batch_to_dataset(batch)
    dataset = fix_dtypes(dataset)
    encoding = {name: {"compression": "lzf"} for name in dataset.data_vars}
    filename = get_netcdf_filename(batch_i)
    local_filename = path / filename
    dataset.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)


def fix_dtypes(concat_ds):
    """
    TODO
    """
    ds_dtypes = {
        "example": np.int32,
        "sat_x_coords": np.int32,
        "sat_y_coords": np.int32,
        "nwp": np.float32,
        "nwp_x_coords": np.float32,
        "nwp_y_coords": np.float32,
        "pv_system_id": np.float32,
        "pv_system_row_number": np.float32,
        "pv_system_x_coords": np.float32,
        "pv_system_y_coords": np.float32,
        GSP_YIELD: np.float32,
        GSP_ID: np.float32,
        GSP_X_COORDS: np.float32,
        GSP_Y_COORDS: np.float32,
        TOPOGRAPHIC_X_COORDS: np.float32,
        TOPOGRAPHIC_Y_COORDS: np.float32,
        TOPOGRAPHIC_DATA: np.float32,
    }

    for name, dtype in ds_dtypes.items():
        concat_ds[name] = concat_ds[name].astype(dtype)

    assert concat_ds["sat_data"].dtype == np.int16
    return concat_ds


def batch_to_dataset(batch: List[Example]) -> xr.Dataset:
    """Concat all the individual fields in an Example into a single Dataset.

    Args:
      batch: List of Example objects, which together constitute a single batch.
    """
    datasets = []
    for i, example in enumerate(batch):
        try:
            individual_datasets = []
            example_dim = {"example": np.array([i], dtype=np.int32)}
            for name in ["sat_data", "nwp"]:
                ds = example[name].to_dataset(name=name)
                short_name = name.replace("_data", "")
                if name == "nwp":
                    ds = ds.rename({"target_time": "time"})
                for dim in ["time", "x", "y"]:
                    ds = coord_to_range(ds, dim, prefix=short_name)
                ds = ds.rename(
                    {
                        "variable": f"{short_name}_variable",
                        "x": f"{short_name}_x",
                        "y": f"{short_name}_y",
                    }
                )
                individual_datasets.append(ds)

            # Datetime features
            for name in DATETIME_FEATURE_NAMES:
                ds = example[name].rename(name).to_xarray().to_dataset().rename({"index": "time"})
                ds = coord_to_range(ds, "time", prefix=None)
                individual_datasets.append(ds)

            # PV
            one_dataset = xr.DataArray(example["pv_yield"], dims=["time", "pv_system"])
            one_dataset = one_dataset.to_dataset(name="pv_yield")
            n_pv_systems = len(example["pv_system_id"])

            # GSP
            n_gsp = len(example[GSP_ID])
            one_dataset[GSP_YIELD] = xr.DataArray(example[GSP_YIELD], dims=["time_30", "gsp"])
            one_dataset[GSP_DATETIME_INDEX] = xr.DataArray(
                example[GSP_DATETIME_INDEX],
                dims=["time_30"],
                coords=[np.arange(len(example[GSP_DATETIME_INDEX]))],
            )

            # Topographic
            ds = example[TOPOGRAPHIC_DATA].to_dataset(name=TOPOGRAPHIC_DATA)
            topo_name = "topo"
            for dim in ["x", "y"]:
                ds = coord_to_range(ds, dim, prefix=topo_name)
            ds = ds.rename(
                {
                    "x": f"{topo_name}_x",
                    "y": f"{topo_name}_y",
                }
            )
            individual_datasets.append(ds)

            # This will expand all dataarrays to have an 'example' dim.
            # 0D
            for name in ["x_meters_center", "y_meters_center", T0_DT]:
                try:
                    one_dataset[name] = xr.DataArray(
                        [example[name]], coords=example_dim, dims=["example"]
                    )
                except Exception as e:
                    _LOG.error(
                        f"Could not make pv_yield data for {name} with example_dim={example_dim}"
                    )
                    if name not in example.keys():
                        _LOG.error(f"{name} not in data keys: {example.keys()}")
                    _LOG.error(e)
                    raise Exception

            # 1D
            for name in [
                "pv_system_id",
                "pv_system_row_number",
                "pv_system_x_coords",
                "pv_system_y_coords",
            ]:
                one_dataset[name] = xr.DataArray(
                    example[name][None, :],
                    coords={
                        **example_dim,
                        **{"pv_system": np.arange(n_pv_systems, dtype=np.int32)},
                    },
                    dims=["example", "pv_system"],
                )

            # GSP
            for name in [GSP_ID, GSP_X_COORDS, GSP_Y_COORDS]:
                try:
                    one_dataset[name] = xr.DataArray(
                        example[name][None, :],
                        coords={
                            **example_dim,
                            **{"gsp": np.arange(n_gsp, dtype=np.int32)},
                        },
                        dims=["example", "gsp"],
                    )
                except Exception as e:
                    _LOG.debug(f"Could not add {name} to dataset. {example[name].shape}")
                    _LOG.error(e)
                    raise e

            individual_datasets.append(one_dataset)

            # Merge
            merged_ds = xr.merge(individual_datasets)
            datasets.append(merged_ds)

        except Exception as e:
            print(e)
            _LOG.error(e)
            raise Exception

    return xr.concat(datasets, dim="example")


def coord_to_range(
    da: xr.DataArray, dim: str, prefix: Optional[str], dtype=np.int32
) -> xr.DataArray:
    # TODO: Actually, I think this is over-complicated?  I think we can
    # just strip off the 'coord' from the dimension.
    coord = da[dim]
    da[dim] = np.arange(len(coord), dtype=dtype)
    if prefix is not None:
        da[f"{prefix}_{dim}_coords"] = xr.DataArray(coord, coords=[da[dim]], dims=[dim])
    return da
