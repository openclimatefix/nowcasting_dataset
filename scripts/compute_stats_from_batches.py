#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# In[2]:


BASE_PATH = Path(
    "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/prepared_ML_training_data/v15/train"
)
DATA_SOURCE_NAMES = ("gsp", "hrvsatellite", "nwp", "pv", "satellite", "sun", "topographic")


# In[23]:


def compute_accumulators(data_array: xr.DataArray) -> pd.DataFrame:
    dims_to_aggregate_over = set(data_array.dims) - set(["channels_index"])
    data_array = data_array.astype(np.float64)  # Minimise numerical instability.
    _count = data_array.count(dim=dims_to_aggregate_over).to_series()
    _sum = data_array.sum(dim=dims_to_aggregate_over).to_series()
    _sum_of_squares = (data_array ** 2).sum(dim=dims_to_aggregate_over).to_series()
    return pd.DataFrame(
        {
            "count": _count,
            "sum": _sum.astype(np.float128),
            "sum_of_squares": _sum_of_squares.astype(np.float128),
        }
    )


# In[24]:


def compute_std(accumulators: pd.DataFrame):
    return np.sqrt(
        (
            accumulators["count"] * accumulators["sum_of_squares"]
            - accumulators["sum"] * accumulators["sum"]
        )
        / (accumulators["count"] * (accumulators["count"] - 1))
    )


# In[30]:


def compute_mean(accumulators: pd.DataFrame):
    return accumulators["sum"] / accumulators["count"]


# In[25]:


def load_and_check_batch(filename: Path) -> pd.DataFrame:
    """Loads a batch NetCDF file. Computes stats. Returns pd.Series mapping stat name to stat value."""
    dataset = xr.load_dataset(filename, mode="r")
    data_array = dataset["data"]

    # Validation checks:
    assert np.isfinite(data_array).all()
    # assert (data_array >= 0).all()
    # assert (data_array <= 1023).all()

    # Compute accumulators for standard deviation and mean:
    return compute_accumulators(data_array)


# In[26]:


def run_on_all_files():
    filenames = (BASE_PATH / "nwp").glob("*.nc")
    filenames = np.sort(list(filenames))[:10]
    n = len(filenames)
    print(n, "filenames found")
    accumulators = None
    for i, filename in enumerate(filenames):
        print(f"{i+1:5,d}/{n:5,d}: {filename}\r", flush=True, end="")
        accumulators_for_filename = load_and_check_batch(filename)
        if accumulators is None:
            accumulators = accumulators_for_filename
        else:
            accumulators += accumulators_for_filename

    return accumulators


accumulators = run_on_all_files()


# In[38]:


accumulators.to_csv("accumulators.csv")


# In[39]:


compute_std(accumulators).to_csv("std.csv")


# In[40]:


compute_mean(accumulators).to_csv("mean.csv")


# In[ ]:
