#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import multiprocessing
import re
from pathlib import Path
from typing import Tuple, Union

import cfgrib
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr

# This notebook is research for [GitHub issue #344: Convert NWP grib files to Zarr intermediate](https://github.com/openclimatefix/nowcasting_dataset/issues/344).
#
# Useful links:
#
# * [Met Office's data docs](http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf)
#
# Known differences between the old Zarr (UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr) and the new zarr:
#
# * Images in the old zarr were top-to-bottom.  Images in the new Zarr follow the ordering in the grib files: bottom-to-top.
# * The x and y coordinates are different by 1km each.
# * The new Zarr is float16?
# * The new Zarr has a few more variables.
#
# Done:
#
# * Merge Wholesale1 and 2 (2 includes dswrf, lcc, mcc, and hcc)
# * Remove dimensions we don't care about (e.g. select temperature at 1 meter, not 0 meters)
# * Reshape images from 1D to 2D.
# * Do we really need to convert datetimes to unix epochs before appending to zarr?  If no, submit a bug report.
# * eccodes takes ages to load multiple datasets from each grib.  Maybe it'd be faster to pre-load each grib into ramdisk?  UPDATE: Nope, it's not faster!  What does give a big speedup, though, is using an idx file!
# * Reshaping is pretty slow.  Maybe go back to using `np.reshape`?
# * Experiment with when it might be fastest to load data into memory.
# * Remove `wholesale_file_number`.  And then use a `pd.Series` instead of a DF.
# * Zarr chunk size and compression.
# * Write to leonardo's SSD
# * Do we need to combine all the DataArrays into a single DataArray (with "variable" being a dimension?).  The upside is that then a single Zarr chunk can include multiple variables.  The downside is that we lose the metadata (but that's not a huge problem, maybe?)
# * Convert to float16? UPDATE: Nope, float16 results in dswrf being inf.
# * Parallelise
#
# Some outstanding questions / Todo items
#
# * Restart from last `time` in existing Zarr.
# * Convert to script
# * Use click to set source and target directories, and pattern for finding source files, and number of expected grib files per NWP init datetime.
# * Experiment with compression
# * Separately log "bad files" to a CSV file?
# * Check for NaNs.  cdcb has NaNs.
# * Fix this issue:
#
# ```
# ecCodes provides no latitudes/longitudes for gridType='transverse_mercator'
# Loading...
# Merging...
# Reshaping...
# Post-processing dataset...
# Merging...
# Reshaping...
# Post-processing dataset...
# Process ForkPoolWorker-5:
# Process ForkPoolWorker-1:
# Process ForkPoolWorker-4:
# Process ForkPoolWorker-8:
# Process ForkPoolWorker-3:
# Process ForkPoolWorker-6:
# Traceback (most recent call last):
# Traceback (most recent call last):
# Traceback (most recent call last):
# Traceback (most recent call last):
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/pool.py", line 131, in worker
#     put((job, i, result))
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/pool.py", line 131, in worker
#     put((job, i, result))
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/queues.py", line 378, in put
#     self._writer.send_bytes(obj)
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/pool.py", line 131, in worker
#     put((job, i, result))
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/queues.py", line 378, in put
#     self._writer.send_bytes(obj)
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 205, in send_bytes
#     self._send_bytes(m[offset:offset + size])
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/queues.py", line 378, in put
#     self._writer.send_bytes(obj)
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/pool.py", line 131, in worker
#     put((job, i, result))
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 205, in send_bytes
#     self._send_bytes(m[offset:offset + size])
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 409, in _send_bytes
#     self._send(header)
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 409, in _send_bytes
#     self._send(header)
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 373, in _send
#     n = write(self._handle, buf)
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 205, in send_bytes
#     self._send_bytes(m[offset:offset + size])
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/queues.py", line 378, in put
#     self._writer.send_bytes(obj)
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 373, in _send
#     n = write(self._handle, buf)
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 205, in send_bytes
#     self._send_bytes(m[offset:offset + size])
#   File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/multiprocessing/connection.py", line 409, in _send_bytes
#     self._send(header)
# BrokenPipeError: [Errno 32] Broken pipe
# ```
#
#

# In[2]:


# Define geographical domain for UKV.
# Taken from page 4 of http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf
# To quote the PDF:
# "The United Kingdom domain is a 1,096km x 1,408km ~2km resolution grid.
# The OS National Grid corners of the domain are:"

DY_METERS = DX_METERS = 2_000
NORTH = 1223_000
SOUTH = -185_000
WEST = -239_000
EAST = 857_000

# Note that the UKV NWPs y is top-to-bottom
NORTHING = np.arange(start=NORTH, stop=SOUTH, step=-DY_METERS, dtype=np.int32)
EASTING = np.arange(start=WEST, stop=EAST, step=DX_METERS, dtype=np.int32)

NUM_ROWS = len(NORTHING)
NUM_COLS = len(EASTING)


# In[7]:


# NWP_PATH = Path("/home/jack/Data/NWP/01")
# NWP_PATH = Path("/media/jack/128GB_USB/NWPs")  # Must not include trailing slash!!!
NWP_PATH = Path(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/UK_Met_Office/UKV/native"
)


# In[37]:


DST_ZARR_PATH = Path(
    "/mnt/storage_ssd/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/UK_Met_Office/UKV/zarr/test.zarr"
)


# In[8]:


assert NWP_PATH.exists()


# In[9]:


print("Getting list of all filenames...")


# In[10]:


filenames = list(NWP_PATH.glob("*/*/*/*Wholesale[12].grib"))


# In[11]:


print("Got", len(filenames), "filenames")


# In[12]:


def grib_filename_to_datetime(full_filename: Path) -> datetime.datetime:
    """Parse the grib filename and return the datetime encoded in the filename.

    Returns a datetime.
      For example, if the filename is '202101010000_u1096_ng_umqv_Wholesale1.grib',
      then the returned datetime will be datetime(year=2021, month=1, day=1, hour=0, minute=0).

    Raises RuntimeError if the filename does not match the expected regex pattern.
    """
    # Get the base_filename, which will be of the form '202101010000_u1096_ng_umqv_Wholesale1.grib'
    base_filename = full_filename.name

    # Use regex to match the year, month, day, hour, minute and wholesale_number.  i.e., group the filename like this:
    # '(2021)(01)(01)(00)(00)_u1096_ng_umqv_Wholesale(1)'.
    # A quick guide to the relevant regex operators:
    #   ^ matches the beginning of the string.
    #   () defines a group.
    #   (?P<name>...) names the group.  We can access the group with `regex_match.groupdict()[name]`.
    #   \d matches a single digit.
    #   {n} matches the preceding item n times.
    #   . matches any character.
    #   $ matches the end of the string.
    regex_pattern_string = (
        "^"  # Match the beginning of the string.
        "(?P<year>\d{4})"  # Match the year.
        "(?P<month>\d{2})"  # Match the month.
        "(?P<day>\d{2})"  # Match the day.
        "(?P<hour>\d{2})"  # Match the hour.
        "(?P<minute>\d{2})"  # Match the minute.
        "_u1096_ng_umqv_Wholesale\d\.grib$"  # Match the end of the string (escape the fullstop).
    )
    regex_pattern = re.compile(regex_pattern_string)
    regex_match = regex_pattern.match(base_filename)
    if regex_match is None:
        raise RuntimeError(
            f"Filename '{full_filename}' does not conform to expected regex pattern '{regex_pattern_string}'!"
        )

    # Convert strings to ints:
    regex_groups = {key: int(value) for key, value in regex_match.groupdict().items()}

    return datetime.datetime(**regex_groups)


# In[26]:


def decode_and_group_grib_filenames(
    filenames: list[Path], n_grib_files_per_nwp_init_time: int = 2
) -> pd.Series:
    """Returns a pd.Series where the index is the datetime of the NWP init time.

    And the values are the full_filename of each grib file.
    """
    n_filenames = len(filenames)
    nwp_init_datetimes = np.full(shape=n_filenames, fill_value=np.NaN, dtype="datetime64[ns]")
    for i, filename in enumerate(filenames):
        nwp_init_datetimes[i] = grib_filename_to_datetime(filename)

    # Swap index and values
    map_datetime_to_filename = pd.Series(
        filenames, index=nwp_init_datetimes, name="nwp_grib_filenames"
    )
    del nwp_init_datetimes
    map_datetime_to_filename = map_datetime_to_filename.sort_index()
    map_datetime_to_filename.index.name = "nwp_init_datetime_utc"

    # Select only rows where there are exactly n_grib_files_per_nwp_init_time:
    filter_func = lambda group: group.count() == n_grib_files_per_nwp_init_time
    return map_datetime_to_filename.groupby(level=0).filter(filter_func)


# In[27]:


def load_grib_file(full_filename: Union[Path, str], verbose: bool = False) -> xr.Dataset:
    """Merges and loads all contiguous xr.Datasets from the grib file.

    Removes unnecessary variables.  Picks heightAboveGround=1meter for temperature.

    Returns an xr.Dataset which has been loaded from disk.  Loading from disk at this point
    takes about 2 seconds for a 250 MB grib file, but speeds up reshape_1d_to_2d
    from about 7 seconds to 0.5 seconds :)

    Args:
      full_filename:  The full filename (including the path) of a single grib file.
      verbose:  If True then print out some useful debugging information.
    """
    # The grib files are "heterogeneous", so we use cfgrib.open_datasets
    # to return a list of contiguous xr.Datasets.
    # See https://github.com/ecmwf/cfgrib#automatic-filtering
    datasets_from_grib = cfgrib.open_datasets(full_filename)
    n_datasets = len(datasets_from_grib)

    # Get each dataset into the right shape for merging:
    # Loop round the datasets using an index (instead of `for ds in datasets_from_grib`)
    # because we will be modifying each dataset:
    for i in range(n_datasets):
        ds = datasets_from_grib[i]

        if verbose:
            print("\nDataset", i, "before processing:\n", ds, "\n")

        # For temperature, we want the temperature at 1 meter above ground,
        # not at 0 meters above ground.  The early NWPs (definitely in the 2016-03-22 NWPs),
        # heightAboveGround only has 1 entry ("1" meter above ground) and `heightAboveGround` isn't set as a dimension for `t`.
        # In later NWPs, 'heightAboveGround' has 2 values (0, 1) is a dimension for `t`.
        if "t" in ds and "heightAboveGround" in ds["t"].dims:
            ds = ds.sel(heightAboveGround=1)

        # Delete unnecessary variables.
        vars_to_delete = [
            "unknown",
            "valid_time",
            "heightAboveGround",
            "heightAboveGroundLayer",
            "atmosphere",
            "cloudBase",
            "surface",
            "meanSea",
            "level",
        ]
        for var_name in vars_to_delete:
            try:
                del ds[var_name]
            except KeyError as e:
                if verbose:
                    print("var name not in dataset:", e)
            else:
                if verbose:
                    print("Deleted", var_name)

        if verbose:
            print("\nDataset", i, "after processing:\n", ds, "\n")
            print("**************************************************")

        datasets_from_grib[i] = ds
        del ds

    merged_ds = xr.merge(datasets_from_grib)
    del datasets_from_grib  # Save memory
    print("Loading...")
    return merged_ds.load()


# In[28]:


def reshape_1d_to_2d(dataset: xr.Dataset) -> xr.Dataset:
    """Convert 1D into 2D array.

    For each `step`, the pixel values in the grib files represent a 2D image.  But, in the grib,
    the values are in a flat 1D array (indexed by the `values` dimension).
    The ordering of the pixels in the grib are left to right, bottom to top.

    We reshape every data variable at once using this trick.
    """
    # Adapted from https://stackoverflow.com/a/62667154

    # Don't reshape yet.  Instead just create new coordinates,
    # which give the `x` and `y` position of each position in the `values` dimension:
    dataset = dataset.assign_coords(
        {
            "x": ("values", np.tile(EASTING, reps=NUM_ROWS)),
            "y": ("values", np.repeat(NORTHING, repeats=NUM_COLS)),
        }
    )

    # Now set "values" to be a MultiIndex, indexed by `y` and `x`:
    dataset = dataset.set_index(values=("y", "x"))

    # Now unstack.  This gets rid of the `values` dimension and indexes
    # the data variables using `y` and `x`.
    return dataset.unstack("values")


# In[29]:


def dataset_has_variables(dataset: xr.Dataset) -> bool:
    """Return True if `dataset` has at least one variable."""
    return len(dataset.variables) > 0


# In[30]:


def post_process_dataset(dataset: xr.Dataset) -> xr.Dataset:
    print("Post-processing dataset...")
    return (
        dataset.to_array(dim="variable", name="UKV")
        .to_dataset()
        # .astype(np.float16)
        .rename({"time": "init_time"})
        .chunk(
            {
                "init_time": 1,
                "step": 1,
                "y": len(dataset.y) // 2,
                "x": len(dataset.x) // 2,
                "variable": -1,
            }
        )
    )


# In[31]:


def append_to_zarr(dataset: xr.Dataset, zarr_path: Union[str, Path]):
    print("Writing to disk...", zarr_path, flush=True)
    zarr_path = Path(zarr_path)
    if zarr_path.exists():
        to_zarr_kwargs = dict(
            append_dim="init_time",
        )
    else:
        to_zarr_kwargs = dict(
            # Need to manually set the time units otherwise xarray defaults to using
            # units of *days* (and hence cannot represent sub-day temporal resolution), which corrupts
            # the `time` values when we appending to Zarr.  See:
            # https://github.com/pydata/xarray/issues/5969 and
            # http://xarray.pydata.org/en/stable/user-guide/io.html#time-units
            encoding={
                "init_time": {"units": "nanoseconds since 1970-01-01"},
                "UKV": {
                    "compressor": numcodecs.Blosc(cname="zstd", clevel=5),
                },
            },
        )

    dataset.to_zarr(zarr_path, **to_zarr_kwargs)
    print("Finished writing to disk!", zarr_path, flush=True)


# In[32]:


def load_grib_files_for_single_nwp_init_time(full_filenames: list[Path]) -> xr.Dataset:
    datasets_for_nwp_init_datetime = []
    for full_filename in full_filenames:
        print("Opening", full_filename)
        try:
            dataset_for_filename = load_grib_file(full_filename)
        except EOFError as e:
            print(e, f"Filesize = {full_filename.stat().st_size:,d} bytes")
            # If any of the files associated with this nwp_init_datetime is broken then
            # skip all, because we don't want incomplete data for an init_datetime.
            datasets_for_nwp_init_datetime = []
            break
        else:
            if dataset_has_variables(dataset_for_filename):
                datasets_for_nwp_init_datetime.append(dataset_for_filename)
    if len(datasets_for_nwp_init_datetime) == 0:
        print("No valid data found for", full_filenames)
        return
    print("Merging...")
    dataset_for_nwp_init_datetime = xr.merge(datasets_for_nwp_init_datetime)
    del datasets_for_nwp_init_datetime
    print("Reshaping...")
    dataset_for_nwp_init_datetime = reshape_1d_to_2d(dataset_for_nwp_init_datetime)
    dataset_for_nwp_init_datetime = dataset_for_nwp_init_datetime.expand_dims("time", axis=0)
    dataset_for_nwp_init_datetime = post_process_dataset(dataset_for_nwp_init_datetime)
    return dataset_for_nwp_init_datetime


# In[33]:


def load_grib_files_from_groupby_tuple(groupby_tuple: tuple[pd.Timestamp, pd.Series]) -> xr.Dataset:
    full_grib_filenames = groupby_tuple[1]
    return load_grib_files_for_single_nwp_init_time(full_grib_filenames)


# In[40]:


def get_last_nwp_init_datetime_in_zarr(zarr_path: Path) -> datetime.datetime:
    dataset = xr.open_dataset(zarr_path, engine="zarr", mode="r")
    return dataset.init_time[-1].values


# In[34]:


map_datetime_to_grib_filename = decode_and_group_grib_filenames(filenames)


# In[44]:


# Re-start at end of Zarr.
if DST_ZARR_PATH.exists():
    last_nwp_init_datetime_in_zarr = get_last_nwp_init_datetime_in_zarr(DST_ZARR_PATH)
    print(
        DST_ZARR_PATH,
        "exists.  The last NWP init datetime (UTC) in the Zarr is",
        last_nwp_init_datetime_in_zarr,
    )
    nwp_init_datetimes_utc = map_datetime_to_grib_filename.index
    map_datetime_to_grib_filename = map_datetime_to_grib_filename[
        nwp_init_datetimes_utc > last_nwp_init_datetime_in_zarr
    ]


# In[ ]:


with multiprocessing.Pool(processes=8) as pool:
    result_iterator = pool.imap(
        func=load_grib_files_from_groupby_tuple,
        iterable=map_datetime_to_grib_filename.groupby(level=0),
        chunksize=1,
    )
    print("After pool.imap!", flush=True)
    for dataset in result_iterator:
        print("Got dataset from iterator!", flush=True)
        if dataset is None:
            continue
        append_to_zarr(
            dataset,
            DST_ZARR_PATH,
        )
