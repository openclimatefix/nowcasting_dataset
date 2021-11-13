#!/usr/bin/env python
# coding: utf-8

"""Convert numerical weather predictions from the UK Met Office "UKV" model to Zarr.

This script uses multiple processes to speed up the conversion.

Useful links:

* Met Office's data docs: http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf

Known differences between the old Zarr
(UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr)
and the new Zarr:

* Images in the old zarr were top-to-bottom.  Images in the new Zarr follow the ordering in the
  grib files: bottom-to-top.
* The x and y coordinates are different by 1km each.
* The new Zarr is float16?
* The new Zarr has a few more variables.

How this script works:

1) Find all the .grib filenames specified by `source_path_and_search_pattern`
2) Group the .grib filenames by the NWP initialisation datetime (which is the datetime given
   in the grib filename).  For example, this groups together all the Wholesale1 and
   Wholesale2 files.
3) If the `destination_zarr_path` exists then find the last NWP init time in the Zarr,
   and ignore all grib files with init times before that time.
4) Use multiple worker processes to load the grib file.  Each worker process is given a list of
   grib files associated with a single NWP init datetime.  The worker process loads these grib
   files and does some simple post-processing, and then the worker process appends to the
   destination zarr.

The multi-processing aspect of the code is engineered to satisfy several constraints:
1) The destination Zarr must be appended to in strict order and only one process can append to
   the Zarr at one time.
2) Passing large objects (such as a few hundred megabytes of NWP data) through a 
   multiprocessing.Queue is *really* slow:  It takes about 7 seconds to pass an xr.Dataset
   representing Wholesale1 and Wholesale2 NWP data.  This slowness is what motivates us to
   avoid passing the loaded Dataset between processes.  Instead, we don't pass large objects
   between processes:  A single process loads the grib files, processes, and writes the data to
   Zarr.

The code guarantees that the processes write to disk in order of the NWP init time by using a
"chain of locks", kind of like a linked list. The iterable passed into multiprocessing.Pool.map
is a tuple of the list of grib filenames, a "previous_lock" and a "next_lock".  When a process
is working on task n, it cannot write to the destination Zarr until the "previous_lock" is released.
Then a process finishes writing to disk, it release "next_lock" (which enables the process working
on the next NWP init time to write to the destination Zarr).

"""
import datetime
import logging
import multiprocessing
import re
import time
from pathlib import Path
from typing import Union, Optional

import cfgrib
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import click
import glob


logger = logging.getLogger(__name__)

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@click.command()
@click.option(
    "--source_grib_path_and_search_pattern",
    default=(
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/"
        "UK_Met_Office/UKV/native/*/*/*/*Wholesale[12].grib"),
    type=str,
    help=(
        "The directory and the search pattern for the source grib files."
        "  For example /foo/bar/*/*/*/*Wholesale[12].grib"),
)
@click.option(
    "--destination_zarr_path",
    type=Path,
    help="The output Zarr path to write to.  Will be appended to if already exists.",
)
@click.option(
    "--n_processes",
    default=8,
    type=int,
    help="The number of processes to use for loading grib files in parallel.",
)
@click.option(
    "--debug_level",
    default="DEBUG",
    type=click.Choice(_LOG_LEVELS),
)
@click.option(
    "--log_filename",
    default=None,
    type=Optional[Path],
    help=(
        "If not set then will default to `destination_zarr_path` with the"
        " suffix replaced with '.log'"),
)
@click.option(
    "--n_grib_files_per_nwp_init_time",
    default=2,
    type=int,
    help=(
        "The number of grib files expected per NWP initialisation time.  For example, if the search"
        " pattern includes Wholesale1 and Wholesale2 files, then set"
        " n_grib_files_per_nwp_init_time to 2."),
)
def main(
        source_grib_path_and_search_pattern: str,
        destination_zarr_path: Path,
        n_processes: int,
        log_level: str,
        log_filename: Optional[Path],
        n_grib_files_per_nwp_init_time: int
):
    # Set up logging.
    if log_filename is None:
        log_filename = destination_zarr_path.parent / (destination_zarr_path.stem + ".log")
    configure_logging(log_level=log_level, log_filename=log_filename)
    filter_eccodes_logging()

    # Get all filenames.
    logger.info(f"Getting list of all filenames in {source_grib_path_and_search_pattern}...")
    filenames = list(glob.glob(source_grib_path_and_search_pattern))
    logger.info(f"Got {len(filenames):,d} filenames.")
    if len(filenames) == 0:
        logger.warning(
            "No files found!  Are you sure the source_grib_path_and_search_pattern is correct?")
        return

    # Decode, group, and filter filenames:
    map_datetime_to_grib_filename = decode_and_group_grib_filenames(
        filenames=filenames,
        n_grib_files_per_nwp_init_time=n_grib_files_per_nwp_init_time)
    map_datetime_to_grib_filename = select_grib_filenames_still_to_process(
        map_datetime_to_grib_filename, destination_zarr_path)

    # The main event!
    process_grib_files_in_parallel(
        map_datetime_to_grib_filename=map_datetime_to_grib_filename,
        destination_zarr_path=destination_zarr_path,
        n_processes=n_processes,
    )


def configure_logging(log_level: str, log_filename: str) -> None:
    """Create and configure logger."""
    assert log_level in _LOG_LEVELS
    log_level = getattr(logging, log_level)  # Convert string to int.
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode="a")]

    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def filter_eccodes_logging():
    # Filter the ecCodes log warning
    # "ecCodes provides no latitudes/longitudes for gridType='transverse_mercator'"
    # generated here: https://github.com/ecmwf/cfgrib/blob/master/cfgrib/dataset.py#L402
    class FilterEccodesWarning(logging.Filter):
        def filter(self, record) -> bool:
            """Inspect `record`. Return True to log `record`. Return False to ignore `record`."""
            return not record.getMessage() == (
                "ecCodes provides no latitudes/longitudes for gridType='transverse_mercator'"
            )

    logging.getLogger("cfgrib.dataset").addFilter(FilterEccodesWarning())

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
# * Restart from last `time` in existing Zarr.
# * Convert to script
# * Write description of how this works.
# * Logging (including wrapping the worker thread in a try except block to log exceptions immediately).
# * Tidy up code.
# * Use click to set source and target directories, and pattern for finding source files, and number of expected grib files per NWP init datetime.
#
# Some outstanding questions / Todo items
#
# * Experiment with compression?
# * Separately log "bad files" to a CSV file?
# * Check for NaNs.  cdcb has NaNs.
# * If we get an error like this then remove the idx file and try again:
"""
Reshaping...
Can't read index file '/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/UK_Met_Office/UKV/native/2017/11/25/201711250300_u1096_ng_umqv_Wholesale2.grib.923a8.idx'
Traceback (most recent call last):
  File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/site-packages/cfgrib/messages.py", line 361, in from_indexpath_or_filestream
    self = cls.from_indexpath(indexpath)
  File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/site-packages/cfgrib/messages.py", line 328, in from_indexpath
    index = pickle.load(file)
EOFError: Ran out of input
Can't read index file '/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/UK_Met_Office/UKV/native/2017/11/25/201711250300_u1096_ng_umqv_Wholesale2.grib.923a8.idx'
Traceback (most recent call last):
  File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/site-packages/cfgrib/messages.py", line 361, in from_indexpath_or_filestream
    self = cls.from_indexpath(indexpath)
  File "/home/jack/miniconda3/envs/nowcasting_dataset/lib/python3.9/site-packages/cfgrib/messages.py", line 328, in from_indexpath
    index = pickle.load(file)
EOFError: Ran out of input
"""


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


def grib_filename_to_datetime(full_filename: Path) -> datetime.datetime:
    """Parse the grib filename and return the datetime encoded in the filename.

    Returns a datetime.
      For example, if the filename is '202101010000_u1096_ng_umqv_Wholesale1.grib',
      then the returned datetime will be datetime(year=2021, month=1, day=1, hour=0, minute=0).

    Raises RuntimeError if the filename does not match the expected regex pattern.
    """
    # Get the base_filename, which will be of the form '202101010000_u1096_ng_umqv_Wholesale1.grib'
    base_filename = full_filename.name

    # Use regex to match the year, month, day, hour, minute and wholesale_number.
    # That is, group the filename like this: '(2021)(01)(01)(00)(00)_u1096_ng_umqv_Wholesale(1)'.
    # A quick guide to the relevant regex operators:
    #   ^ matches the beginning of the string.
    #   () defines a group.
    #   (?P<name>...) names the group.  We can access the group with `regex_match.groupdict()[name]`
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
        msg = (
            f"Filename '{full_filename}' does not conform to expected"
            f" regex pattern '{regex_pattern_string}'!"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Convert strings to ints:
    regex_groups = {key: int(value) for key, value in regex_match.groupdict().items()}

    return datetime.datetime(**regex_groups)


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
    def _filter_func(group):
        return group.count() == n_grib_files_per_nwp_init_time
    return map_datetime_to_filename.groupby(level=0).filter(_filter_func)


def select_grib_filenames_still_to_process(
        map_datetime_to_grib_filename: pd.Series, destination_zarr_path: Path) -> pd.Series:
    """Remove grib filenames for NWP init times that already exist in Zarr."""
    if destination_zarr_path.exists():
        last_nwp_init_datetime_in_zarr = get_last_nwp_init_datetime_in_zarr(destination_zarr_path)
        logger.info(
            f"{destination_zarr_path} exists.  The last NWP init datetime (UTC) in the Zarr is"
            f" {last_nwp_init_datetime_in_zarr}"
        )
        nwp_init_datetimes_utc = map_datetime_to_grib_filename.index
        map_datetime_to_grib_filename = map_datetime_to_grib_filename[
            nwp_init_datetimes_utc > last_nwp_init_datetime_in_zarr
        ]
    return map_datetime_to_grib_filename


def load_grib_file(full_grib_filename: Union[Path, str], verbose: bool = False) -> xr.Dataset:
    """Merges and loads all contiguous xr.Datasets from the grib file.

    Removes unnecessary variables.  Picks heightAboveGround=1meter for temperature.

    Returns an xr.Dataset which has been loaded from disk.  Loading from disk at this point
    takes about 2 seconds for a 250 MB grib file, but speeds up reshape_1d_to_2d
    from about 7 seconds to 0.5 seconds :)

    Args:
      full_grib_filename:  The full filename (including the path) of a single grib file.
      verbose:  If True then print out some useful debugging information.
    """
    # The grib files are "heterogeneous", so we use cfgrib.open_datasets
    # to return a list of contiguous xr.Datasets.
    # See https://github.com/ecmwf/cfgrib#automatic-filtering
    logger.debug(f"Opening {full_grib_filename}...")
    datasets_from_grib = cfgrib.open_datasets(full_grib_filename)
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
        # heightAboveGround only has 1 entry ("1" meter above ground) and `heightAboveGround`
        # isn't set as a dimension for `t`.
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
    logger.debug(f"Loading {full_grib_filename}...")
    return merged_ds.load()


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


def dataset_has_variables(dataset: xr.Dataset) -> bool:
    """Return True if `dataset` has at least one variable."""
    return len(dataset.variables) > 0


def post_process_dataset(dataset: xr.Dataset) -> xr.Dataset:
    logger.debug("Post-processing dataset...")
    return (
        dataset.to_array(dim="variable", name="UKV")
        .to_dataset()
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


def append_to_zarr(dataset: xr.Dataset, zarr_path: Union[str, Path]):
    logger.debug(f"Writing to disk {zarr_path}...")
    zarr_path = Path(zarr_path)
    if zarr_path.exists():
        to_zarr_kwargs = dict(
            append_dim="init_time",
        )
    else:
        to_zarr_kwargs = dict(
            # Need to manually set the time units otherwise xarray defaults to using
            # units of *days* (and hence cannot represent sub-day temporal resolution), which
            # corrupts the `time` values when we appending to Zarr.  See:
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
    logger.debug(f"Finished writing to disk! {zarr_path}")


def load_grib_files_for_single_nwp_init_time(
        full_grib_filenames: list[Path], task_number: int) -> Union[xr.Dataset, None]:
    """Returns processed Dataset merging all grib files specified by full_filenames.

    Returns None if any of the grib filenames are invalid.
    """
    assert len(full_grib_filenames) > 0
    datasets_for_nwp_init_datetime = []
    for full_grib_filename in full_grib_filenames:
        logger.debug(f"Task number {task_number} opening {full_grib_filename}")
        try:
            dataset_for_filename = load_grib_file(full_grib_filename)
        except EOFError as e:
            logger.warning(f"{e}. Filesize = {full_grib_filename.stat().st_size:,d} bytes")
            # If any of the files associated with this nwp_init_datetime is broken then
            # skip all, because we don't want incomplete data for an init_datetime.
            return
        else:
            if dataset_has_variables(dataset_for_filename):
                datasets_for_nwp_init_datetime.append(dataset_for_filename)
    logger.debug(f"Task number {task_number} merging datasets...")
    dataset_for_nwp_init_datetime = xr.merge(datasets_for_nwp_init_datetime)
    del datasets_for_nwp_init_datetime
    logger.debug(f"Task number {task_number} reshaping datasets...")
    dataset_for_nwp_init_datetime = reshape_1d_to_2d(dataset_for_nwp_init_datetime)
    dataset_for_nwp_init_datetime = dataset_for_nwp_init_datetime.expand_dims("time", axis=0)
    dataset_for_nwp_init_datetime = post_process_dataset(dataset_for_nwp_init_datetime)
    return dataset_for_nwp_init_datetime


def get_last_nwp_init_datetime_in_zarr(zarr_path: Path) -> datetime.datetime:
    dataset = xr.open_dataset(zarr_path, engine="zarr", mode="r")
    return dataset.init_time[-1].values


def load_grib_files_and_save_zarr_with_lock(task: dict[str, object]) -> None:
    full_filenames = task['row']
    previous_lock = task['previous_lock']
    next_lock = task['next_lock']
    task_number = task['task_number']
    destination_zarr_path = task['destination_zarr_path']
    start_time = task['start_time']

    TIMEOUT_SECONDS = 120
    dataset = load_grib_files_for_single_nwp_init_time(full_filenames, task_number=task_number)
    if dataset is not None:
        # Block if reader processes are getting ahead of the Zarr writing process.
        logger.debug(f"Before previous_lock.acquire() for task number {task_number}")
        previous_lock.acquire(blocking=True, timeout=TIMEOUT_SECONDS)
        logger.debug(f"After previous_lock.acquire() for task number {task_number}")
        append_to_zarr(dataset, destination_zarr_path)
    else:
        logger.warning(
            f"Dataset is None!  For task number {task_number}, grib filenames = {full_filenames}")

    # Calculate timings.
    time_taken = pd.Timestamp.now() - start_time
    seconds_per_task = (time_taken / (task_number + 1)).total_seconds()
    logger.debug(
        f"{task_number + 1:,d} tasks completed in {time_taken}"
        f". That's {seconds_per_task:,.1f} seconds  per task.")
    next_lock.release()


def load_grib_files_and_save_zarr_with_lock_wrapper(task: dict[str, object]) -> None:
    """Simple wrapper around load_grib_files_and_save_zarr_with_lock to catch exceptions."""
    try:
        task_number = task["task_number"]
        full_filenames = task["row"]
        load_grib_files_and_save_zarr_with_lock(task)
    except Exception:
        logger.exception(
            f"Exception raised when processing task number {task_number},"
            f" loading grib filenames {full_filenames}")
        raise


def process_grib_files_in_parallel(
        map_datetime_to_grib_filename: pd.Series,
        destination_zarr_path: Path,
        n_processes: int,
) -> None:
    # To pass the shared Lock into the worker processes, we must use a Manager():
    multiprocessing_manager = multiprocessing.Manager()

    # Make note of when this script started.  This is used to compute how many
    # tasks the script completes in a given time.
    start_time = pd.Timestamp.now()

    # Create a list of `tasks` which include the grib filenames, the prev_lock & next_lock:
    tasks: list[dict[str, object]] = []
    previous_lock = multiprocessing_manager.Lock()  # Lock starts in a "released" state.
    for i, (_, row) in enumerate(map_datetime_to_grib_filename.groupby(level=0)):
        next_lock = multiprocessing_manager.Lock()
        next_lock.acquire()
        tasks.append(
            dict(
                row=row,
                previous_lock=previous_lock,
                next_lock=next_lock,
                task_number=i,
                destination_zarr_path=destination_zarr_path,
                start_time=start_time,
            )
        )
        previous_lock = next_lock

    # Run the processes!
    with multiprocessing.Pool(processes=n_processes) as pool:
        result_iterator = pool.map(
            func=load_grib_files_and_save_zarr_with_lock_wrapper, iterable=tasks)

    # Loop through the results to trigger any exceptions:
    try:
        for iterator in result_iterator:
            pass
    except Exception:
        logger.exception()
        raise

    logger.info("Done!")


if __name__ == "__main__":
    main()
