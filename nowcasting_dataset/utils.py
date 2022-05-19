""" utils functions """
import logging
import re
import tempfile
import threading
from concurrent import futures
from functools import wraps

import fsspec.asyn
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset.consts import LOG_LEVELS, Array

logger = logging.getLogger(__name__)


def set_fsspec_for_multiprocess() -> None:
    """
    Clear reference to the loop and thread.

    This is a nasty hack that was suggested but NOT recommended by the lead fsspec developer!

    This appears necessary otherwise gcsfs hangs when used after forking multiple worker processes.
    Only required for fsspec >= 0.9.0

    See:
    - https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    - https://github.com/fsspec/filesystem_spec/pull/963#issuecomment-1131709948

    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None


# TODO: Issue #170. Is this this function still used?
def is_monotonically_increasing(a: Array) -> bool:
    """Check the array is monotonically increasing"""
    # TODO: Can probably replace with pd.Index.is_monotonic_increasing()
    assert a is not None
    assert len(a) > 0
    if isinstance(a, pd.DatetimeIndex):
        a = a.view(int)
    a = np.asarray(a)
    return np.all(np.diff(a) > 0)


def get_netcdf_filename(batch_idx: int) -> str:
    """Generate full filename, excluding path."""
    assert 0 <= batch_idx < 1e6
    return f"{batch_idx:06d}.nc"


class OpenData:
    """Open a file, but if from GCS, the file is downloaded to a temp file first."""

    def __init__(self, file_name):
        """Check file is there, and create temporary file"""
        self.file_name = file_name

        filesystem = nd_fs_utils.get_filesystem(file_name)
        if not filesystem.exists(file_name):
            raise RuntimeError(f"{file_name} does not exist!")

        self.temp_file = tempfile.NamedTemporaryFile()

    def __enter__(self):
        """Return filename

        1. if from gcs, download the file to temporary file, and return the temporary file name
        2. if local, return local file name
        """
        fs = nd_fs_utils.get_filesystem(self.file_name)
        if type(fs) == gcsfs.GCSFileSystem:
            fs.get_file(self.file_name, self.temp_file.name)
            filename = self.temp_file.name
        else:
            filename = self.file_name

        return filename

    def __exit__(self, type, value, traceback):
        """Close temporary file"""
        self.temp_file.close()


def remove_regex_pattern_from_keys(d: dict, pattern_to_remove: str, **regex_compile_kwargs) -> dict:
    """Remove `pattern_to_remove` from all keys in `d`.

    Return a new dict with the same values as `d`, but where the key names
    have had `pattern_to_remove` removed.
    """
    new_dict = {}
    regex = re.compile(pattern_to_remove, **regex_compile_kwargs)
    for old_key, value in d.items():
        new_key = regex.sub(string=old_key, repl="")
        new_dict[new_key] = value
    return new_dict


def configure_logger(log_level: str, logger_name: str, handlers=list[logging.Handler]) -> None:
    """Configure logger.

    Args:
      log_level: String representing logging level, e.g. 'DEBUG'.
      logger_name: String.
      handlers: A list of logging.Handler objects.
    """
    assert log_level in LOG_LEVELS
    log_level = getattr(logging, log_level)  # Convert string to int.

    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(module)s#L%(lineno)d:PID=%(process)d:%(message)s"
    )

    local_logger = logging.getLogger(logger_name)
    local_logger.setLevel(log_level)

    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        local_logger.addHandler(handler)


def get_start_and_end_example_index(batch_idx: int, batch_size: int) -> tuple[int, int]:
    """
    Get the start and end example index

    Args:
        batch_idx: the batch number
        batch_size: the size of the batches

    Returns: start and end example index

    """
    start_example_idx = batch_idx * batch_size
    end_example_idx = (batch_idx + 1) * batch_size

    return start_example_idx, end_example_idx


class DummyExecutor(futures.Executor):
    """Drop-in replacement for ThreadPoolExecutor or ProcessPoolExecutor

    This is currently not used in any code, but very useful when debugging.

    Adapted from https://stackoverflow.com/a/10436851/732596
    """

    def __init__(self, *args, **kwargs):
        """Initialise DummyExecutor."""
        self._shutdown = False
        self._shutdownLock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        """Submit task to DummyExecutor."""
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = futures.Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        """Shutdown dummy executor."""
        with self._shutdownLock:
            self._shutdown = True


def arg_logger(func):
    """A function decorator to log all the args and kwargs passed into a function."""
    # Adapted from https://stackoverflow.com/a/23983263/732596
    @wraps(func)
    def inner_func(*args, **kwargs):
        logger.debug(f"Arguments passed into function `{func.__name__}`: {args=}; {kwargs=}")
        return func(*args, **kwargs)

    return inner_func


def exception_logger(func):
    """A function decorator to log exceptions thrown by the inner function."""
    # Adapted from
    # www.blog.pythonlibrary.org/2016/06/09/python-how-to-create-an-exception-logging-decorator
    @wraps(func)
    def inner_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:  # noqa: E722
            logger.exception(
                f"EXCEPTION when calling `{func.__name__}`!"
                f" Arguments passed into function: {args=}; {kwargs=}"
            )
            raise

    return inner_func


def drop_duplicate_times(data_array: xr.DataArray, class_name: str, time_dim: str) -> xr.DataArray:
    """
    Drop duplicate times in data array

    Args:
        data_array: main data
        class_name: the data source name
        time_dim: the time dimension we want to look at

    Returns: data array with no duplicated times

    """
    # If there are any duplicated init_times then drop the duplicated init_times:
    time = pd.DatetimeIndex(data_array[time_dim])
    if not time.is_unique:
        n_duplicates = time.duplicated().sum()
        logger.warning(f"{class_name} Zarr has {n_duplicates:,d} duplicated init_times.  Fixing...")
        data_array = data_array.drop_duplicates(dim=time_dim)

    return data_array


def drop_non_monotonic_increasing(
    data_array: xr.DataArray, class_name: str, time_dim: str
) -> xr.DataArray:
    """
    Drop non monotonically increasing time steps

    Args:
        data_array: main data
        class_name: the data source name
        time_dim: the name of the time dimension we want to check

    Returns: data array with monotonically increase time

    """
    # If any init_times are not monotonic_increasing then drop the out-of-order init_times:
    time = pd.DatetimeIndex(data_array[time_dim])
    if not time.is_monotonic_increasing:
        total_n_out_of_order_times = 0
        logger.warning(f"{class_name} Zarr {time_dim} is not monotonic_increasing.  Fixing...")
        while not time.is_monotonic_increasing:
            # get the first set of out of order time value
            diff = np.diff(time.view(int))
            out_of_order = np.where(diff < 0)[0]
            out_of_order = time[out_of_order]

            # remove value
            data_array = data_array.drop_sel(**{time_dim: out_of_order})

            # get time vector for next while loop
            time = pd.DatetimeIndex(data_array[time_dim])

            # save how many have been removed, just for logging
            total_n_out_of_order_times += len(out_of_order)

        logger.info(f"Fixed {total_n_out_of_order_times:,d} out of order {time_dim}.")

    return data_array


def is_sorted(array: np.ndarray) -> bool:
    """Return True if array is sorted in ascending order."""
    # Adapted from https://stackoverflow.com/a/47004507/732596
    if len(array) == 0:
        return False
    return np.all(array[:-1] <= array[1:])
