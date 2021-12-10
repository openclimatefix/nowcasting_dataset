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

import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset.consts import LOG_LEVELS, Array

logger = logging.getLogger(__name__)


def set_fsspec_for_multiprocess() -> None:
    """
    Clear reference to the loop and thread.

    This is necessary otherwise
    gcsfs hangs in the ML training loop.  Only required for fsspec >= 0.9.0
    See https://github.com/dask/gcsfs/issues/379#issuecomment-839929801
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


def get_start_and_end_example_index(batch_idx: int, batch_size: int) -> (int, int):
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
