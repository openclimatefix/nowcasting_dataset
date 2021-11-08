""" utils functions """
import logging
import os
import re
import tempfile
from functools import wraps

import fsspec.asyn
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset
import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset.config import load, model
from nowcasting_dataset.consts import Array

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


# TODO: Issue #170. Is this this function still used?
def is_unique(a: Array) -> bool:
    """Check array has unique values"""
    # TODO: Can probably replace with pd.Index.is_unique()
    return len(a) == len(np.unique(a))


# TODO: Issue #170. Is this this function still used?
def scale_to_0_to_1(a: Array) -> Array:
    """Scale to the range [0, 1]."""
    a = a - a.min()
    a = a / a.max()
    np.testing.assert_almost_equal(np.nanmin(a), 0.0)
    np.testing.assert_almost_equal(np.nanmax(a), 1.0)
    return a


def get_netcdf_filename(batch_idx: int) -> str:
    """Generate full filename, excluding path."""
    assert 0 <= batch_idx < 1e6
    return f"{batch_idx:06d}.nc"


# TODO: Issue #170. Is this this function still used?
def to_numpy(value):
    """Change generic data to numpy"""
    if isinstance(value, xr.DataArray):
        # TODO: Use to_numpy() or as_numpy(), introduced in xarray v0.19?
        value = value.data

    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.values
    elif isinstance(value, pd.DatetimeIndex):
        value = value.values.astype("datetime64[s]").astype(np.int32)
    elif isinstance(value, pd.Timestamp):
        value = np.int32(value.timestamp())
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.datetime64):
        value = value.astype("datetime64[s]").astype(np.int32)

    return value


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


def get_config_with_test_paths(config_filename: str) -> model.Configuration:
    """Sets the base paths to point to the testing data in this repository."""
    local_path = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "../")

    # load configuration, this can be changed to a different filename as needed
    filename = os.path.join(local_path, "tests", "config", config_filename)
    config = load.load_yaml_configuration(filename)
    config.set_base_path(local_path)
    return config


def arg_logger(func):
    """A function decorator to log all the args and kwargs passed into a function."""
    # Adapted from https://stackoverflow.com/a/23983263/732596
    @wraps(func)
    def inner_func(*args, **kwargs):
        logger.debug(
            f"Arguments passed into function `{func.__name__}`:" f" args={args}; kwargs={kwargs}"
        )
        return func(*args, **kwargs)

    return inner_func
