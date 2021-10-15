""" utils functions """
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Optional

import fsspec.asyn
import gcsfs
import numpy as np
import pandas as pd
import torch
import xarray as xr

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


def is_monotonically_increasing(a: Array) -> bool:
    """ Check the array is monotonically increasing """
    # TODO: Can probably replace with pd.Index.is_monotonic_increasing()
    assert a is not None
    assert len(a) > 0
    if isinstance(a, pd.DatetimeIndex):
        a = a.view(int)
    a = np.asarray(a)
    return np.all(np.diff(a) > 0)


def is_unique(a: Array) -> bool:
    """ Check array has unique values """
    # TODO: Can probably replace with pd.Index.is_unique()
    return len(a) == len(np.unique(a))


def scale_to_0_to_1(a: Array) -> Array:
    """Scale to the range [0, 1]."""
    a = a - a.min()
    a = a / a.max()
    np.testing.assert_almost_equal(np.nanmin(a), 0.0)
    np.testing.assert_almost_equal(np.nanmax(a), 1.0)
    return a


def sin_and_cos(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every column in df, creates cols for sin and cos of that col.

    Args:
      df: Input DataFrame.  The values must be in the range [0, 1].

    Raises:
      ValueError if any value in df is not within the range [0, 1].

    Returns:
      A new DataFrame, with twice the number of columns as the input df.
      For each col in df, the output DataFrame will have a <col name>_sin
      and a <col_name>_cos.
    """
    columns = []
    for col_name in df.columns:
        columns.append(f"{col_name}_sin")
        columns.append(f"{col_name}_cos")
    output_df = pd.DataFrame(index=df.index, columns=columns, dtype=np.float32)
    for col_name in df.columns:
        series = df[col_name]
        if series.min() < 0.0 or series.max() > 1.0:
            raise ValueError(
                f"{col_name} has values outside the range [0, 1]!"
                f" min={series.min()}; max={series.max()}"
            )
        radians = series * 2 * np.pi
        output_df[f"{col_name}_sin"] = np.sin(radians)
        output_df[f"{col_name}_cos"] = np.cos(radians)
    return output_df


def get_netcdf_filename(batch_idx: int, add_hash: bool = False) -> Path:
    """Generate full filename, excluding path.

    Filename includes the first 6 digits of the MD5 hash of the filename,
    as recommended by Google Cloud in order to distribute data across
    multiple back-end servers.

    Add option to turn on and off hashing

    """
    filename = f"{batch_idx}.nc"
    # Remove 'hash' at the moment. In the future could has the configuration file, and use this to make sure we are
    # saving and loading the same thing
    if add_hash:
        hash_of_filename = hashlib.md5(filename.encode()).hexdigest()
        filename = f"{hash_of_filename[0:6]}_{filename}"

    return filename


def to_numpy(value):
    """ Change generic data to numpy"""
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
    elif isinstance(value, torch.Tensor):
        value = value.numpy()

    return value


def coord_to_range(
    da: xr.DataArray, dim: str, prefix: Optional[str], dtype=np.int32
) -> xr.DataArray:
    """
    TODO

    TODO: Actually, I think this is over-complicated?  I think we can
    just strip off the 'coord' from the dimension.

    """
    coord = da[dim]
    da[dim] = np.arange(len(coord), dtype=dtype)
    if prefix is not None:
        da[f"{prefix}_{dim}_coords"] = xr.DataArray(coord, coords=[da[dim]], dims=[dim])
    return da


class OpenData:
    """ General method to open a file, but if from GCS, the file is downloaded to a temp file first """

    def __init__(self, file_name):
        """ Check file is there, and create temporary file """
        self.file_name = file_name

        filesystem = fsspec.open(file_name).fs
        if not filesystem.exists(file_name):
            raise RuntimeError(f"{file_name} does not exist!")

        self.temp_file = tempfile.NamedTemporaryFile()

    def __enter__(self):
        """Return filename

        1. if from gcs, download the file to temporary file, and return the temporary file name
        2. if local, return local file name
        """
        fs = fsspec.open(self.file_name).fs
        if type(fs) == gcsfs.GCSFileSystem:
            fs.get_file(self.file_name, self.temp_file.name)
            filename = self.temp_file.name
        else:
            filename = self.file_name

        return filename

    def __exit__(self, type, value, traceback):
        """ Close temporary file """
        self.temp_file.close()
