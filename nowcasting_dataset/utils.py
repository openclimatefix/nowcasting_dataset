import numpy as np
import pandas as pd
from nowcasting_dataset.consts import Array
import fsspec.asyn
from pathlib import Path
import hashlib


def set_fsspec_for_multiprocess() -> None:
    """Clear reference to the loop and thread.  This is necessary otherwise
    gcsfs hangs in the ML training loop.  Only required for fsspec >= 0.9.0
    See https://github.com/dask/gcsfs/issues/379#issuecomment-839929801
    TODO: Try deleting this two lines to make sure this is still relevant."""
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None


def is_monotonically_increasing(a: Array) -> bool:
    # TODO: Can probably replace with pd.Index.is_monotonic_increasing()
    assert a is not None
    assert len(a) > 0
    if isinstance(a, pd.DatetimeIndex):
        a = a.astype(int)
    a = np.asarray(a)
    return np.all(np.diff(a) > 0)


def is_unique(a: Array) -> bool:
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
    """For every column in df, creates cols for sin and cos of that col.

    Args:
      df: Input DataFrame.  The values must be in the range [0, 1].

    Raises:
      ValueError if any value in df is not within the range [0, 1].

    Returns:
      A new DataFrame, with twice the number of columns as the input df.
      For each col in df, the output DataFrame will have a <col name>_sin
      and a <col_name>_cos."""
    columns = []
    for col_name in df.columns:
        columns.append(f'{col_name}_sin')
        columns.append(f'{col_name}_cos')
    output_df = pd.DataFrame(index=df.index, columns=columns, dtype=np.float32)
    for col_name in df.columns:
        series = df[col_name]
        if series.min() < 0.0 or series.max() > 1.0:
            raise ValueError(
                f'{col_name} has values outside the range [0, 1]!'
                f' min={series.min()}; max={series.max()}')
        radians = series * 2 * np.pi
        output_df[f'{col_name}_sin'] = np.sin(radians)
        output_df[f'{col_name}_cos'] = np.cos(radians)
    return output_df


def get_netcdf_filename(batch_idx: int) -> Path:
    """Generate full filename, excluding path.

    Filename includes the first 6 digits of the MD5 hash of the filename,
    as recommended by Google Cloud in order to distribute data across
    multiple back-end servers.
    """
    filename = f'{batch_idx}.nc'
    hash_of_filename = hashlib.md5(filename.encode()).hexdigest()
    return f'{hash_of_filename[:6]}_{filename}'


def pad_nans(array, pad_width) -> np.ndarray:
    array = array.astype(np.float32)
    return np.pad(array, pad_width, constant_values=np.NaN)
