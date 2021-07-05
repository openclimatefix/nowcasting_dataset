import numpy as np
import pandas as pd
from nowcasting_dataset.consts import Array
import fsspec.asyn


def set_fsspec_for_multiprocess() -> None:
    """Clear reference to the loop and thread.  This is necessary otherwise
    gcsfs hangs in the ML training loop.  Only required for fsspec >= 0.9.0
    See https://github.com/dask/gcsfs/issues/379#issuecomment-839929801
    TODO: Try deleting this two lines to make sure this is still relevant."""
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None


def is_monotonically_increasing(a: Array) -> bool:
    assert a is not None
    assert len(a) > 0
    if isinstance(a, pd.DatetimeIndex):
        a = a.astype(int)
    a = np.asarray(a)
    return np.all(np.diff(a) > 0)


def is_unique(a: Array) -> bool:
    return len(a) == len(np.unique(a))


def scale_to_0_to_1(a: Array) -> Array:
    """Scale to the range [0, 1]."""
    a = a - a.min()
    a = a / a.max()
    np.testing.assert_almost_equal(a.min().min(), 0.0)
    np.testing.assert_almost_equal(a.max().max(), 1.0)
    return a
