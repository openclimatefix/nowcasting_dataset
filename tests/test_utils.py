import numpy as np
import pandas as pd
import pytest

from nowcasting_dataset import utils


def test_is_monotically_increasing():
    assert utils.is_monotonically_increasing([1, 2, 3, 4])
    assert not utils.is_monotonically_increasing([1, 2, 3, 3])
    assert not utils.is_monotonically_increasing([1, 2, 3, 0])

    index = pd.date_range("2010-01-01", freq="H", periods=4)
    assert utils.is_monotonically_increasing(index)
    assert not utils.is_monotonically_increasing(index[::-1])


def test_sin_and_cos():
    df = pd.DataFrame({"a": range(30), "b": np.arange(30) - 30})
    with pytest.raises(ValueError) as _:
        utils.sin_and_cos(pd.DataFrame({"a": [-1, 0, 1]}))
    with pytest.raises(ValueError) as _:
        utils.sin_and_cos(pd.DataFrame({"a": [0, 1, 2]}))
    with pytest.raises(ValueError) as _:
        utils.sin_and_cos(df)

    rescaled = utils.scale_to_0_to_1(df)
    sin_and_cos = utils.sin_and_cos(rescaled)
    np.testing.assert_array_equal(sin_and_cos.columns, ["a_sin", "a_cos", "b_sin", "b_cos"])


def test_get_netcdf_filename():
    assert utils.get_netcdf_filename(10) == "10.nc"
    assert utils.get_netcdf_filename(10, add_hash=True) == "77eb6f_10.nc"
