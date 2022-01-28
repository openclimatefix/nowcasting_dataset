# noqa: D100
import pandas as pd

from nowcasting_dataset import utils


def test_is_monotically_increasing():  # noqa: D103
    assert utils.is_monotonically_increasing([1, 2, 3, 4])
    assert not utils.is_monotonically_increasing([1, 2, 3, 3])
    assert not utils.is_monotonically_increasing([1, 2, 3, 0])

    index = pd.date_range("2010-01-01", freq="H", periods=4)
    assert utils.is_monotonically_increasing(index)
    assert not utils.is_monotonically_increasing(index[::-1])


def test_drop_monotonically_increasing_nothing():
    """Test to check nothing is dropped when times are correctly in order"""
    times = ["2022-01-01", "2022-01-02", "2022-01-03"]
    data_xr = pd.Series(index=pd.DatetimeIndex(times), data=[1, 2, 3]).to_xarray()

    t = utils.drop_non_monotonic_increasing(
        data_array=data_xr, class_name="test_class", time_dim="index"
    )
    assert (t == data_xr).all()


def test_drop_monotonically_increasing():
    """Test to correct value is droped when times are not in order"""
    times = ["2022-01-02", "2022-01-01", "2022-01-03"]
    data_xr = pd.Series(index=pd.DatetimeIndex(times), data=[1, 2, 3]).to_xarray()

    t = utils.drop_non_monotonic_increasing(
        data_array=data_xr, class_name="test_class", time_dim="index"
    )
    assert (t.index == pd.DatetimeIndex(["2022-01-01", "2022-01-03"])).all()


def test_drop_duplicate_nothing():
    """Test to check nothing is dropped when times are not duplicated"""
    times = ["2022-01-01", "2022-01-02", "2022-01-03"]
    data_xr = pd.Series(index=pd.DatetimeIndex(times), data=[1, 2, 3]).to_xarray()

    t = utils.drop_duplicate_times(data_array=data_xr, class_name="test_class", time_dim="index")
    assert (t == data_xr).all()


def test_drop_duplicate():
    """Test to correct value is droped when times are duplicated"""
    times = ["2022-01-01", "2022-01-01", "2022-01-03"]
    data_xr = pd.Series(index=pd.DatetimeIndex(times), data=[1, 2, 3]).to_xarray()

    t = utils.drop_duplicate_times(data_array=data_xr, class_name="test_class", time_dim="index")
    assert (t.index == pd.DatetimeIndex(["2022-01-01", "2022-01-03"])).all()


def test_get_netcdf_filename():  # noqa: D103
    assert utils.get_netcdf_filename(10) == "000010.nc"


def test_remove_regex_pattern_from_keys():  # noqa: D103
    d = {
        "satellite_zarr_path": "/a/b/c/foo.zarr",
        "bar": "baz",
        "satellite_channels": ["HRV"],
        "n_satellite_per_batch": 4,
    }
    correct = {
        "zarr_path": "/a/b/c/foo.zarr",
        "bar": "baz",
        "channels": ["HRV"],
        "n_satellite_per_batch": 4,
    }
    new_dict = utils.remove_regex_pattern_from_keys(d, pattern_to_remove=r"^satellite_")
    assert new_dict == correct
