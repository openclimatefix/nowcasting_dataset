from nowcasting_dataset import utils
import pandas as pd


def test_is_monotically_increasing():
    assert utils.is_monotonically_increasing([1, 2, 3, 4])
    assert not utils.is_monotonically_increasing([1, 2, 3, 3])
    assert not utils.is_monotonically_increasing([1, 2, 3, 0])

    index = pd.date_range('2010-01-01', freq='H', periods=4)
    assert utils.is_monotonically_increasing(index)
    assert not utils.is_monotonically_increasing(index[::-1])
