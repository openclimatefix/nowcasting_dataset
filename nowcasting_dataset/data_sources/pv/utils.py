""" Util functions for PV data source"""
from typing import List

from nowcasting_dataset.consts import PV_PROVIDERS


def encode_label(indexes: List[str], label: str) -> List[str]:
    """
    Encode the label to a list of indexes.

    The new encoding must be integers and unique.
    It would be useful if the indexes can read and deciphered by humans.
    This is done by times the original index by 10
    and adding 1 for passive or 2 for other lables

    Args:
        indexes: list of indexes
        label: either 'passiv' or 'pvoutput'

    Returns: list of indexes encoded by label
    """
    assert label in PV_PROVIDERS
    # this encoding does work if the number of pv providers is more than 10
    assert len(PV_PROVIDERS) < 10

    label_index = PV_PROVIDERS.index(label)
    new_index = [str(int(col) * 10 + label_index) for col in indexes]

    return new_index
