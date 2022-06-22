""" Util functions for PV data source"""
from typing import List

from nowcasting_datamodel.models.pv import providers


def encode_label(indexes: List[str], label: str) -> List[str]:
    """
    Encode the label to a list of indexes.

    The new encoding must be integers and unique.
    It would be useful if the indexes can read and deciphered by humans.
    This is done by times the original index by 10
    and adding 1 for passiv or 2 for other lables

    Args:
        indexes: list of indexes
        label: either 'solar_sheffield_passiv' or 'pvoutput.org'

    Returns: list of indexes encoded by label
    """
    assert label in providers
    # this encoding does work if the number of pv providers is more than 10
    assert len(providers) < 10

    label_index = providers.index(label)
    new_index = [str(int(col) * 10 + label_index) for col in indexes]

    return new_index
