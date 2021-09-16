from typing import Union, List

import pandas as pd
from pydantic import BaseModel, validator


class TrainValidationTestYear(BaseModel):
    train: List[int]
    validation: List[int]
    test: List[int]

    @validator("train")
    def train_validation_test(cls, v, values):
        for vv in ["test", "validation"]:
            if vv in values.keys():
                overlap = [year for year in v if year in values[vv]]
                if len(overlap):
                    raise ValueError(f"There is a year in both validation and {vv} sets")

        return v

    @validator("validation")
    def validation_overlap(cls, v, values):
        for vv in ["test", "train"]:
            if vv in values.keys():
                overlap = [year for year in v if year in values[vv]]
                if len(overlap):
                    raise ValueError(f"There is a year in both validation and {vv} sets")

        return v

    @validator("test")
    def test_overlap(cls, v, values):
        for vv in ["validation", "train"]:
            if vv in values.keys():
                overlap = [year for year in v if year in values[vv]]
                if len(overlap):
                    raise ValueError(f"There is a year in both validation and {vv} sets")

        return v


default_train_test_validation_year = TrainValidationTestYear(
    **{
        "train": [2015, 2016, 2017, 2018, 2019],
        "validation": [2020],
        "test": [2021],
    }
)


def split_year(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    train_test_validation_year: TrainValidationTestYear = default_train_test_validation_year,
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the data by the year.

    The train_test_validation_split shows if each datetime should train, validation or test.
    So if 2015 is in 'train' then all the datetimes from 2015 will be entered into the train dataset

    Args:
        datetimes: list of datetimes
        train_test_validation_year: dictionary of 'train', 'validation' and 'test'

    Returns: train, validation and test datetimes

    """

    # select the datetimes
    train = [
        datetime for datetime in datetimes if datetime.year in train_test_validation_year.train
    ]
    validation = [
        datetime for datetime in datetimes if datetime.year in train_test_validation_year.validation
    ]
    test = [datetime for datetime in datetimes if datetime.year in train_test_validation_year.test]

    return train, validation, test
