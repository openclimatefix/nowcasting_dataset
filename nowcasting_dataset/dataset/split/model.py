""" Model for splitting data """
from typing import List

from pydantic import BaseModel, validator


class TrainValidationTestSpecific(BaseModel):
    """Class on how to specifically split the data into train, validation and test."""

    train: List[str]
    validation: List[str]
    test: List[str]

    @validator("train")
    def train_validation_test(cls, v, values):
        """Make sure there is no overlap for the train data"""
        for vv in ["test", "validation"]:
            if vv in values.keys():
                overlap = [period for period in v if period in values[vv]]
                if len(overlap):
                    raise ValueError(f"There is a period in both validation and {vv} sets")

        return v

    @validator("validation")
    def validation_overlap(cls, v, values):
        """Make sure there is no overlap for the validation data"""
        for vv in ["test", "train"]:
            if vv in values.keys():
                overlap = [period for period in v if period in values[vv]]
                if len(overlap):
                    raise ValueError(f"There is a period in both validation and {vv} sets")

        return v

    @validator("test")
    def test_overlap(cls, v, values):
        """Make sure there is no overlap for the test data"""
        for vv in ["validation", "train"]:
            if vv in values.keys():
                overlap = [period for period in v if period in values[vv]]
                if len(overlap):
                    raise ValueError(f"There is a period in both test and {vv} sets")

        return v


default_train_test_validation_specific = TrainValidationTestSpecific(
    train=["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01"],
    validation=["2020-01-01"],
    test=["2021-01-01"],
)
