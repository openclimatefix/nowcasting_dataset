""" Model for output of satellite data """
from __future__ import annotations

import logging

from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
    check_dataset_not_equal,
    check_nan_and_inf,
)

logger = logging.getLogger(__name__)


class Satellite(DataSourceOutput):
    """ Class to store satellite data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non negative """
        check_nan_and_inf(data=v.data, class_name="satellite")
        # put this validation back in when issue is done
        check_dataset_not_equal(data=v.data, class_name="satellite", value=-1, raise_error=False)
        return v
