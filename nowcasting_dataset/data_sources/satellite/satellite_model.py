""" Model for output of satellite data """
from __future__ import annotations

import logging

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class Satellite(DataSourceOutput):
    """ Class to store satellite data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non negative """
        v.check_nan_and_inf(data=v.data)
        # put this validation back in when issue is done
        v.check_dataset_not_equal(data=v.data, value=-1, raise_error=False)
        return v
