"""
Test for datasource output validations
"""
import numpy as np
import pytest

from nowcasting_dataset.data_sources.fake.batch import gsp_fake
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP


def test_datasource_output_validation(configuration):  # noqa: D103
    configuration.process.batch_size = 2
    configuration.input_data.gsp.history_minutes = 60
    configuration.input_data.gsp.forecast_minutes = 60
    configuration.input_data.gsp.n_gsp_per_example = 6

    gsp = gsp_fake(configuration)

    GSP.model_validation(gsp)

    # nan error
    gsp.power_mw[0, 0] = np.nan
    with pytest.raises(Exception):
        GSP.model_validation(gsp)

    # inf error
    gsp.power_mw[0, 0] = np.inf
    with pytest.raises(Exception):
        GSP.model_validation(gsp)

    # fill nan
    gsp.power_mw[0, 0] = np.nan
    gsp.__setitem__(
        "power_mw", gsp.check_nan_and_fill_warning(data=gsp.power_mw, variable_name="power_mw")
    )
    assert gsp.power_mw[0, 0, 0].values == 0

    # greater than
    gsp.power_mw[0, 0] = 0
    with pytest.raises(Exception):
        gsp.check_dataset_greater_than_or_equal_to(
            data=gsp.power_mw, min_value=1, variable_name="power_mw"
        )

    # less than
    gsp.power_mw[0, 0] = 0
    with pytest.raises(Exception):
        gsp.check_dataset_less_than_or_equal_to(
            data=gsp.power_mw, max_value=-1, variable_name="power_mw"
        )

    # equal to than
    gsp.power_mw[0, 0] = 0
    with pytest.raises(Exception):
        gsp.check_dataset_less_than_or_equal_to(
            data=gsp.power_mw, value=0, variable_name="power_mw"
        )

    # dims
    with pytest.raises(Exception):
        gsp.check_data_var_dim(data=gsp.power_mw, expected_dims="fake_dim")
