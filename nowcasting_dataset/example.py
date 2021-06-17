from typing import TypedDict, Union
import pandas as pd
from nowcasting_dataset.consts import Array


class Example(TypedDict):
    """Simple class for structuring data for each ML example.

    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of the fields
         in each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple,
    which would provide runtime checks, but the deal-breaker with Tuples is
    that they're immutable so we cannot change the values in the transforms.
    """
    # IMAGES
    # Shape: batch_size, seq_length, width, height, channel
    sat_data: Array

    # PV yield time series
    pv_yield: Array

    # Numerical weather predictions (NWPs)
    nwp: Array  #: NWP covering the entire geographical extent.
    nwp_above_pv: Array  #: The NWP at a single point nearest to the PV system.
    #: TODO: Document shape.

    # METADATA
    pv_system_id: int
    pv_system_row_number: int  #: Guaranteed to be in the range [0, len(pv_metadata)]
    x_meters_center: float  #: OSGB coordinates for center of the image.
    y_meters_center: float

    # Datetimes (abbreviated to "dt")
    # The date range of the example is [start_date, end_dt].
    # At 5-minutes past the hour {0, 5, ..., 55}
    # *not* the {4, 9, ..., 59} timings of the satellite imagery.
    # Datetimes become Unix epochs (UTC) represented as int64 just before being
    # passed into the ML model.
    start_dt: Union[pd.Timestamp, int]
    end_dt: Union[pd.Timestamp, int]

    #: t0_dt is 'now', the most recent observation.
    # For an example timeseries with only 1 timestep of history,
    # t0_dt will equal start_dt
    t0_dt: Union[pd.Timestamp, int]


def to_numpy(example: Example) -> Example:
    XARRAY_ITEMS = ('sat_data', 'nwp', 'nwp_above_pv')
    for key in XARRAY_ITEMS:
        if key in example:
            example[key] = example[key].data

    DATETIME_ITEMS = ('start_dt', 'end_dt', 't0_dt')
    for key in DATETIME_ITEMS:
        if key in example:
            example[key] = int(example[key].timestamp())

    return example
