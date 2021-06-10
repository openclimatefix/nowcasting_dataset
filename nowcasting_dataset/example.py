from typing import TypedDict, Union
import datetime
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
    # Shape: batch_size, seq_length, width, height
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
    pv_location_x: float
    pv_location_y: float

    # Datetimes
    # At 5-minute timings like 00, 05, 10, ...;
    # *not* the 04, 09, ... sequence of the satellite imagery.
    # Datetimes become Unix epochs (UTC) just before being passed into the ML
    # model.
    start_datetime: Union[datetime.datetime, int]
    end_datetime: Union[datetime.datetime, int]
    t0_datetime: Union[datetime.datetime, int]  #: t0 is 'now', the most recent observation.
