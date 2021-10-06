This folder contains the code for the different data sources.

# Data Sources
- datetime: datetime information like 'hour_of_day'
- general: metadata for the batch like t0_dt
- gsp: Grid Supply Point data from Sheffield Solar
- nwp: Numerical Weather predictions from ...
- pv: PV output data from pvoutput.org
- satellite: satellite data from ...
- sun: Sun position data
- topographic: Topographic data i.e the elevation of the land.

# data_source.py

General class used for making a data source. It has the following functions
- get_batch: gets batch data
- datetime_index: gets the datatimes of the source
- get_example: gets one data, need many fo these for a batch
- get_locations_for_batch: gets the x,y locations of the datasource, that will be used to make batches.


# datasource_output.py

General pydantic model of output of the data source. Contains the following methods
- to_numpy: changes all data points to numpy objects
- split: converts a batch to a list of items
- join: joins list of items to one
- to_xr_dataset: changes data items to xarrays and returns a dataset
- from_xr_dataset: loads from a xr dataset
- select_time_period: subselect data, depending on a time period

# General Data Source folder

Roughly each of the data source folders follows this pattern
- A class which is how to load the data source, how to select for batches e.t.c. This is built from 'data_source.py',
- A class which contains the output model of the data source. This is the information used in the batches.
This is built from 'datasource_output.py',
