This folder contains the code for the different data sources.

# Data Sources
- metadata: metadata for the batch like t0_dt, x_meters_center ....
- datetime: datetime information like 'hour_of_day'
- gsp: Grid Supply Point data from Sheffield Solar (e.g. the estimated total solar PV power generation for each
GSP region, and the geospatial shape of each GSP region).
- nwp: Numerical Weather predictions from UK Met Office
- pv: PV output data from pvoutput.org
- satellite: satellite data from ...
- sun: Sun position data (e.g. the estimated total solar PV power generation for each GSP region,
and the geospatial shape of each GSP region).
- topographic: Topographic data e.g. the elevation of the land.

# data_source.py

General class used for making a data source. It has the following functions
- get_batch: gets a whole batch of data for that data source
- datetime_index: gets the all available datatimes of the source
- get_example: gets one "example" (a single consecutive sequence). Each batch is made up of multiple examples.
- get_locations_for_batch: Samples the geospatial x,y location for each example in a batch. This is useful because,
 typically, we want a single DataSource to dictate the geospatial locations of the examples (for example,
 we want each example to be centered on the centroid of the grid supply point region). All the other
 `DataSources` will use these same geospatial locations.


# datasource_output.py

General pydantic model of output of the data source. Contains the following methods
- to_numpy: changes all data points to numpy objects
- split: converts a batch to a list of items
- join: joins list of items to one
- to_xr_dataset: changes data items to xarrays and returns a dataset
- from_xr_dataset: loads from an xarray dataset
- select_time_period: subselect data, depending on a time period

# <X> Data Source folder

Roughly each of the data source folders follows this pattern
- A class which defines how to load the data source, how to select for batches etc. This inherits from 'data_source.DataSource',
- A class which contains the output model of the data source, built from an xarray Dataset. This is the information used in the batches.
This inherits from 'datasource_output.DataSourceOutput'.
- A second class (pydantic) which moves the xarray Dataset to tensor fields. This will be used for training in ML models


# fake

`fake.py` has several function to create fake `Batch` data. This is useful for testing,
and I assume useful outside this module too.
