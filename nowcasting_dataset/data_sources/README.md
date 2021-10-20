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
and hopefully useful outside this module too.


## How to add a new data source

Below is a checklist of general things to do when creating a new data source.
1. Assuming that data can not be made on the fly, create script to make process data.

2. Create folder in nowcasting/data_sources with the name of the new data source

3. Create a file called `<name>_datasource.py`. This file should contain class which
inherits `nowcasting_dataset.data_source.DataSource`. This class will need `get_example` method.
(there is also an option to use a `get_batch` method instead)
```python
def get_example(
    self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
) -> NewDataSource:
    """
    Get a single example

    Args:
        t0_dt: Current datetime for the example, unused
        x_meters_center: Center of the example in meters in the x direction in OSGB coordinates
        y_meters_center: Center of the example in meters in the y direction in OSGB coordinates

    Returns:
        Example containing xxx data for the selected area
    """
```

4. Create a file called `<name>_model.py` which a class with the name of the data soure. This class is an extension
of an xr.Dataset with some pydantic validation
```python
class NewDataSource(DataSourceOutput):
    """ Class to store <name> data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data
    __slots__ = ()
    _expected_dimensions = ("x", "y")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are not NaNs """
        assert (v.data != np.nan).all(), "Some data values are NaNs"
        return v

```
6. Add to new data source `Batch` object.

7. Add new data source to `nowcasting.dataset.datamodule.NowcastingDataModule`.

8. Add configuration data to configuration model, for example where the raw data is loaded from.

### Testing
1. Create a test to check that new data source is loaded correctly.
2. Create a script to make test data in `scritps/generate_data_for_tests`
3. Create a function to make a randomly generated xr.Dataset for generating fake data in `nowcasting.dataset.fake.py` \
and to batch fake function
4. Re-run script to generate batch test data.
