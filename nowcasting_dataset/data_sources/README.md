This folder contains the code for the different data sources.

# Data Sources
- metadata: metadata for the batch like t0_dt, x_centers_osgb ....
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
- get_batch: gets a whole batch of data for that data source. The list of 'xr.Dataset' examples are converted to
one xr.Dataset by changing the coordinates to indexes, and then joining the examples along an extra dimension.
- datetime_index: gets the all available datatimes of the source
- get_example: gets one "example" (a single consecutive sequence). Each batch is made up of multiple examples.
  Each example is a 'xr.Dataset'
- get_locations_for_batch: Samples the geospatial x,y location for each example in a batch. This is useful because,
 typically, we want a single DataSource to dictate the geospatial locations of the examples (for example,
 we want each example to be centered on the centroid of the grid supply point region). All the other
 `DataSources` will use these same geospatial locations.


# datasource_output.py

General pydantic model of output of the data source. Contains the following methods
- save_netcdf: save to netcdf file
- check_nan_and_inf: check if any values are nans or infinite
- check_dataset_greater_than_or_equal_to: check values are >= a value
- check_dataset_less_than_or_equal_to: check values are <= a value
- check_dataset_not_equal: check values are !>= a value
- check_data_var_dim: check the dimensions of a data variable

# <X> Data Source folder

Roughly each of the data source folders follows this pattern
- A class which defines how to load the data source, how to select for batches etc. This inherits from 'data_source.DataSource',
- A class which contains the output model of the data source, built from a xarray Dataset. This is the information used in the batches.
This inherits from 'datasource_output.DataSourceOutput'.


# fake

`fake.py` has several function to create fake `Batch` data. This is useful for testing,
and hopefully useful outside this module too.


## How to add a new data source

Below is a checklist of general things to do when creating a new data source.
1. Assuming that data cannot be made on the fly, create script to process data.

2. Create folder in nowcasting/data_sources with the name of the new data source

3. Create a file called `<name>_data_source.py`. This file should contain a class which
inherits from `nowcasting_dataset.data_source.DataSource`. This class will need to implement the `get_example` method.
(there is also an option to use a `get_batch` method instead)
```python
def get_example(
    self, t0_datetime_utc: pd.Timestamp, x_center_osgb: Number, y_center_osgb: Number
) -> NewDataSource:
    """
    Get a single example

    Args:
        t0_datetime_utc: At inference time, t0 can be thought of as "now":
            it is the time of the most recent observation.
            Anything after t0 is "future". Anything before t0 is "history".
        x_center_osgb: Center of the example in meters in the x direction in OSGB coordinates
        y_center_osgb: Center of the example in meters in the y direction in OSGB coordinates

    Returns:
        Example containing xxx data for the selected area
    """
```

4. Create a file called `<name>_model.py` which a class with the name of the data source. This class is an extension
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
6. Add new data source to the `nowcasting_dataset.dataset.batch.Batch` class.

7. Add new data source to `nowcasting.dataset.datamodule.NowcastingDataModule`.

8. Add configuration data to configuration model (`nowcasting_dataset/config/model.py`), for example where the raw data is loaded from.

### Testing
1. Create a test to check that new data source is loaded correctly.
2. Create a script to make test data in `scripts/generate_data_for_tests`
3. Create a function to make a randomly generated xr.Dataset for generating fake data in `nowcasting.dataset.fake.py` \
and to batch fake function
