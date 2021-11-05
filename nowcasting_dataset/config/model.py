""" Configuration model for the dataset.

All paths must include the protocol prefix.  For local files,
it's sufficient to just start with a '/'.  For aws, start with 's3://',
for gcp start with 'gs://'.

This file is mostly about _configuring_ the DataSources.

Separate Pydantic models in
`nowcasting_dataset/data_sources/<data_source_name>/<data_source_name>_model.py`
are used to validate the values of the data itself.

"""
from datetime import datetime
from typing import Optional

import git
import pandas as pd
from pathy import Pathy
from pydantic import BaseModel, Field, root_validator, validator

# nowcasting_dataset imports
from nowcasting_dataset.consts import (
    DEFAULT_N_GSP_PER_EXAMPLE,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    NWP_VARIABLE_NAMES,
    SAT_VARIABLE_NAMES,
)
from nowcasting_dataset.dataset.split import split

IMAGE_SIZE_PIXELS_FIELD = Field(64, description="The number of pixels of the region of interest.")
METERS_PER_PIXEL_FIELD = Field(2000, description="The number of meters per pixel.")


class General(BaseModel):
    """General pydantic model"""

    name: str = Field("example", description="The name of this configuration file.")
    description: str = Field(
        "example configuration", description="Description of this configuration file"
    )


class Git(BaseModel):
    """Git model"""

    hash: str = Field(
        ..., description="The git hash of nowcasting_dataset when a dataset is created."
    )
    message: str = Field(..., description="The git message for when a dataset is created.")
    committed_date: datetime = Field(
        ..., description="The git datestamp for when a dataset is created."
    )


class DataSourceMixin(BaseModel):
    """Mixin class, to add forecast and history minutes"""

    forecast_minutes: int = Field(
        None,
        ge=0,
        description="how many minutes to forecast in the future. "
        "If set to None, the value is defaulted to InputData.default_forecast_minutes",
    )
    history_minutes: int = Field(
        None,
        ge=0,
        description="how many historic minutes to use. "
        "If set to None, the value is defaulted to InputData.default_history_minutes",
    )

    @property
    def seq_length_30_minutes(self):
        """How many steps are there in 30 minute datasets"""
        return int((self.history_minutes + self.forecast_minutes) / 30 + 1)

    @property
    def seq_length_5_minutes(self):
        """How many steps are there in 5 minute datasets"""
        return int((self.history_minutes + self.forecast_minutes) / 5 + 1)


class PV(DataSourceMixin):
    """PV configuration model"""

    pv_filename: str = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_timeseries_batch.nc",
        description=("The NetCDF file holding the solar PV power timeseries."),
    )
    pv_metadata_filename: str = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_metadata.csv",
        description="The CSV file describing each PV system.",
    )
    n_pv_systems_per_example: int = Field(
        DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
        description="The number of PV systems samples per example. "
        "If there are less in the ROI then the data is padded with zeros. ",
    )
    pv_image_size_pixels: int = IMAGE_SIZE_PIXELS_FIELD
    pv_meters_per_pixel: int = METERS_PER_PIXEL_FIELD


class Satellite(DataSourceMixin):
    """Satellite configuration model"""

    satellite_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr",  # noqa: E501
        description="The path which holds the satellite zarr.",
    )
    satellite_channels: tuple = Field(
        SAT_VARIABLE_NAMES, description="the satellite channels that are used"
    )
    satellite_image_size_pixels: int = IMAGE_SIZE_PIXELS_FIELD
    satellite_meters_per_pixel: int = METERS_PER_PIXEL_FIELD


class NWP(DataSourceMixin):
    """NWP configuration model"""

    nwp_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr",  # noqa: E501
        description="The path which holds the NWP zarr.",
    )
    nwp_channels: tuple = Field(NWP_VARIABLE_NAMES, description="the channels used in the nwp data")
    nwp_image_size_pixels: int = IMAGE_SIZE_PIXELS_FIELD
    nwp_meters_per_pixel: int = METERS_PER_PIXEL_FIELD


class GSP(DataSourceMixin):
    """GSP configuration model"""

    gsp_zarr_path: str = Field("gs://solar-pv-nowcasting-data/PV/GSP/v2/pv_gsp.zarr")
    n_gsp_per_example: int = Field(
        DEFAULT_N_GSP_PER_EXAMPLE,
        description="The number of GSP samples per example. "
        "If there are less in the ROI then the data is padded with zeros. ",
    )
    gsp_image_size_pixels: int = IMAGE_SIZE_PIXELS_FIELD
    gsp_meters_per_pixel: int = METERS_PER_PIXEL_FIELD

    @validator("history_minutes")
    def history_minutes_divide_by_30(cls, v):
        """Validate 'history_minutes'"""
        assert v % 30 == 0  # this means it also divides by 5
        return v

    @validator("forecast_minutes")
    def forecast_minutes_divide_by_30(cls, v):
        """Validate 'forecast_minutes'"""
        assert v % 30 == 0  # this means it also divides by 5
        return v


class Topographic(DataSourceMixin):
    """Topographic configuration model"""

    topographic_filename: str = Field(
        "gs://solar-pv-nowcasting-data/Topographic/europe_dem_1km_osgb.tif",
        description="Path to the GeoTIFF Topographic data source",
    )
    topographic_image_size_pixels: int = IMAGE_SIZE_PIXELS_FIELD
    topographic_meters_per_pixel: int = METERS_PER_PIXEL_FIELD


class Sun(DataSourceMixin):
    """Sun configuration model"""

    sun_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/Sun/v0/sun.zarr/",
        description="Path to the Sun data source i.e Azimuth and Elevation",
    )


class InputData(BaseModel):
    """
    Input data model.
    """

    pv: Optional[PV] = None
    satellite: Optional[Satellite] = None
    nwp: Optional[NWP] = None
    gsp: Optional[GSP] = None
    topographic: Optional[Topographic] = None
    sun: Optional[Sun] = None

    default_forecast_minutes: int = Field(
        60,
        ge=0,
        description="how many minutes to forecast in the future. "
        "This sets the default for all the data sources if they are not set.",
    )
    default_history_minutes: int = Field(
        30,
        ge=0,
        description="how many historic minutes are used. "
        "This sets the default for all the data sources if they are not set.",
    )
    data_source_which_defines_geospatial_locations: str = Field(
        "gsp",
        description=(
            "The name of the DataSource which will define the geospatial position of each example."
        ),
    )

    @property
    def default_seq_length_5_minutes(self):
        """How many steps are there in 5 minute datasets"""
        return int((self.default_history_minutes + self.default_forecast_minutes) / 5 + 1)

    @root_validator
    def set_forecast_and_history_minutes(cls, values):
        """
        Set default history and forecast values, if needed.

        Run through the different data sources and  if the forecast or history minutes are not set,
        then set them to the default values
        """
        # It would be much better to use nowcasting_dataset.data_sources.ALL_DATA_SOURCE_NAMES,
        # but that causes a circular import.
        ALL_DATA_SOURCE_NAMES = ("pv", "satellite", "nwp", "gsp", "topographic", "sun")
        enabled_data_sources = [
            data_source_name
            for data_source_name in ALL_DATA_SOURCE_NAMES
            if values[data_source_name] is not None
        ]

        for data_source_name in enabled_data_sources:
            if values[data_source_name].forecast_minutes is None:
                values[data_source_name].forecast_minutes = values["default_forecast_minutes"]

            if values[data_source_name].history_minutes is None:
                values[data_source_name].history_minutes = values["default_history_minutes"]

        return values

    @classmethod
    def set_all_to_defaults(cls):
        """Returns an InputData instance with all fields set to their default values.

        Used for unittests.
        """
        return cls(
            pv=PV(),
            satellite=Satellite(),
            nwp=NWP(),
            gsp=GSP(),
            topographic=Topographic(),
            sun=Sun(),
        )


class OutputData(BaseModel):
    """Output data model"""

    filepath: Pathy = Field(
        Pathy("gs://solar-pv-nowcasting-data/prepared_ML_training_data/v7/"),
        description=(
            "Where the data is saved to.  If this is running on the cloud then should include"
            " 'gs://' or 's3://'"
        ),
    )


class Process(BaseModel):
    """Pydantic model of how the data is processed"""

    seed: int = Field(1234, description="Random seed, so experiments can be repeatable")
    batch_size: int = Field(32, description="The number of examples per batch")
    t0_datetime_frequency: pd.Timedelta = Field(
        pd.Timedelta("5 minutes"),
        description=(
            "The temporal frequency at which t0 datetimes will be sampled."
            "  Can be any string that `pandas.Timedelta()` understands."
            "  For example, if this is set to '5 minutes', then, for each example, the t0 datetime"
            " could be at 0, 5, ..., 55 minutes past the hour.  If there are DataSources with a"
            " lower sample rate (e.g. half-hourly) then these lower-sample-rate DataSources will"
            " still produce valid examples.  For example, if a half-hourly DataSource is asked for"
            " an example with t0=12:05, history_minutes=60, forecast_minutes=60, then it will"
            " return data at 11:30, 12:00, 12:30, and 13:00."
        ),
    )
    split_method: split.SplitMethod = Field(
        split.SplitMethod.DAY,
        description=(
            "The method used to split the t0 datetimes into train, validation and test sets."
        ),
    )
    n_train_batches: int = 250
    n_validation_batches: int = 10
    n_test_batches: int = 10
    upload_every_n_batches: int = Field(
        16,
        description=(
            "How frequently to move batches from the local temporary directory to the cloud bucket."
            "  If 0 then write batches directly to output_data.filepath, not to a temp directory."
        ),
    )

    local_temp_path: str = Field("~/temp/")


class Configuration(BaseModel):
    """Configuration model for the dataset"""

    general: General = General()
    input_data: InputData = InputData()
    output_data: OutputData = OutputData()
    process: Process = Process()
    git: Optional[Git] = None

    def set_base_path(self, base_path: str):
        """Append base_path to all paths. Mostly used for testing."""
        base_path = Pathy(base_path)
        path_attrs = [
            "pv.pv_filename",
            "pv.pv_metadata_filename",
            "satellite.satellite_zarr_path",
            "nwp.nwp_zarr_path",
            "gsp.gsp_zarr_path",
        ]
        for cls_and_attr_name in path_attrs:
            cls_name, attribute = cls_and_attr_name.split(".")
            cls = getattr(self.input_data, cls_name)
            path = getattr(getattr(self.input_data, cls_name), attribute)
            path = base_path / path
            setattr(cls, attribute, path)
            setattr(self.input_data, cls_name, cls)


def set_git_commit(configuration: Configuration):
    """
    Set the git information in the configuration file

    Args:
        configuration: configuration object

    Returns: configuration object with git information

    """
    repo = git.Repo(search_parent_directories=True)

    git_details = Git(
        hash=repo.head.object.hexsha,
        committed_date=datetime.fromtimestamp(repo.head.object.committed_date),
        message=repo.head.object.message,
    )

    configuration.git = git_details

    return configuration
