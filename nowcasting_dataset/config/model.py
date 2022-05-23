""" Configuration model for the dataset.

All paths must include the protocol prefix.  For local files,
it's sufficient to just start with a '/'.  For aws, start with 's3://',
for gcp start with 'gs://'.

This file is mostly about _configuring_ the DataSources.

Separate Pydantic models in
`nowcasting_dataset/data_sources/<data_source_name>/<data_source_name>_model.py`
are used to validate the values of the data itself.

"""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import git
import numpy as np
import pandas as pd
from pathy import Pathy
from pydantic import BaseModel, Field, root_validator, validator

# nowcasting_dataset imports
from nowcasting_dataset.consts import (
    DEFAULT_N_GSP_PER_EXAMPLE,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    NWP_VARIABLE_NAMES,
    PV_PROVIDERS,
    SAT_VARIABLE_NAMES,
)
from nowcasting_dataset.dataset.split import split

IMAGE_SIZE_PIXELS = 64
IMAGE_SIZE_PIXELS_FIELD = Field(
    IMAGE_SIZE_PIXELS, description="The number of pixels of the region of interest."
)
METERS_PER_PIXEL_FIELD = Field(2000, description="The number of meters per pixel.")

logger = logging.getLogger(__name__)


class Base(BaseModel):
    """Pydantic Base model where no extras can be added"""

    class Config:
        """config class"""

        extra = "forbid"  # forbid use of extra kwargs


class General(Base):
    """General pydantic model"""

    name: str = Field("example", description="The name of this configuration file.")
    description: str = Field(
        "example configuration", description="Description of this configuration file"
    )


class Git(Base):
    """Git model"""

    hash: str = Field(
        ..., description="The git hash of nowcasting_dataset when a dataset is created."
    )
    message: str = Field(..., description="The git message for when a dataset is created.")
    committed_date: datetime = Field(
        ..., description="The git datestamp for when a dataset is created."
    )


class DataSourceMixin(Base):
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

    log_level: str = Field(
        "DEBUG",
        description="The logging level for this data source. T"
        "his is the default value and can be set in each data source",
    )

    @property
    def seq_length_30_minutes(self):
        """How many steps are there in 30 minute datasets"""
        return int(np.ceil((self.history_minutes + self.forecast_minutes) / 30 + 1))

    @property
    def seq_length_5_minutes(self):
        """How many steps are there in 5 minute datasets"""
        return int(np.ceil((self.history_minutes + self.forecast_minutes) / 5 + 1))

    @property
    def seq_length_60_minutes(self):
        """How many steps are there in 60 minute datasets"""
        return int(np.ceil((self.history_minutes + self.forecast_minutes) / 60 + 1))

    @property
    def history_seq_length_5_minutes(self):
        """How many historical steps are there in 5 minute datasets"""
        return int(np.ceil(self.history_minutes / 5))

    @property
    def history_seq_length_30_minutes(self):
        """How many historical steps are there in 30 minute datasets"""
        return int(np.ceil(self.history_minutes / 30))

    @property
    def history_seq_length_60_minutes(self):
        """How many historical steps are there in 60 minute datasets"""
        return int(np.ceil(self.history_minutes / 60))


class TimeResolutionMixin(Base):
    """Time resolution mix in"""

    # TODO: Issue #584: Rename to `sample_period_minutes`
    time_resolution_minutes: int = Field(
        5,
        description="The temporal resolution (in minutes) of the satellite images."
        "Note that this needs to be divisible by 5.",
    )

    @validator("time_resolution_minutes")
    def forecast_minutes_divide_by_5(cls, v):
        """Validate 'forecast_minutes'"""
        assert v % 5 == 0, f"The time resolution ({v}) is not divisible by 5"
        return v


class StartEndDatetimeMixin(Base):
    """Mixin class to add start and end date"""

    start_datetime: datetime = Field(
        datetime(2020, 1, 1),
        description="Load date from data sources from this date. "
        "If None, this will get overwritten by InputData.start_date. ",
    )
    end_datetime: datetime = Field(
        datetime(2021, 9, 1),
        description="Load date from data sources up to this date. "
        "If None, this will get overwritten by InputData.start_date. ",
    )

    @root_validator
    def check_start_and_end_datetime(cls, values):
        """
        Make sure start datetime is before end datetime
        """

        start_datetime = values["start_datetime"]
        end_datetime = values["end_datetime"]

        # check start datetime is less than end datetime
        if start_datetime >= end_datetime:
            message = (
                f"Start datetime ({start_datetime}) "
                f"should be less than end datetime ({end_datetime})"
            )
            logger.error(message)
            assert Exception(message)

        return values


class PVFiles(BaseModel):
    """Model to hold pv file and metadata file"""

    pv_filename: str = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_timeseries_batch.nc",
        description="The NetCDF files holding the solar PV power timeseries.",
    )
    pv_metadata_filename: str = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_metadata.csv",
        description="Tthe CSV files describing each PV system.",
    )

    label: str = Field("pvoutput", description="Label of where the pv data came from")

    @validator("label")
    def v_label0(cls, v):
        """Validate 'label'"""
        assert v in PV_PROVIDERS
        return v


class PV(DataSourceMixin, StartEndDatetimeMixin):
    """PV configuration model"""

    pv_files_groups: List[PVFiles] = [PVFiles()]

    n_pv_systems_per_example: int = Field(
        DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
        description="The number of PV systems samples per example. "
        "If there are less in the ROI then the data is padded with zeros. ",
    )
    pv_image_size_pixels_height: int = IMAGE_SIZE_PIXELS_FIELD
    pv_image_size_pixels_width: int = IMAGE_SIZE_PIXELS_FIELD
    pv_meters_per_pixel: int = METERS_PER_PIXEL_FIELD
    get_center: bool = Field(
        False,
        description="If the batches are centered on one PV system (or not). "
        "The other options is to have one GSP at the center of a batch. "
        "Typically, get_center would be set to true if and only if "
        "PVDataSource is used to define the geospatial positions of each example.",
    )

    pv_filename: str = Field(
        None,
        description="The NetCDF files holding the solar PV power timeseries.",
    )
    pv_metadata_filename: str = Field(
        None,
        description="Tthe CSV files describing each PV system.",
    )

    is_live: bool = Field(
        False, description="Option if to use live data from the nowcasting pv database"
    )

    live_interpolate_minutes: int = Field(
        30, description="The number of minutes we allow PV data to interpolate"
    )
    live_load_extra_minutes: int = Field(
        0,
        description="The number of extra minutes in the past we should load. Then the recent "
        "values can be interpolated, and the extra minutes removed. This is "
        "because some live data takes ~1 hour to come in.",
    )

    @classmethod
    def model_validation(cls, v):
        """Move old way of storing filenames to new way"""

        if (v.pv_filename is not None) and (v.pv_metadata_filename is not None):
            logger.warning(
                "Loading pv files the old way, and moving them the new way. "
                "Please update configuration file"
            )
            label = "pvoutput" if "pvoutput" in v.pv_filename.lower() else "passiv"
            pv_file = PVFiles(
                pv_filename=v.pv_filename, pv_metadata_filename=v.pv_metadata_filename, label=label
            )
            v.pv_files_groups = [pv_file]
            v.pv_filename = None
            v.pv_metadata_filename = None

        return v


class Satellite(DataSourceMixin, TimeResolutionMixin):
    """Satellite configuration model"""

    satellite_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr",  # noqa: E501
        description="The path which holds the satellite zarr.",
    )
    satellite_channels: tuple = Field(
        SAT_VARIABLE_NAMES[1:], description="the satellite channels that are used"
    )
    satellite_image_size_pixels_height: int = Field(
        IMAGE_SIZE_PIXELS_FIELD.default // 3,
        description="The number of pixels of the height of the region of interest"
        " for non-HRV satellite channels.",
    )
    satellite_image_size_pixels_width: int = Field(
        IMAGE_SIZE_PIXELS_FIELD.default // 3,
        description="The number of pixels of the width of the region "
        "of interest for non-HRV satellite channels.",
    )
    satellite_meters_per_pixel: int = Field(
        METERS_PER_PIXEL_FIELD.default * 3,
        description="The number of meters per pixel for non-HRV satellite channels.",
    )

    keep_dawn_dusk_hours: int = Field(
        0,
        description="The number hours around dawn and dusk that should be keep. "
        "I.e 'keep_dawn_dusk_hours'=2,"
        " then if dawn if 07.00, "
        " then data is keep from 06.00",
    )

    is_live: bool = Field(
        False,
        description="Option if to use live data from the satelite consumer. "
        "This is useful becasuse the data is about ~30 mins behind, "
        "so we need to expect that",
    )

    live_delay_minutes: int = Field(
        30, description="The expected delay in minutes of the satellite data"
    )


class HRVSatellite(DataSourceMixin, TimeResolutionMixin):
    """Satellite configuration model for HRV data"""

    hrvsatellite_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr",  # noqa: E501
        description="The path which holds the satellite zarr.",
    )

    hrvsatellite_channels: tuple = Field(
        SAT_VARIABLE_NAMES[0:1], description="the satellite channels that are used"
    )
    # HRV is 3x the resolution, so to cover the same area, its 1/3 the meters per pixel and 3
    # time the number of pixels
    hrvsatellite_image_size_pixels_height: int = IMAGE_SIZE_PIXELS_FIELD
    hrvsatellite_image_size_pixels_width: int = IMAGE_SIZE_PIXELS_FIELD
    hrvsatellite_meters_per_pixel: int = METERS_PER_PIXEL_FIELD

    keep_dawn_dusk_hours: int = Field(
        0,
        description="The number hours around dawn and dusk that should be keep. "
        "I.e 'keep_dawn_dusk_hours'=2,"
        " then if dawn if 07.00, "
        " then data is keep from 06.00",
    )

    is_live: bool = Field(
        False,
        description="Option if to use live data from the satelite consumer. "
        "This is useful becasuse the data is about ~30 mins behind, "
        "so we need to expect that",
    )

    live_delay_minutes: int = Field(
        30, description="The expected delay in minutes of the satellite data"
    )


class OpticalFlow(DataSourceMixin, TimeResolutionMixin):
    """Optical Flow configuration model"""

    opticalflow_zarr_path: str = Field(
        "",
        description=(
            "The satellite Zarr data to use. If in doubt, use the same value as"
            " satellite.satellite_zarr_path."
        ),
    )

    # history_minutes, set in DataSourceMixin.
    # Duration of historical data to use when computing the optical flow field.
    # For example, set to 5 to use just two images: the t-1 and t0 images.  Set to 10 to
    # compute the optical flow field separately for the image pairs (t-2, t-1), and
    # (t-1, t0) and to use the mean optical flow field.

    # forecast_minutes, set in DataSourceMixin.
    # Duration of the optical flow predictions.

    opticalflow_meters_per_pixel: int = METERS_PER_PIXEL_FIELD
    opticalflow_input_image_size_pixels_height: int = Field(
        IMAGE_SIZE_PIXELS * 2,
        description=(
            "The *input* image height (i.e. the image size to load off disk)."
            " This should be larger than output_image_size_pixels to provide sufficient border to"
            " mean that, even after the image has been flowed, all edges of the output image are"
            " real pixels values, and not NaNs."
        ),
    )
    opticalflow_output_image_size_pixels_height: int = Field(
        IMAGE_SIZE_PIXELS,
        description=(
            "The height of the images after optical flow has been applied. The output image is a"
            " center-crop of the input image, after it has been flowed."
        ),
    )
    opticalflow_input_image_size_pixels_width: int = Field(
        IMAGE_SIZE_PIXELS * 2,
        description=(
            "The *input* image width (i.e. the image size to load off disk)."
            " This should be larger than output_image_size_pixels to provide sufficient border to"
            " mean that, even after the image has been flowed, all edges of the output image are"
            " real pixels values, and not NaNs."
        ),
    )
    opticalflow_output_image_size_pixels_width: int = Field(
        IMAGE_SIZE_PIXELS,
        description=(
            "The width of the images after optical flow has been applied. The output image is a"
            " center-crop of the input image, after it has been flowed."
        ),
    )
    opticalflow_channels: tuple = Field(
        SAT_VARIABLE_NAMES[1:], description="the satellite channels that are used"
    )
    opticalflow_source_data_source_class_name: str = Field(
        "SatelliteDataSource",
        description=(
            "Either SatelliteDataSource or HRVSatelliteDataSource."
            "  The name of the DataSource that will load the satellite images."
        ),
    )


class NWP(DataSourceMixin):
    """NWP configuration model"""

    # TODO change to nwp_path, as it could be a netcdf now.
    # https://github.com/openclimatefix/nowcasting_dataset/issues/582
    nwp_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr",  # noqa: E501
        description="The path which holds the NWP zarr.",
    )
    nwp_channels: tuple = Field(NWP_VARIABLE_NAMES, description="the channels used in the nwp data")
    nwp_image_size_pixels_height: int = IMAGE_SIZE_PIXELS_FIELD
    nwp_image_size_pixels_width: int = IMAGE_SIZE_PIXELS_FIELD
    nwp_meters_per_pixel: int = METERS_PER_PIXEL_FIELD


class GSP(DataSourceMixin, StartEndDatetimeMixin):
    """GSP configuration model"""

    gsp_zarr_path: str = Field("gs://solar-pv-nowcasting-data/PV/GSP/v2/pv_gsp.zarr")
    n_gsp_per_example: int = Field(
        DEFAULT_N_GSP_PER_EXAMPLE,
        description="The number of GSP samples per example. "
        "If there are less in the ROI then the data is padded with zeros. ",
    )
    gsp_image_size_pixels_height: int = IMAGE_SIZE_PIXELS_FIELD
    gsp_image_size_pixels_width: int = IMAGE_SIZE_PIXELS_FIELD
    gsp_meters_per_pixel: int = METERS_PER_PIXEL_FIELD
    metadata_only: bool = Field(False, description="Option to only load metadata.")

    is_live: bool = Field(
        False, description="Option if to use live data from the nowcasting GSP/Forecast database"
    )

    live_interpolate_minutes: int = Field(
        60, description="The number of minutes we allow GSP data to be interpolated"
    )
    live_load_extra_minutes: int = Field(
        60,
        description="The number of extra minutes in the past we should load. Then the recent "
        "values can be interpolated, and the extra minutes removed. This is "
        "because some live data takes ~1 hour to come in.",
    )

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
    topographic_image_size_pixels_height: int = IMAGE_SIZE_PIXELS_FIELD
    topographic_image_size_pixels_width: int = IMAGE_SIZE_PIXELS_FIELD
    topographic_meters_per_pixel: int = METERS_PER_PIXEL_FIELD


class Sun(DataSourceMixin):
    """Sun configuration model"""

    sun_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/Sun/v1/sun.zarr/",
        description="Path to the Sun data source i.e Azimuth and Elevation",
    )


class InputData(Base):
    """
    Input data model.
    """

    pv: Optional[PV] = None
    satellite: Optional[Satellite] = None
    hrvsatellite: Optional[HRVSatellite] = None
    opticalflow: Optional[OpticalFlow] = None
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
        ALL_DATA_SOURCE_NAMES = (
            "pv",
            "hrvsatellite",
            "satellite",
            "nwp",
            "gsp",
            "topographic",
            "sun",
            "opticalflow",
        )
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
            hrvsatellite=HRVSatellite(),
            nwp=NWP(),
            gsp=GSP(),
            topographic=Topographic(),
            sun=Sun(),
            opticalflow=OpticalFlow(),
        )


class OutputData(Base):
    """Output data model"""

    filepath: Union[str, Pathy] = Field(
        Pathy("gs://solar-pv-nowcasting-data/prepared_ML_training_data/v7/"),
        description=(
            "Where the data is saved to.  If this is running on the cloud then should include"
            " 'gs://' or 's3://'"
        ),
    )

    @validator("filepath")
    def filepath_pathy(cls, v):
        """Make sure filepath is a Pathy object"""
        return Pathy(v)


class Process(Base):
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
        split.SplitMethod.DAY_RANDOM_TEST_DATE,
        description=(
            "The method used to split the t0 datetimes into train, validation and test sets."
            " If the split method produces no t0 datetimes for any split_name, then"
            " n_<split_name>_batches must also be set to 0."
        ),
    )

    train_test_validation_split: List[int] = Field(
        [10, 1, 1],
        description=(
            "The ratio of how the entire dataset is split. "
            "Note different split methods interact different with these numbers"
        ),
    )

    n_train_batches: int = Field(
        250,
        description=(
            "Number of train batches.  Must be 0 if split_method produces no t0 datetimes for"
            " the train split"
        ),
    )
    n_validation_batches: int = Field(
        0,  # Currently not using any validation batches!
        description=(
            "Number of validation batches.  Must be 0 if split_method produces no t0 datetimes for"
            " the validation split"
        ),
    )
    n_test_batches: int = Field(
        10,
        description=(
            "Number of test batches.  Must be 0 if split_method produces no t0 datetimes for"
            " the test split."
        ),
    )
    upload_every_n_batches: int = Field(
        16,
        description=(
            "How frequently to move batches from the local temporary directory to the cloud bucket."
            "  If 0 then write batches directly to output_data.filepath, not to a temp directory."
        ),
    )

    local_temp_path: Path = Field(
        Path("~/temp/").expanduser(),
        description=(
            "This is only necessary if using a VM on a public cloud and when the finished batches"
            " will be uploaded to a cloud bucket. This is the local temporary path on the VM."
            "  This will be emptied."
        ),
    )

    @validator("local_temp_path")
    def local_temp_path_to_path_object_expanduser(cls, v):
        """
        Convert the local path to Path

        Convert the path in string format to a `pathlib.PosixPath` object
        and call `expanduser` on the latter.
        """
        return Path(v).expanduser()


class Configuration(Base):
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
            "hrvsatellite.hrvsatellite_zarr_path",
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
