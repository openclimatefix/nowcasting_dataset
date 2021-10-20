""" Configuration model for the dataset """
from datetime import datetime
from typing import Optional

import git
from pathy import Pathy
from pydantic import BaseModel, Field
from pydantic import validator, root_validator

from nowcasting_dataset.consts import NWP_VARIABLE_NAMES
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES


class General(BaseModel):
    """General pydantic model"""

    name: str = Field("example", description="The name of this configuration file.")
    description: str = Field(
        "example configuration", description="Description of this confgiruation file"
    )
    cloud: str = Field(
        "gcp",
        description=(
            "local, gcp, or aws.  Deprecated.  Will be removed when issue"
            " https://github.com/openclimatefix/nowcasting_dataset/issues/153"
            " is implemented"
        ),
    )


class Git(BaseModel):
    """Git model"""

    hash: str = Field(..., description="The git hash has for when a dataset is created.")
    message: str = Field(..., description="The git message has for when a dataset is created.")
    committed_date: datetime = Field(
        ..., description="The git datestamp has for when a dataset is created."
    )


class DataSourceMixin(BaseModel):

    forecast_minutes: int = Field(
        None,
        ge=0,
        description="how many minutes to forecast in the future. "
        "If set to None, the value is defaulted to InputData.default_forecast_minutes",
    )
    history_minutes: int = Field(
        None,
        ge=0,
        description="how many historic minutes are used. "
        "If set to None, the value is defaulted to InputData.default_history_minutes",
    )


class PV(DataSourceMixin):
    """PV configuration model"""

    solar_pv_data_filename: str = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_timeseries_batch.nc",
        description=("The NetCDF file holding the solar PV power timeseries."),
    )
    solar_pv_metadata_filename: str = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_metadata.csv",
        description="The CSV file describing each PV system.",
    )


class Satellite(DataSourceMixin):
    """Satellite configuration model"""

    satellite_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr",
        description="The path which holds the satellite zarr.",
    )

    sat_channels: tuple = Field(
        SAT_VARIABLE_NAMES, description="the satellite channels that are used"
    )

    satellite_image_size_pixels: int = Field(64, description="the size of the satellite images")


class NWP(DataSourceMixin):
    """NWP configuration model"""

    nwp_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr",
        description="The path which holds the NWP zarr.",
    )

    nwp_channels: tuple = Field(NWP_VARIABLE_NAMES, description="the channels used in the nwp data")

    nwp_image_size_pixels: int = Field(64, description="the size of the nwp images")


class GSP(DataSourceMixin):
    """GSP configuration model"""

    gsp_zarr_path: str = Field("gs://solar-pv-nowcasting-data/PV/GSP/v0/pv_gsp.zarr")


class Topographic(DataSourceMixin):
    """Topographic configuration model"""

    topographic_filename: str = Field(
        "gs://solar-pv-nowcasting-data/Topographic/europe_dem_1km_osgb.tif",
        description="Path to the GeoTIFF Topographic data source",
    )


class Sun(DataSourceMixin):
    """Sun configuration model"""

    sun_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/Sun/v0/sun.zarr/",
        description="Path to the Sun data source i.e Azimuth and Elevation",
    )


class InputData(BaseModel):
    """
    Input data model

    All paths must include the protocol prefix.  For local files,
    it's sufficient to just start with a '/'.  For aws, start with 's3://',
    for gcp start with 'gs://'.
    """

    pv: PV = PV()
    satellite: Satellite = Satellite()
    nwp: NWP = NWP()
    gsp: GSP = GSP()
    topographic: Topographic = Topographic()
    sun: Sun = Sun()

    default_forecast_minutes: int = Field(
        60,
        ge=0,
        description="how many minutes to forecast in the future. "
        "This is defaulted to all the data sources if theya re not set. ",
    )
    default_history_minutes: int = Field(
        30,
        ge=0,
        description="how many historic minutes are used. "
        "This is defaulted to all the data sources if theya re not set.",
    )

    @root_validator
    def check_forecast_and_history_minutes(cls, values):
        """
        Set default history and forecast values, if needed.

        Run through the different data sources and  if the forecast or history minutes are not set,
        then set them to the default valyes
        """

        for data_source_name in ["pv", "nwp", "satellite", "gsp", "topographic", "sun"]:

            if values[data_source_name].forecast_minutes is None:
                values[data_source_name].forecast_minutes = values["default_forecast_minutes"]

            if values[data_source_name].history_minutes is None:
                values[data_source_name].history_minutes = values["default_history_minutes"]

        return values

    # add method to validate data sources
    # @validator('pv', always=True)
    # def name_must_contain_space(cls, v, values):
    #     print(values)
    #     print(v)
    #     if (v.forecast_minutes is None) and ('default_forecast_minutes' in values.keys()):
    #         v.forecast_minutes = values['default_forecast_minutes']
    #     if (v.history_minutes is None) and ('default_history_minutes' in values.keys()):
    #         v.history_minutes = values['default_history_minutes']
    #     return v


class OutputData(BaseModel):
    """Output data model"""

    filepath: str = Field(
        "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/",
        description=(
            "Where the data is saved to.  If this is running on the cloud then should include"
            " 'gs://' or 's3://'"
        ),
    )


class Process(BaseModel):
    """Pydantic model of how the data is processed"""

    seed: int = Field(1234, description="Random seed, so experiments can be repeatable")
    batch_size: int = Field(32, description="the number of examples per batch")
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
            "solar_pv_data_filename",
            "solar_pv_metadata_filename",
            "satellite_zarr_path",
            "nwp_zarr_path",
            "gsp_zarr_path",
        ]
        for attr_name in path_attrs:
            path = getattr(self.input_data, attr_name)
            path = base_path / path
            setattr(self.input_data, attr_name, path)
            print(path)


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
