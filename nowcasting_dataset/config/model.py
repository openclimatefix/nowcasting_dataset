""" Configuration model for the dataset """
from datetime import datetime
from typing import Optional

import git
from pathy import Pathy
from pydantic import BaseModel, Field
from pydantic import validator

from nowcasting_dataset.consts import NWP_VARIABLE_NAMES
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES


class General(BaseModel):
    """ General pydantic model """

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
    """ Git model """

    hash: str = Field(..., description="The git hash has for when a dataset is created.")
    message: str = Field(..., description="The git message has for when a dataset is created.")
    committed_date: datetime = Field(
        ..., description="The git datestamp has for when a dataset is created."
    )


class InputData(BaseModel):
    """
    Input data model

    All paths must include the protocol prefix.  For local files,
    it's sufficient to just start with a '/'.  For aws, start with 's3://',
    for gcp start with 'gs://'.
    """

    solar_pv_data_filename: str = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_timeseries_batch.nc",
        description=("The NetCDF file holding the solar PV power timeseries."),
    )
    solar_pv_metadata_filename: str = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_metadata.csv",
        description="The CSV file describing each PV system.",
    )

    satellite_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr",
        description="The path which holds the satellite zarr.",
    )

    nwp_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr",
        description="The path which holds the NWP zarr.",
    )

    gsp_zarr_path: str = Field("gs://solar-pv-nowcasting-data/PV/GSP/v0/pv_gsp.zarr")
    topographic_filename: str = Field(
        "gs://solar-pv-nowcasting-data/Topographic/europe_dem_1km_osgb.tif",
        description="Path to the GeoTIFF Topographic data source",
    )

    sun_zarr_path: str = Field(
        "gs://solar-pv-nowcasting-data/Sun/v0/sun.zarr/",
        description="Path to the Sun data source i.e Azimuth and Elevation",
    )


class OutputData(BaseModel):
    """ Output data model """

    filepath: str = Field(
        "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/",
        description=(
            "Where the data is saved to.  If this is running on the cloud then should include"
            " 'gs://' or 's3://'"
        ),
    )


class Process(BaseModel):
    """ Pydantic model of how the data is processed """

    seed: int = Field(1234, description="Random seed, so experiments can be repeatable")
    batch_size: int = Field(32, description="the number of examples per batch")
    upload_every_n_batches: int = Field(
        16,
        description=(
            "How frequently to move batches from the local temporary directory to the cloud bucket."
            "  If 0 then write batches directly to output_data.filepath, not to a temp directory."
        ),
    )
    forecast_minutes: int = Field(
        60, ge=0, description="how many minutes to forecast in the future"
    )
    history_minutes: int = Field(30, ge=0, description="how many historic minutes are used")
    satellite_image_size_pixels: int = Field(64, description="the size of the satellite images")
    nwp_image_size_pixels: int = Field(64, description="the size of the nwp images")

    sat_channels: tuple = Field(
        SAT_VARIABLE_NAMES, description="the satellite channels that are used"
    )
    nwp_channels: tuple = Field(NWP_VARIABLE_NAMES, description="the channels used in the nwp data")
    local_temp_path: str = Field("~/temp/")

    # TODO: Remove!
    @property
    def seq_length_30_minutes(self):
        """ How many steps are there in 30 minute datasets """
        return int((self.history_minutes + self.forecast_minutes) / 30 + 1)

    # TODO: Remove!
    @property
    def seq_length_5_minutes(self):
        """ How many steps are there in 5 minute datasets """
        return int((self.history_minutes + self.forecast_minutes) / 5 + 1)

    @validator("history_minutes")
    def history_minutes_divide_by_30(cls, v):
        """ Validate 'history_minutes' """
        assert v % 30 == 0  # this means it also divides by 5
        return v

    @validator("forecast_minutes")
    def forecast_minutes_divide_by_30(cls, v):
        """ Validate 'forecast_minutes' """
        assert v % 30 == 0  # this means it also divides by 5
        return v


class Configuration(BaseModel):
    """ Configuration model for the dataset """

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
