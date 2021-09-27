from pydantic import BaseModel, Field
from typing import Optional
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES

from datetime import datetime
import git


class General(BaseModel):
    name: str = Field("example", description="The name of this configuration file.")
    description: str = Field(
        "example configuration", description="Description of this confgiruation file"
    )
    cloud: str = Field("gcp", description="gcp, aws, or local")


class Git(BaseModel):
    hash: str = Field(..., description="The git hash has for when a dataset is created.")
    message: str = Field(..., description="The git message has for when a dataset is created.")
    committed_date: datetime = Field(
        ..., description="The git datestamp has for when a dataset is created."
    )


class InputData(BaseModel):
    base_path_or_bucket: str = Field(
        "solar-pv-nowcasting-data",
        description=(
            "If cloud==local then this should be the absolute path which holds nwp_zarr_path,"
            " satellite_zarr_path, solar_pv_path, and output_data.filepath.  If cloud=={aws,gcp}"
            " then this should be the bucket name, including 's3://' or 'gs://'."
        ),
    )

    solar_pv_path: str = Field(
        "PV/PVOutput.org",
        description=(
            "The path that contains solar_pv_data_filename and solar_pv_metadata_filename"
        ),
    )
    solar_pv_data_filename: str = Field(
        "UK_PV_timeseries_batch.nc",
        description=("The NetCDF file holding the solar PV power timeseries."),
    )
    solar_pv_metadata_filename: str = Field(
        "UK_PV_metadata.csv", description="The CSV file describing each PV system."
    )

    satellite_zarr_path: str = Field(
        "satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr",
        description="The path within base_path_or_bucket which holds the satellite zarr.",
    )

    nwp_zarr_path: str = Field(
        "NWP/UK_Met_Office/UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr",
        description="The path within base_path_or_bucket which holds the NWP zarr.",
    )

    gsp_zarr_path: str = Field("PV/GSP/v0/pv_gsp.zarr")


class OutputData(BaseModel):
    filepath: str = Field(
        "prepared_ML_training_data/v5/",
        description=(
            "Where the data is saved to.  If this is running on the cloud then should include"
            " 'gs://' or 's3://'"
        ),
    )


class Process(BaseModel):
    seed: int = Field(1234, description="Random seed, so experiments can be repeatable")
    batch_size: int = Field(32, description="the number of examples per batch")
    upload_every_n_batches: int = Field(
        16,
        description=(
            "How frequently to move batches from the local temporary directory to the cloud bucket."
            "  If 0 then write batches directly to output_data.filepath, not to a temp directory."
        ),
    )
    forecast_minutes: int = Field(60, description="how many minutes to forecast in the future")
    history_minutes: int = Field(30, description="how many historic minutes are used")
    satellite_image_size_pixels: int = Field(64, description="the size of the satellite images")
    nwp_image_size_pixels: int = Field(2, description="the size of the nwp images")

    sat_channels: tuple = Field(
        SAT_VARIABLE_NAMES, description="the satellite channels that are used"
    )
    nwp_channels: tuple = Field(NWP_VARIABLE_NAMES, description="the channels used in the nwp data")
    local_temp_path: str = Field("~/temp/")


class Configuration(BaseModel):
    general: General = General()
    input_data: InputData = InputData()
    output_data: OutputData = OutputData()
    process: Process = Process()
    git: Optional[Git] = None


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
