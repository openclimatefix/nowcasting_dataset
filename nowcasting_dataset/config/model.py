from pydantic import BaseModel, Field

from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES


class General(BaseModel):

    name: str = Field("example", description="The name of this configuration file.")
    description: str = Field("example configuration", description="Description of this confgiruation file")


class InputData(BaseModel):
    bucket: str = Field("solar-pv-nowcasting-data", description="The gcp bucket used to load the data.")

    solar_pv_path: str = Field("PV/PVOutput.org", description="TODO")
    solar_pv_data_filename: str = Field("UK_PV_timeseries_batch.nc", description="TODO")
    solar_pv_metadata_filename: str = Field("UK_PV_metadata.csv", description="TODO")

    satelite_filename: str = Field(
        "satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr", description="TODO"
    )

    npw_base_path: str = Field(
        "NWP/UK_Met_Office/UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr",
        description="TODO",
    )


class OutputData(BaseModel):
    filepath: str = Field("prepared_ML_training_data/v4/", description="Where the data is saved")


class Process(BaseModel):
    batch_size: int = Field(32, description="the batch size of the data")
    forecast_length: int = Field(12, description="how many time steps to forecast in the future")
    history_length: int = Field(6, description="how many historic times teps are used")
    image_size_pixels: int = Field(64, description="the size of the satelite images")

    sat_channels: tuple = Field(SAT_VARIABLE_NAMES, description="the satelite channels that are used")
    nwp_channels: tuple = Field(NWP_VARIABLE_NAMES, description="the channels used in the nwp data")

    precision: int = Field(16, description="what precision to use")
    val_check_interval: int = Field(1000, description="TODO")


class Configuration(BaseModel):

    general: General = General()
    input_data: InputData = InputData()
    output_data: OutputData = OutputData()
    process: Process = Process()
