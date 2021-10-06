from datetime import datetime

from nowcasting_dataset.data_sources.satellite.satellite_data_source import SatelliteDataSource

s = SatelliteDataSource(
    filename="gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/"
    "all_zarr_int16_single_timestep.zarr",
    history_len=6,
    forecast_len=12,
    convert_to_numpy=True,
    image_size_pixels=64,
    meters_per_pixel=2000,
    n_timesteps_per_batch=32,
)

s.open()
start_dt = datetime.fromisoformat("2019-01-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2019-01-02 00:00:00.000+00:00")

data_xarray = s._data
data_xarray = data_xarray.sel(time=slice(start_dt, end_dt))
data_xarray = data_xarray.sel(variable=["HRV"])
data_xarray = data_xarray.sel(x=slice(122000, 122001))

data_df = data_xarray.to_dataframe()
