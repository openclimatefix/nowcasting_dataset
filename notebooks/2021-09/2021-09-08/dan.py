import xarray as xr

filename = "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr/"
# filename2 = 'gs://solar-pv-nowcasting-data/prepared_ML_training_data/testing.zarr'
ds = xr.open_dataset(filename, engine="zarr")
