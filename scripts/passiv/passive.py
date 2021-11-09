""" Test script, to see if passive data works"""
from datetime import datetime

from nowcasting_dataset.data_sources.pv.pv_data_source import PVDataSource
from nowcasting_dataset.time import time_periods_to_datetime_index

output_dir = "gs://solar-pv-nowcasting-data/PV/Passive/ocf_formatted/v0"

filename = output_dir + "/passive.netcdf"
filename_metadata = output_dir + "/system_metadata.csv"


pv = PVDataSource(
    filename=filename,
    metadata_filename=filename_metadata,
    start_dt=datetime(2020, 3, 28),
    end_dt=datetime(2020, 4, 1),
    history_minutes=60,
    forecast_minutes=30,
    image_size_pixels=64,
    meters_per_pixel=2000,
)

datetime = pv.get_contiguous_t0_time_periods()

times = time_periods_to_datetime_index(datetime, freq="5T")
x_locations, y_locations = pv.get_locations(times)

i = 150
example = pv.get_example(
    t0_dt=times[i], x_meters_center=x_locations[i], y_meters_center=y_locations[i]
)
