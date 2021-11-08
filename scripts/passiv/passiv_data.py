""" Script to merge Passiv data, ready for nowcasting"""
import time

import numpy as np
import pandas as pd

dir = "gs://solar-pv-nowcasting-data/PV/Passive/20211027_Passiv_PV_Data/5min"
file_output = "../passive.netcdf"

t = time.time()

months = pd.date_range("2021-01-01", "2021-12-01", freq="MS").strftime("%b").tolist()
years = ["2020", "2021"]


def load_file():
    """
    Load all passiv data

    Join years and months together
    """

    data_df = []
    for year in years:
        for month in months:
            file = f"{dir}/{year}/{month}.csv.gz"
            print(file)
            try:
                df = pd.read_csv(file)
                data_df.append(df)
            except Exception as e:  # noqa F481
                print(f"Could not load {file}")

    return pd.concat(data_df)


def format_df_to_xr(passive_5min_df):
    """Format 'pandas' to 'xarray'"""
    print("Format to xr")

    # change generation_wh to power_w
    passive_5min_df["power_w"] = passive_5min_df["generation_wh"] / (5 / 60)
    passive_5min_df["datetime"] = pd.to_datetime(passive_5min_df["timestamp"])

    # pivot on ssid
    passive_5min_on_ssid = passive_5min_df.pivot(
        index="datetime", columns="ss_id", values="power_w"
    )
    passive_5min_on_ssid.columns = [str(col) for col in passive_5min_on_ssid.columns]

    # change to xarray
    passive_5min_xr = passive_5min_on_ssid.to_xarray()

    # change to float32
    passive_5min_xr = passive_5min_xr.astype(np.float32)

    # change datetime to 'datetime'. Save it as UTC time
    datetime = passive_5min_on_ssid.index.tz_convert(None)
    passive_5min_xr["datetime"] = datetime

    return passive_5min_xr


def save_netcdf(passive_5min_xr, file_output):
    """
    Save to netcdf

    Each month of data seems to be about 11MB
    """
    print(f"Save file to {file_output}")
    encoding = {name: {"compression": "lzf"} for name in passive_5min_xr.data_vars}
    passive_5min_xr.to_netcdf(file_output, engine="h5netcdf", mode="w", encoding=encoding)


data_df = load_file()
data_xr = format_df_to_xr(passive_5min_df=data_df)
save_netcdf(data_xr, file_output)
