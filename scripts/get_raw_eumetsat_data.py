############
# Pull raw pv satellite data from EUMetSat
#
# 2021-09-28
# Jacob Bieker
#
# The data is about 1MB for a month of data
############
from datetime import datetime

import fsspec
import pytz
import yaml
from satip import eumetsat
import pandas as pd

from typing import Optional, List, Union, Tuple

import re
from datetime import datetime, timedelta

from pathlib import Path
from nowcasting_dataset.cloud.local import delete_all_files_in_temp_path
from nowcasting_dataset.cloud.gcp import gcp_upload_and_delete_local_files
import logging
import click

import glob

NATIVE_FILESIZE_MB = 102.210123

format_dt_str = lambda dt: pd.to_datetime(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_date(ctx, param, value):
    try:
        return format_dt_str(value)
    except ValueError:
        raise click.BadParameter("Date must be in format accepted by pd.to_datetime()")


@click.command()
@click.option(
    "--download_directory",
    "-dir",
    default="./",
    help="Where to download the data to. Also where the script searches for previously downloaded data.",
)
@click.option(
    "--start_date",
    "--start",
    prompt="Starting date to download data, in format accepted by pd.to_datetime()",
    callback=validate_date,
)
@click.option(
    "--end_date",
    "--end",
    prompt="Ending date to download data, in format accepted by pd.to_datetime()",
    callback=validate_date,
)
@click.option(
    "--backfill",
    "-b",
    prompt="Whether to download any missing data from the start date of the data on disk to the end date",
    is_flag=True,
)
@click.option(
    "--user_key",
    "--key",
    default=None,
    help="The User Key for EUMETSAT access",
)
@click.option(
    "--user_secret",
    "--secret",
    default=None,
    help="The User secret for EUMETSAT access",
)
@click.option(
    "--auth_filename",
    default="auth.json",
    help="The auth file containing the user key and access key for EUMETSAT access",
)
@click.option("--bandwidth_limit", "--bw_limit", prompt="Bandwidth limit, in MB/sec", type=float)
def download_eumetsat_data(
    download_directory,
    start_date: datetime,
    end_date: datetime,
    backfill: bool = False,
    bandwidth_limit: Optional[float] = None,
    user_key: Optional[str] = None,
    user_secret: Optional[str] = None,
    auth_filename: Optional[str] = None,
):
    """
    Downloads EUMETSAT RSS and Cloud Masks to the given directory,
     checking first to see if the requested files are already downloaded

    Args:
        download_directory:
        start_date:
        end_date:
        backfill:
        bandwidth_limit:
        user_key:
        user_secret:
        auth_filename: Path to a file containing the user_secret and user_key

    Returns:

    """
    # Get authentication
    if auth_filename is not None:
        user_key, user_secret = load_key_secret(auth_filename)

    # Get already downloaded timestamps
    times_to_use = determine_datetimes_to_download_files(download_directory)

    # Download the data
    dm = eumetsat.DownloadManager(
        user_key, user_secret, download_directory, download_directory, download_directory
    )

    for start_time, end_time in times_to_use:
        # Because the satellite data is off by 1 minute
        # Need to download RSS images from 23:59 the day before start_time
        # And end 23:59 the day of
        # Derived products are at the 5min, so need to go from 00:00 to 23:59 each day
        dm.download_date_range(
            start_time,
            end_time,
            product_id="EO:EUM:DAT:MSG:RSS-CLM",  # Cloud mask
        )
        dm.download_date_range(
            start_time - pd.to_timedelta("1 minute"),
            end_time - pd.to_timedelta("1 minute"),
            product_id="EO:EUM:DAT:MSG:MSG15-RSS",  # RSS Image
        )

        # Sanity check, able to open/right size

        # Move to the correct directory

    # Sanity check each time range after downloaded

    #
    pass


def load_key_secret(filename) -> Tuple[str, str]:
    """
    Load user secret and key stored in a yaml file

    Args:
        filename: Filename to read

    Returns:
        The user secret and key

    """
    with fsspec.open(filename, mode="r") as f:
        keys = yaml.load(f, Loader=yaml.FullLoader)
        return keys["key"], keys["secret"]


def determine_datetimes_to_download_files(
    directory, start_date, end_date, product_id
) -> List[Tuple[datetime, datetime]]:
    """
    Check the given directory, and sub-directories, for all downloaded files.

    Args:
        directory: The top-level directory to check in
        start_date:
        end_date:
        product_id:

    Returns:
        List of tuples of datetimes giving the ranges of time to download

    """
    # TODO get list of all files in directories
    # Get all days from start_date to end_date
    day_split = pd.date_range(start_date, end_date, freq="D")

    # Go through files and get all examples in each
    fs = fsspec.filesystem("file")  # TODO Update for other ones?

    # Go through each directory, and for each day, list any missing data
    missing_rss_timesteps = []
    missing_cloud_mask_timesteps = []
    for day in day_split:
        day_string = day.strftime(format="%Y/%m/%d")
        rss_images = fs.glob(day_string + "*.nat")
        cloud_masks = fs.glob(day_string + "*.grb")
        missing_rss_timesteps = missing_rss_timesteps + get_missing_datetimes_from_list_of_files(
            rss_images
        )
        missing_cloud_mask_timesteps = (
            missing_cloud_mask_timesteps + get_missing_datetimes_from_list_of_files(cloud_masks)
        )

    # Convert filenames to datetimes, remove those datetimes from ones to download
    # Return list of all datetime range tuples to download files
    available_dates = eumetsat.identify_available_datasets(
        start_date, end_date, product_id=product_id
    )

    pass


def eumetsat_native_filename_to_datetime(filename: str):
    """Takes a file from the EUMETSAT API and returns
    the date and time part of the filename"""

    p = re.compile("^MSG[23]-SEVI-MSG15-0100-NA-(\d*)\.")
    title_match = p.match(filename)
    date_str = title_match.group(1)
    return datetime.strptime(date_str, "%Y%m%d%H%M%S")


def eumetsat_cloud_name_to_datetime(filename: str):
    date_str = filename.split("0100-0100-")[-1].split(".")[0]
    return datetime.strptime(date_str, "%Y%m%d%H%M%S")


def get_missing_datetimes_from_list_of_files(
    filenames: List[str],
) -> List[Tuple[datetime, datetime]]:
    """
    Get a list of all datetimes not covered by the set of images
    Args:
        filenames: Filenames of the EUMETSAT Native files or Cloud Masks

    Returns:
        List of datetime ranges that are missing from the filename range
    """
    # Sort in order from earliest to latest
    filenames = sorted(filenames)
    is_rss = ".nat" in filenames[0]  # Which type of file it is
    func = eumetsat_native_filename_to_datetime if is_rss else eumetsat_cloud_name_to_datetime
    current_time = func(filenames[0])
    # Start from first one and go through, adding date range between each one, as long as difference is
    # greater than or equal to 5min
    missing_date_ranges = []
    five_minutes = timedelta(minutes=5)
    for i in range(len(filenames)):
        next_time = func(filenames[i])
        time_difference = current_time - next_time
        if time_difference > five_minutes:
            # Add breaks to list, only want the ones between, so add/subtract 5minutes
            # In the case its missing only a single timestep, start and end would be the same time
            missing_date_ranges.append((current_time + five_minutes, next_time - five_minutes))
        current_time = next_time

    return missing_date_ranges


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

start = datetime(2018, 1, 1, tzinfo=pytz.utc)
end = datetime(2021, 1, 1, tzinfo=pytz.utc)
gcp_path = "gs://solar-pv-nowcasting-data/PV/GSP/v1"

config = {"start": start, "end": end, "gcp_path": gcp_path}
dm = eumetsat.DownloadManager(user_key, user_secret, data_dir, metadata_db_fp, debug_fp)

for day in range(0, 32):
    for month in range(0, 13):
        for year in range(2010, 2022):
            # Download 1 day at a time
            dm.download_date_range(
                f"{year}-{month}-{day} 00:00",
                f"{year}-{month}-{day} 23:59",
                product_id="EO:EUM:DAT:MSG:RSS-CLM",
            )
            dm.download_date_range(
                f"{year}-{month}-{day} 00:00",
                f"{year}-{month}-{day} 23:59",
                product_id="EO:EUM:DAT:MSG:MSG15-RSS",
            )

# format local temp folder
LOCAL_TEMP_PATH = Path("~/temp/").expanduser()
delete_all_files_in_temp_path(path=LOCAL_TEMP_PATH)

# upload to gcp
gcp_upload_and_delete_local_files(dst_path=gcp_path, local_path=LOCAL_TEMP_PATH)

if __name__ == "__main__":
    download_eumetsat_data()
