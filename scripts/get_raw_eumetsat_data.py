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
import yaml
from satip import eumetsat
import pandas as pd
import os
import math
from satpy import Scene

from typing import Optional, List, Union, Tuple

import re
from datetime import datetime, timedelta

from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from pathlib import Path
from nowcasting_dataset.cloud.local import delete_all_files_in_temp_path
from nowcasting_dataset.cloud.gcp import gcp_upload_and_delete_local_files
import logging
import click

# SatPy gives a UserWarning about Proj, which isn't actually used here, so can be
# safely ignored
import warnings

warnings.simplefilter("ignore", UserWarning)


NATIVE_FILESIZE_MB = 102.210123
RSS_ID = "EO:EUM:DAT:MSG:MSG15-RSS"
CLOUD_ID = "EO:EUM:DAT:MSG:RSS-CLM"

format_dt_str = lambda dt: pd.to_datetime(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_date(ctx, param, value):
    try:
        return format_dt_str(value)
    except ValueError:
        raise click.BadParameter("Date must be in format accepted by pd.to_datetime()")


@click.command()
@click.option(
    "--download_directory",
    "--dir",
    default="/run/media/jacob/data/",
    help="Where to download the data to. Also where the script searches for previously downloaded data.",
)
@click.option(
    "--start_date",
    "--start",
    default="2021-01-01",
    prompt="Starting date to download data, in format accepted by pd.to_datetime()",
    callback=validate_date,
)
@click.option(
    "--end_date",
    "--end",
    default="2021-01-10",
    prompt="Ending date to download data, in format accepted by pd.to_datetime()",
    callback=validate_date,
)
@click.option(
    "--backfill",
    "-b",
    default=False,
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
    default="auth.yaml",
    help="The auth file containing the user key and access key for EUMETSAT access",
)
@click.option(
    "--bandwidth_limit",
    "--bw_limit",
    default=0.0,
    prompt="Bandwidth limit, in MB/sec, currently ignored",
    type=float,
)
def download_eumetsat_data(
    download_directory,
    start_date: str,
    end_date: str,
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
        download_directory: Directory to download the files and store them
        start_date: Start date, in a format accepted by pandas to_datetime()
        end_date: End date, in a format accepted by pandas to_datetime()
        backfill: Whether to backfill between the dates available on disk or not
        bandwidth_limit: Bandwidth limit, currently unused
        user_key: User key for the EUMETSAT API
        user_secret: User secret for the EUMETSAT API
        auth_filename: Path to a file containing the user_secret and user_key

    """
    # Get authentication
    if auth_filename is not None:
        user_key, user_secret = load_key_secret(auth_filename)

    # Download the data
    dm = eumetsat.DownloadManager(user_key, user_secret, download_directory, download_directory)

    for product_id in [
        RSS_ID,
        CLOUD_ID,
    ]:
        times_to_use = determine_datetimes_to_download_files(
            download_directory, start_date, end_date, product_id=product_id
        )
        for start_time, end_time in times_to_use:
            print(format_dt_str(start_time))
            print(format_dt_str(end_time))
            # pass
            dm.download_date_range(
                format_dt_str(start_time),
                format_dt_str(end_time),
                product_id=product_id,
            )
        # Sanity check, able to open/right size and move to correct directory
        sanity_check_files_and_move_to_directory(
            directory=download_directory, product_id=product_id
        )
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


def sanity_check_files_and_move_to_directory(directory, product_id):
    """
    Runs a sanity check for all the files of a given product_id in the directory
    Deletes incomplete files, moves checked files to final location

    This does a sanity check by:
        Checking the filesize of RSS images
        Attempting to open them in SatPy

    While loading the file should be somewhat slow, it ensures the file can be opened

    Args:
        directory: Directory where the native files were downloaded
        product_id: The product ID of the files to check

    Returns:
        The number of incomplete files deleted
    """
    pattern = "*.nat" if product_id == RSS_ID else "*.grb"
    fs = fsspec.filesystem("file")  # TODO Update for other ones?
    new_files = fs.glob(os.path.join(directory, pattern))

    date_func = (
        eumetsat_native_filename_to_datetime
        if product_id == RSS_ID
        else eumetsat_cloud_name_to_datetime
    )
    # Check all the right size
    satpy_reader = "seviri_l1b_native" if product_id == RSS_ID else "seviri_l2_grib"

    if product_id == RSS_ID:
        for f in new_files:
            file_size = eumetsat.get_filesize_megabytes(f)
            if not math.isclose(file_size, NATIVE_FILESIZE_MB, abs_tol=1):
                print("Wrong Size")
            scene = Scene(filenames=[f], reader=satpy_reader)
            scene.load(SAT_VARIABLE_NAMES)
            # Now that the file has been checked and can be open, move it to the final directory
            base_name = get_basename(f)
            file_date = date_func(base_name)
            if not fs.exists(os.path.join(directory, file_date.strftime(format="%Y/%m/%d"))):
                fs.mkdir(os.path.join(directory, file_date.strftime(format="%Y/%m/%d")))
            fs.move(f, os.path.join(directory, file_date.strftime(format="%Y/%m/%d"), base_name))
    else:
        for f in new_files:
            # Fails to open for this
            scene = Scene(filenames=[f], reader=satpy_reader)
            scene.load("cloud_mask")
            base_name = get_basename(f)
            file_date = date_func(base_name)
            fs.move(f, os.path.join(directory, file_date.strftime(format="%Y/%m/%d"), base_name))

    return


def determine_datetimes_to_download_files(
    directory,
    start_date,
    end_date,
    product_id,
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

    pattern = "*.nat" if product_id == RSS_ID else "*.grb"
    # Get all days from start_date to end_date
    day_split = pd.date_range(start_date, end_date, freq="D")

    # Go through files and get all examples in each
    fs = fsspec.filesystem("file")  # TODO Update for other ones?

    # Go through each directory, and for each day, list any missing data
    missing_rss_timesteps = []
    for day in day_split:
        day_string = day.strftime(format="%Y/%m/%d")
        rss_images = fs.glob(os.path.join(directory, day_string, pattern))
        if len(rss_images) > 0:
            missing_rss_timesteps = (
                missing_rss_timesteps + get_missing_datetimes_from_list_of_files(rss_images)
            )
        else:
            # No files, so whole day should be included
            # Each one is at the start of the day, this then needs 1 minute before for the RSS image
            # End 2 minutes before the end of the day, as that image would be for midnight, the next day
            missing_day = (
                day - timedelta(minutes=1),
                day + timedelta(hours=23, minutes=58),
            )
            missing_rss_timesteps.append(missing_day)

    return missing_rss_timesteps


def eumetsat_native_filename_to_datetime(filename: str):
    """Takes a file from the EUMETSAT API and returns
    the date and time part of the filename"""

    p = re.compile("^MSG[23]-SEVI-MSG15-0100-NA-(\d*)\.")
    title_match = p.match(filename)
    date_str = title_match.group(1)
    return datetime.strptime(date_str, "%Y%m%d%H%M%S").replace(second=0)


def eumetsat_cloud_name_to_datetime(filename: str):
    date_str = filename.split("0100-0100-")[-1].split(".")[0]
    return datetime.strptime(date_str, "%Y%m%d%H%M%S").replace(second=0)


def get_basename(filename: str):
    return filename.split("/")[-1]


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
    current_time = func(get_basename(filenames[0]))
    # Start from first one and go through, adding date range between each one, as long as difference is
    # greater than or equal to 5min
    missing_date_ranges = []
    five_minutes = timedelta(minutes=5)
    for i in range(len(filenames)):
        next_time = func(get_basename(filenames[i]))
        time_difference = next_time - current_time
        if time_difference > five_minutes:
            # Add breaks to list, only want the ones between, so add 5 minutes to the start
            # In the case its missing only a single timestep, start and end would be the same time
            missing_date_ranges.append((current_time + five_minutes, next_time))
        current_time = next_time
    return missing_date_ranges


if __name__ == "__main__":
    download_eumetsat_data()
