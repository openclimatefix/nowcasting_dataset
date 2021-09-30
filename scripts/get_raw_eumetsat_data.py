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
import os
import requests
import urllib

from typing import Optional, List, Union, Tuple

import re
from datetime import datetime, timedelta

from pathlib import Path
from nowcasting_dataset.cloud.local import delete_all_files_in_temp_path
from nowcasting_dataset.cloud.gcp import gcp_upload_and_delete_local_files
import logging
import click


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
    default="auth.yaml",
    help="The auth file containing the user key and access key for EUMETSAT access",
)
@click.option("--bandwidth_limit", "--bw_limit", prompt="Bandwidth limit, in MB/sec", type=float)
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
    print(user_key)
    print(user_secret)

    # API endpoint
    apis_endpoint = "https://api.eumetsat.int/"

    # Application Token
    token_url = apis_endpoint + "token"

    response = requests.post(
        token_url,
        auth=requests.auth.HTTPBasicAuth(user_key, user_secret),
        data={"grant_type": "client_credentials"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200, "API Request Failed: {}\n{}".format(
        response.status_code, response.content
    )

    access_token = response.json()["access_token"]
    print("Your access token is: ", access_token)

    # API base endpoint
    apis_endpoint = "http://api.eumetsat.int"

    # Searching endpoint
    service_search = apis_endpoint + "/data/search-products/os"

    # Downloading endpoint
    service_download = apis_endpoint + "/data/download"

    # Select our collection
    selected_collection_id = "EO:EUM:DAT:MSG:MSG15-RSS"

    product_id = "MSG3-SEVI-MSG15-0100-NA-20201231235915.622000000Z-NA"
    download_url = service_download + "/collections/{}/products/{}".format(
        urllib.parse.quote(selected_collection_id), urllib.parse.quote(product_id)
    )
    print(
        "Download URL (via product-ID) ({}): ".format(str(1)),
        download_url + "?access_token=" + access_token,
    )
    # Download the data
    dm = eumetsat.DownloadManager(
        user_key, user_secret, download_directory, download_directory, download_directory
    )

    for product_id in [RSS_ID, CLOUD_ID]:
        times_to_use = determine_datetimes_to_download_files(
            download_directory, start_date, end_date, product_id=product_id
        )
        for start_time, end_time in times_to_use:
            print(format_dt_str(start_time))
            print(format_dt_str(end_time))
            dm.download_date_range(
                format_dt_str(start_time),
                format_dt_str(end_time),
                product_id=product_id,
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
            missing_day = (
                day - timedelta(minutes=1),
                day + timedelta(hours=23, minutes=59, seconds=59),
            )
            missing_rss_timesteps.append(missing_day)

    return missing_rss_timesteps


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


if __name__ == "__main__":
    download_eumetsat_data()
