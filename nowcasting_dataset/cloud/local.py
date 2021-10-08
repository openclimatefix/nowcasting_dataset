""" Functions for local files """
import logging
import fsspec
from pathlib import Path
from typing import Union, List

_LOG = logging.getLogger(__name__)


def delete_all_files_in_temp_path(path: Union[Path, str]):
    """
    Delete all the files in a temporary path
    """

    filesystem = fsspec.open(path).fs
    filenames = get_all_filenames_in_path(path=path)

    _LOG.info(f"Deleting {len(filenames)} files from {path}.")

    for file in filenames:
        filesystem.rm(file, recursive=True)


def check_path_exists(path: Union[str, Path]):
    """Raises a RuntimeError if `path` does not exist in the local filesystem."""

    filesystem = fsspec.open(path).fs
    if not filesystem.exists(path):
        raise RuntimeError(f"{path} does not exist!")


def rename_file(remote_file: str, new_filename: str):
    """
    Rename file within one filesystem

    Args:
        remote_file: The current file name
        new_filename: What the file should be renamed too

    """
    filesystem = fsspec.open(remote_file).fs
    filesystem.mv(remote_file, new_filename)


def get_all_filenames_in_path(path: Union[str, Path]) -> List[str]:
    """
    Get all the files names from one folder in gcp

    Args:
        remote_path: the path that we should look in

    Returns: a list of files names represented as strings.
    """
    filesystem = fsspec.open(path).fs
    return filesystem.ls(path)


def download_to_local(remote_filename: str, local_filename: str):
    """
    Download file from gcs.

    Args:
        remote_filename: the file name, should start with gs:// or s3://
        local_filename: the local filename
        gcs: gcsfs.GCSFileSystem connection, means a new one doesnt have to be made everytime.
    """
    _LOG.debug(f"Downloading from GCP {remote_filename} to {local_filename}")

    filesystem = fsspec.open(remote_filename).fs
    filesystem.get(remote_filename, local_filename)


def upload_one_file(
    remote_filename: str,
    local_filename: str,
):
    """
    Upload one file to s3

    Args:
        remote_filename: the aws key name
        local_filename: the local file name

    """
    filesystem = fsspec.open(remote_filename).fs
    filesystem.put(local_filename, remote_filename)
