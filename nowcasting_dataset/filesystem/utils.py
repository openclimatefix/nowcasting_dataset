""" General utils functions """
import logging
from pathlib import Path
from typing import Union, List

import fsspec

_LOG = logging.getLogger("nowcasting_dataset")


def upload_and_delete_local_files(dst_path: str, local_path: Path):
    """
    Upload an entire folder and delete local files to either AWS or GCP
    """
    _LOG.info("Uploading!")
    filesystem = fsspec.open(dst_path).fs
    filesystem.put(str(local_path), dst_path, recursive=True)
    delete_all_files_in_temp_path(local_path)


def get_maximum_batch_id(path: str):
    """
    Get the last batch ID. Works with GCS, AWS, and local.

    Args:
        path: the path folder to look in.  Begin with 'gs://' for GCS. Begin with 's3://' for AWS S3.

    Returns: the maximum batch id of data in `path`.
    """
    _LOG.debug(f"Looking for maximum batch id in {path}")

    filesystem = fsspec.open(path).fs
    if not filesystem.exists(path):
        _LOG.debug(f"{path} does not exists")
        return None

    filenames = get_all_filenames_in_path(path=path)

    # just take filename
    filenames = [filename.split("/")[-1] for filename in filenames]

    # remove suffix
    filenames = [filename.split(".")[0] for filename in filenames]

    # change to integer
    batch_indexes = [int(filename) for filename in filenames if len(filename) > 0]

    # if there is no files, return None
    if len(batch_indexes) == 0:
        _LOG.debug(f"Did not find any files in {path}")
        return None

    # get the maximum batch id
    maximum_batch_id = max(batch_indexes)
    _LOG.debug(f"Found maximum of batch it of {maximum_batch_id} in {path}")

    return maximum_batch_id


def delete_all_files_in_temp_path(path: Union[Path, str], delete_dirs: bool = False):
    """
    Delete all the files in a temporary path. Option to delete the folders or not
    """
    filesystem = fsspec.open(path).fs
    filenames = get_all_filenames_in_path(path=path)

    _LOG.info(f"Deleting {len(filenames)} files from {path}.")

    if delete_dirs:
        for file in filenames:
            filesystem.rm(file, recursive=True)
    else:
        # loop over folder structure, but only delete files
        for root, dirs, files in filesystem.walk(path):

            for f in files:
                filesystem.rm(f"{root}/{f}")


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
        path: the path that we should look in

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
    """
    _LOG.debug(f"Downloading from GCP {remote_filename} to {local_filename}")

    filesystem = fsspec.open(remote_filename).fs
    filesystem.get(remote_filename, local_filename)


def upload_one_file(
    remote_filename: str,
    local_filename: str,
):
    """
    Upload one file to aws or gcp

    Args:
        remote_filename: the aws/gcp key name
        local_filename: the local file name

    """
    filesystem = fsspec.open(remote_filename).fs
    filesystem.put(local_filename, remote_filename)


def make_folder(path: Union[str, Path]):
    """ Make folder """
    filesystem = fsspec.open(path).fs
    if not filesystem.exists(path):
        filesystem.mkdir(path)
