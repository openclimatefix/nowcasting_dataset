""" General utils functions """
import logging
from pathlib import Path
from typing import List, Union

import fsspec
import numpy as np
from pathy import Pathy

_LOG = logging.getLogger(__name__)


def upload_and_delete_local_files(dst_path: Union[str, Path], local_path: Union[str, Path]):
    """
    Upload an entire folder and delete local files to either AWS or GCP
    """
    _LOG.info("Uploading!")
    filesystem = get_filesystem(dst_path)
    filesystem.copy(str(local_path), str(dst_path), recursive=True)
    delete_all_files_in_temp_path(local_path)


def get_filesystem(path: Union[str, Path]) -> fsspec.AbstractFileSystem:
    r"""Get the fsspect FileSystem from a path.

    For example, if `path` starts with `gs:\\` then return a fsspec.GCSFileSystem.

    It is safe for `path` to include wildcards in the final filename.
    """
    path = Pathy(path)
    return fsspec.open(path.parent).fs


def get_maximum_batch_id(path: Pathy) -> int:
    """
    Get the last batch ID. Works with GCS, AWS, and local.

    Args:
        path: The path folder to look in.
              Begin with 'gs://' for GCS. Begin with 's3://' for AWS S3.
              Supports wildcards *, **, ?, and [..].

    Returns: The maximum batch id of data in `path`.  If `path` exists but contains no files
        then returns -1.

    Raises FileNotFoundError if `path` does not exist.
    """
    _LOG.debug(f"Looking for maximum batch id in {path}")

    filesystem = get_filesystem(path)
    if not filesystem.exists(path.parent):
        msg = f"{path.parent} does not exist"
        _LOG.warning(msg)
        raise FileNotFoundError(msg)

    if "*" in str(path):
        filenames = filesystem.glob(path)
    else:
        filenames = get_all_filenames_in_path(path)

    # if there is no files, return 0
    if len(filenames) == 0:
        _LOG.debug(f"Did not find any files in {path}")
        return -1

    # Now that filenames have leading zeros (like 000001.nc), we can use lexographical sorting
    # to find the last filename, instead of having to convert all filenames to int.
    filenames = np.sort(filenames)
    last_filename = filenames[-1]
    last_filename = Pathy(last_filename)
    last_filename_stem = last_filename.stem
    maximum_batch_id = int(last_filename_stem)
    _LOG.debug(f"Found maximum of batch it of {maximum_batch_id} in {path}")

    return maximum_batch_id


def delete_all_files_in_temp_path(path: Union[Path, str], delete_dirs: bool = False):
    """
    Delete all the files in a temporary path. Option to delete the folders or not
    """
    filesystem = get_filesystem(path)
    filenames = get_all_filenames_in_path(path=path)

    _LOG.info(f"Deleting {len(filenames)} files from {path}.")

    if delete_dirs:
        for filename in filenames:
            filesystem.rm(str(filename), recursive=True)
    else:
        # loop over folder structure, but only delete files
        for root, dirs, files in filesystem.walk(path):

            for f in files:
                filesystem.rm(f"{root}/{f}")


def check_path_exists(path: Union[str, Path]):
    """Raises a FileNotFoundError if `path` does not exist.

    `path` can include wildcards.
    """
    if not path:
        raise FileNotFoundError("Not a valid path!")
    filesystem = get_filesystem(path)
    if not filesystem.exists(path):
        # Now try using `glob`.  Maybe `path` includes a wildcard?
        # Try `exists` before `glob` because `glob` might be slower.
        files = filesystem.glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f"{path} does not exist!")


def rename_file(remote_file: str, new_filename: str):
    """
    Rename file within one filesystem

    Args:
        remote_file: The current file name
        new_filename: What the file should be renamed too

    """
    filesystem = get_filesystem(remote_file)
    filesystem.mv(remote_file, new_filename)


def get_all_filenames_in_path(path: Union[str, Path]) -> List[str]:
    """
    Get all the files names from one folder.

    Args:
        path: The path that we should look in.

    Returns: A list of filenames represented as strings.
    """
    filesystem = get_filesystem(path)
    return filesystem.ls(path)


def download_to_local(remote_filename: str, local_filename: str):
    """
    Download file from gcs.

    Args:
        remote_filename: the file name, should start with gs:// or s3://
        local_filename: the local filename
    """
    _LOG.debug(f"Downloading from GCP {remote_filename} to {local_filename}")

    # Check the inputs are strings
    remote_filename = str(remote_filename)
    local_filename = str(local_filename)

    filesystem = get_filesystem(remote_filename)
    try:
        filesystem.get(remote_filename, local_filename)
    except FileNotFoundError:
        _LOG.error(f"Could not find {remote_filename}")
        raise FileNotFoundError(f"Could not find {remote_filename}")


def upload_one_file(remote_filename: str, local_filename: str, overwrite: bool = True):
    """
    Upload one file to aws or gcp

    Args:
        remote_filename: the aws/gcp key name
        local_filename: the local file name
        overwrite: overwrite file

    """
    filesystem = get_filesystem(remote_filename)
    if overwrite:
        filesystem.put(local_filename, remote_filename)
    elif ~filesystem.exists(remote_filename):
        filesystem.put(local_filename, remote_filename)


def makedirs(path: Union[str, Path], exist_ok: bool = True) -> None:
    """Recursively make directories

    Creates directory at path and any intervening required directories.

    Raises exception if, for instance, the path already exists but is a file.

    Args:
        path: The path to create.
        exist_ok: If False then raise an exception if `path` already exists.
    """
    filesystem = get_filesystem(path)
    filesystem.makedirs(path, exist_ok=exist_ok)
