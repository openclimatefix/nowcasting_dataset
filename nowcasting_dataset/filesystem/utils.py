""" General utils functions """
import logging
from pathlib import Path
from typing import List, Union

import fsspec
from pathy import Pathy

_LOG = logging.getLogger("nowcasting_dataset")


def upload_and_delete_local_files(dst_path: str, local_path: Path):
    """
    Upload an entire folder and delete local files to either AWS or GCP
    """
    _LOG.info("Uploading!")
    filesystem = get_filesystem(dst_path)
    filesystem.put(str(local_path), dst_path, recursive=True)
    delete_all_files_in_temp_path(local_path)


def get_filesystem(path: Union[str, Path]) -> fsspec.AbstractFileSystem:
    r"""Get the fsspect FileSystem from a path.

    For example, if `path` starts with `gs:\\` then return a fsspec.GCSFileSystem.

    It is safe for `path` to include wildcards in the final filename.
    """
    path = Pathy(path)
    return fsspec.open(path.parent).fs


# TODO: Issue #308: Use leading zeros in batch filenames, then we can sort the filename strings
# and take the last one, instead of converting all filenames to ints!
def get_maximum_batch_id(path: str) -> int:
    """
    Get the last batch ID. Works with GCS, AWS, and local.

    Args:
        path: The path folder to look in.
              Begin with 'gs://' for GCS. Begin with 's3://' for AWS S3.
              Supports wildcards *, **, ?, and [..].

    Returns: The maximum batch id of data in `path`.

    Raises FileNotFoundError if `path` does not exist.
    """
    _LOG.debug(f"Looking for maximum batch id in {path}")

    filesystem = get_filesystem(path)
    if not filesystem.exists(path):
        msg = f"{path} does not exists"
        _LOG.warning(msg)
        raise FileNotFoundError(msg)

    filenames = get_all_filenames_in_path(path=path)

    # if there is no files, return 0
    if len(filenames) == 0:
        _LOG.debug(f"Did not find any files in {path}")
        return 0

    # just take the stem (the filename without the suffix and without the path)
    stems = [filename.stem for filename in filenames]

    # change to integer
    batch_indexes = [int(stem) for stem in stems if len(stem) > 0]

    # get the maximum batch id
    maximum_batch_id = max(batch_indexes)
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
        for file in filenames:
            filesystem.rm(file, recursive=True)
    else:
        # loop over folder structure, but only delete files
        for root, dirs, files in filesystem.walk(path):

            for f in files:
                filesystem.rm(f"{root}/{f}")


def check_path_exists(path: Union[str, Path]):
    """Raises a FileNotFoundError if `path` does not exist."""
    filesystem = get_filesystem(path)
    if not filesystem.exists(path):
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


def get_all_filenames_in_path(path: Union[str, Path]) -> List[Pathy]:
    """
    Get all the files names from one folder.

    Args:
        path: The path that we should look in.  Supports wildcards *, ?, and [..].

    Returns: A list of filenames represented as Pathy objects.
    """
    filesystem = get_filesystem(path)
    filename_strings = filesystem.glob(path)
    return [Pathy(filename) for filename in filename_strings]


def download_to_local(remote_filename: str, local_filename: str):
    """
    Download file from gcs.

    Args:
        remote_filename: the file name, should start with gs:// or s3://
        local_filename: the local filename
    """
    _LOG.debug(f"Downloading from GCP {remote_filename} to {local_filename}")

    filesystem = get_filesystem(remote_filename)
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
    filesystem = get_filesystem(remote_filename)
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
    filesystem.mkdir(path, exist_ok=exist_ok)
