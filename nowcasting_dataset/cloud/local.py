""" Functions for local files """
import glob
import logging
import os
import shutil
import fsspec
from pathlib import Path
from typing import Union

_LOG = logging.getLogger(__name__)


def delete_all_files_in_temp_path(path: Path):
    """
    Delete all the files in a temporary path
    """

    filesystem = fsspec.open(path).fs
    filenames = filesystem.ls(path)

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
