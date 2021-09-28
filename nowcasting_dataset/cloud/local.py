import glob
import os
import shutil
from typing import Union

import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)


def delete_all_files_and_folder_in_temp_path(path: str):
    """
    Delete all the files and folders in a temporary path
    """

    _LOG.info(f"Deleting files and folder from {path} .")

    for files in os.listdir(path):
        file_or_dir = os.path.join(path, files)
        try:
            shutil.rmtree(file_or_dir)
        except OSError:
            os.remove(file_or_dir)


def delete_all_files_in_temp_path(path: Path):
    """
    Delete all the files in a temporary path
    """
    files = glob.glob(str(path / "*.*"))
    _LOG.info(f"Deleting {len(files)} files from {path}.")
    for f in files:
        os.remove(f)


def check_path_exists(path: Union[str, Path]):
    """Raises a RuntimeError if `path` does not exist in the local filesystem."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} does not exist!")
