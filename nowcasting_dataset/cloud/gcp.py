import logging
from pathlib import Path
from typing import List

import gcsfs

from nowcasting_dataset.cloud.local import delete_all_files_in_temp_path

_LOG = logging.getLogger(__name__)


def check_path_exists(path: Path):
    """
    Check that the path exists in GCS
    @param path: the path in GCS that is checked
    """
    gcs = gcsfs.GCSFileSystem()
    if not gcs.exists(path):
        raise RuntimeError(f"{path} does not exist!")


def gcp_upload_and_delete_local_files(dst_path: str, local_path: Path):
    """
    Upload the files in a local path, to a path in gcs
    """
    _LOG.info("Uploading to GCS!")
    gcs = gcsfs.GCSFileSystem()
    gcs.put(str(local_path), dst_path, recursive=True)
    delete_all_files_in_temp_path(local_path)


def gcp_download_to_local(remote_filename: str, local_filename: str, gcs: gcsfs.GCSFileSystem = None):
    """
    Download file from gcs
    @param remote_filename: the gcs file name, should start with gs://
    @param local_filename:
    @param gcs: gcsfs.GCSFileSystem connection, means a new one doesnt have to be made everytime.
    """

    _LOG.debug(f'Downloading from GCP {remote_filename} to {local_filename}')

    if gcs is None:
        gcs = gcsfs.GCSFileSystem()
    gcs.get(remote_filename, local_filename)


def get_all_filenames_in_path(remote_path) -> List[str]:
    """
    Get all the files names from one folder in gcp
    @param remote_path: the path that we should look in
    @return: a list of strings, of files names
    """
    gcs = gcsfs.GCSFileSystem()

    return gcs.ls(remote_path)
