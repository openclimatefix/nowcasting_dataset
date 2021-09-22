import logging
from pathlib import Path
import gcsfs
import tempfile
import fsspec

from nowcasting_dataset.cloud.aws import aws_upload_and_delete_local_files, upload_one_file
from nowcasting_dataset.cloud.gcp import gcp_upload_and_delete_local_files, gcp_download_to_local

_LOG = logging.getLogger("nowcasting_dataset")


def upload_and_delete_local_files(dst_path: str, local_path: Path, cloud: str = "gcp"):
    """
    Upload and delete local files to either AWS or GCP
    """

    assert cloud in ["gcp", "aws"]

    if cloud == "gcp":
        gcp_upload_and_delete_local_files(dst_path=dst_path, local_path=local_path)
    else:
        aws_upload_and_delete_local_files(aws_path=dst_path, local_path=local_path)


def gcp_to_aws(gcp_filename: str, gcs: gcsfs.GCSFileSystem, aws_filename: str, aws_bucket: str):
    """
    Download a file from gcp and upload it to aws
    @param gcp_filename: the gcp file name
    @param gcs: the gcs file system (so it doesnt have to be made more than once)
    @param aws_filename: the aws filename and path
    @param aws_bucket: the asw bucket
    """

    # create temp file
    with tempfile.NamedTemporaryFile() as fp:
        local_filename = fp.name
        print(local_filename)

        # download from gcp
        gcp_download_to_local(remote_filename=gcp_filename, gcs=gcs, local_filename=local_filename)

        # upload to aws
        upload_one_file(
            remote_filename=aws_filename, bucket=aws_bucket, local_filename=local_filename
        )


def get_maximum_batch_id(path: str):
    """
    Get the last batch ID. Works with GCS, AWS, and local.

    Args:
        path: the path folder to look in.  Begin with 'gs://' for GCS.

    Returns: the maximum batch id of data in `path`.
    """
    _LOG.debug(f"Looking for maximum batch id in {path}")

    filesystem = fsspec.open(path).fs
    filenames = filesystem.ls(path)

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
