import logging
from pathlib import Path
import gcsfs
import tempfile

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
        upload_one_file(remote_filename=aws_filename, bucket=aws_bucket, local_filename=local_filename)
