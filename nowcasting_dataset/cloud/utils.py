import logging
from pathlib import Path

from nowcasting_dataset.cloud.aws import aws_upload_and_delete_local_files
from nowcasting_dataset.cloud.gcp import gcp_upload_and_delete_local_files

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
