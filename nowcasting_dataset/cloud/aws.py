import logging
from pathlib import Path
import os

import boto3

from nowcasting_dataset.cloud.local import delete_all_files_and_folder_in_temp_path

_LOG = logging.getLogger("nowcasting_dataset")


def aws_upload_and_delete_local_files(aws_path: str, local_path: Path, bucket: str = "solar-pv-nowcasting-data"):
    """
    1. Upload the files in a local path, to a path in aws
    2. Delete files in that local path
    @param aws_path: the folder in the aws bucket that files will be saved too
    @param local_path: the local path where fiels will be copied from
    @param bucket: the aws bucket that files are saved too
    @return:
    """

    _LOG.info("Uploading to AWS!")

    # create s3 resource
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket)

    # upload files to s3
    for subdir, dirs, files in os.walk(local_path):
        for file in files:

            # get full local path of file
            full_path = os.path.join(subdir, file)

            # get key (or aws file name) by removing the original folder name from the full path and
            # join with AWS path name
            sub_dirs_and_filename = full_path.replace(f"{str(local_path)}/", "")
            key = os.path.join(Path(aws_path), sub_dirs_and_filename)

            _LOG.debug(f"uploading {full_path} to {key} in bucket {bucket}")
            with open(full_path, "rb") as data:
                bucket.put_object(Key=key, Body=data)

    # delete files in local path
    delete_all_files_and_folder_in_temp_path(local_path)
