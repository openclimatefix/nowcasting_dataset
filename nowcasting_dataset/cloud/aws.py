import logging
from pathlib import Path
import os
from typing import List
import boto3

from nowcasting_dataset.cloud.local import delete_all_files_and_folder_in_temp_path

_LOG = logging.getLogger(__name__)


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


def aws_download_to_local(
    remote_filename: str,
    local_filename: str,
    s3_resource: boto3.resource = None,
    bucket: str = "solar-pv-nowcasting-data",
):
    """
    Download file from gcs
    @param remote_filename: the gcs file name, should start with gs://
    @param local_filename:
    @param s3_resource: s3 resource, means a new one doesnt have to be made everytime.
    @param bucket: The s3 bucket name, from which to load the file from.
    """

    _LOG.debug(f"Downloading {remote_filename} from AWS to {local_filename}")

    if s3_resource is None:
        s3_resource = boto3.resource("s3")

    # see if file exists in s3
    s3_resource.Object(bucket, remote_filename).load()

    # download file
    s3_resource.Bucket(bucket).download_file(remote_filename, local_filename)


def upload_one_file(
    remote_filename: str,
    local_filename: str,
    bucket: str = "solar-pv-nowcasting-data",
):
    """
    Upload one file to s3
    @param remote_filename: the aws key name
    @param local_filename: the local file name
    @param bucket: the s3 bucket
    """

    # create s3 resource
    s3 = boto3.client("s3")

    _LOG.debug(f"uploading {local_filename} to {remote_filename} in bucket {bucket}")
    with open(local_filename, "rb") as data:
        s3.upload_fileobj(data, bucket, remote_filename)


def get_all_filenames_in_path_aws(remote_path, bucket: str = "solar-pv-nowcasting-data") -> List[str]:
    """
    Get all the files names from one folder in gcp
    @param remote_path: the path that we should look in
    @return: a list of strings, of files names
    """
    # get client
    s3 = boto3.client("s3")

    # get all objects, need to loop round as limit number are loaded each time
    is_truncated = True
    filenames = []
    inputs = {"Bucket": bucket, "Prefix": remote_path, "MaxKeys": 1000}
    while is_truncated:
        all_objects = s3.list_objects_v2(**inputs)
        is_truncated = all_objects["IsTruncated"]

        if "NextContinuationToken" in all_objects.keys():
            inputs["ContinuationToken"] = all_objects["NextContinuationToken"]

        filenames = filenames + [obj["Key"] for obj in all_objects["Contents"]]

    # remove remote path from file name
    filenames = [file.replace(remote_path, "").replace("/", "") for file in filenames]

    return filenames
