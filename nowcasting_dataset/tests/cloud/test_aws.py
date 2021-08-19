from moto import mock_s3
import boto3
from nowcasting_dataset.cloud.aws import aws_upload_and_delete_local_files
from pathlib import Path
import tempfile
import os


@mock_s3
def test_aws_upload_and_delete_local_files():

    bucket_name = "test_bucket"
    file1 = "test_file1.txt"
    file2 = "test_dir/test_file2.txt"

    s3_client = boto3.client("s3")
    s3_client.create_bucket(Bucket=bucket_name)

    aws_path = "test/dir/in/aws"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        with open(os.path.join(local_path, file1), "w"):
            pass

        # add fake file to dir
        os.mkdir(f"{tmpdirname}/test_dir")
        with open(os.path.join(local_path, file2), "w"):
            pass

        # run function
        aws_upload_and_delete_local_files(aws_path=aws_path, local_path=local_path, bucket=bucket_name)

        # check the object are there
        s3 = boto3.resource("s3")
        s3.Object(bucket_name, os.path.join(aws_path, file1)).load()
        s3.Object(bucket_name, os.path.join(aws_path, file2)).load()

        # check the local objects have been deleted
        for subdir, dirs, files in os.walk(local_path):
            for _ in files:
                assert 0, f"There is a file in {local_path}, but there shouldn't be"
            for _ in dirs:
                assert 0, f"There is a dir in {local_path}, but there shouldn't be"
