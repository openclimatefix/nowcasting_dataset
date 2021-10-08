import os
import tempfile
from pathlib import Path

from datetime import datetime
import pytest

from nowcasting_dataset.filesystem.utils import (
    upload_and_delete_local_files,
    get_all_filenames_in_path,
    delete_all_files_in_temp_path,
    download_to_local,
    upload_one_file,
)


@pytest.mark.skip("CI does not have access to AWS ro GCP")
@pytest.mark.parametrize("prefix", ["s3", "gs"])
def test_aws_upload_and_delete_local_files(prefix):

    file1 = "test_file1.txt"
    file2 = "test_dir/test_file2.txt"

    now = datetime.now().isoformat()
    dst_path = f"{prefix}://solar-pv-nowcasting-data/temp_dir_for_unit_tests/{now}"

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
        upload_and_delete_local_files(dst_path=dst_path, local_path=local_path)

        # check the object are there
        filenames = get_all_filenames_in_path(dst_path)
        assert len(filenames) == 2

        delete_all_files_in_temp_path(dst_path)


@pytest.mark.parametrize("prefix", ["s3", "gs"])
def test_upload_one_file(prefix):

    file1 = "test_file1.txt"
    now = datetime.now().isoformat()
    dst_path = f"{prefix}://solar-pv-nowcasting-data/temp_dir_for_unit_tests/{now}"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        local_filename = os.path.join(local_path, file1)
        with open(local_filename, "w"):
            pass

        # run function
        upload_one_file(remote_filename=dst_path, local_filename=local_filename)

        # check the object are there
        filenames = get_all_filenames_in_path(dst_path)
        assert len(filenames) == 1

        delete_all_files_in_temp_path(dst_path)


@pytest.mark.parametrize("prefix", ["s3", "gs"])
def test_download_file(prefix):

    file1 = "test_file1.txt"
    now = datetime.now().isoformat()
    dst_path = f"{prefix}://solar-pv-nowcasting-data/temp_dir_for_unit_tests/{now}"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        with open(os.path.join(local_path, file1), "w"):
            pass

        # run function
        upload_and_delete_local_files(dst_path=dst_path, local_path=local_path)

        # download object
        download_filename = os.path.join(local_path, "test_download_file1.txt")
        download_to_local(
            remote_filename=os.path.join(dst_path, file1),
            local_filename=download_filename,
        )

        # check the object are there
        os.path.isfile(download_filename)
