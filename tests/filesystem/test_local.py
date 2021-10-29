# noqa: D100
import os
import tempfile
from pathlib import Path

from nowcasting_dataset.filesystem.utils import (
    check_path_exists,
    delete_all_files_in_temp_path,
    download_to_local,
    get_all_filenames_in_path,
    makedirs,
    upload_one_file,
)


def test_check_file_exists():  # noqa: D103

    file1 = "test_file1.txt"
    file2 = "test_dir/test_file2.txt"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        path_and_filename_1 = os.path.join(local_path, file1)
        with open(path_and_filename_1, "w"):
            pass

        # add fake file to dir
        os.mkdir(f"{tmpdirname}/test_dir")
        _ = os.path.join(local_path, file2)
        with open(os.path.join(local_path, file2), "w"):
            pass

        # run function
        check_path_exists(path=f"{tmpdirname}/test_dir")


def test_makedirs():  # noqa: D103

    folder_1 = "test_dir_1"
    folder_2 = "test_dir_2"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        folder_1 = os.path.join(local_path, folder_1)
        folder_2 = os.path.join(local_path, folder_2)

        # use the make folder function
        makedirs(folder_1)
        check_path_exists(path=folder_1)

        # make a folder
        os.mkdir(folder_2)

        # make a folder that already exists
        check_path_exists(path=folder_2)


def test_delete_local_files():  # noqa: D103

    file1 = "test_file1.txt"
    folder1 = "test_dir"
    file2 = "test_dir/test_file2.txt"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        path_and_filename_1 = os.path.join(local_path, file1)
        with open(path_and_filename_1, "w"):
            pass

        # add fake file to dir
        path_and_folder_1 = os.path.join(local_path, folder1)
        os.mkdir(path_and_folder_1)
        path_and_filename_2 = os.path.join(local_path, file2)
        with open(os.path.join(local_path, file2), "w"):
            pass

        # run function
        delete_all_files_in_temp_path(path=local_path)

        # check the object are not there
        assert not os.path.exists(path_and_filename_1)
        assert not os.path.exists(path_and_filename_2)
        assert os.path.exists(path_and_folder_1)


def test_delete_local_files_and_folder():  # noqa: D103

    file1 = "test_file1.txt"
    folder1 = "test_dir"
    file2 = "test_dir/test_file2.txt"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        path_and_filename_1 = os.path.join(local_path, file1)
        with open(path_and_filename_1, "w"):
            pass

        # add fake file to dir
        path_and_folder_1 = os.path.join(local_path, folder1)
        os.mkdir(path_and_folder_1)
        path_and_filename_2 = os.path.join(local_path, file2)
        with open(os.path.join(local_path, file2), "w"):
            pass

        # run function
        delete_all_files_in_temp_path(path=local_path, delete_dirs=True)

        # check the object are not there
        assert not os.path.exists(path_and_filename_1)
        assert not os.path.exists(path_and_filename_2)
        assert not os.path.exists(path_and_folder_1)


def test_download():  # noqa: D103

    file1 = "test_file1.txt"
    file2 = "test_dir/test_file2.txt"
    file3 = "test_file3.txt"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        path_and_filename_1 = os.path.join(local_path, file1)
        with open(path_and_filename_1, "w"):
            pass

        # add fake file to dir
        os.mkdir(f"{tmpdirname}/test_dir")
        _ = os.path.join(local_path, file2)
        with open(os.path.join(local_path, file2), "w"):
            pass

        # run function
        path_and_filename_3 = os.path.join(local_path, file3)
        download_to_local(remote_filename=path_and_filename_1, local_filename=path_and_filename_3)

        # check the object are not there
        filenames = get_all_filenames_in_path(local_path)
        assert len(filenames) == 3


def test_upload():  # noqa: D103

    file1 = "test_file1.txt"
    file2 = "test_dir/test_file2.txt"
    file3 = "test_file3.txt"

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = Path(tmpdirname)

        # add fake file to dir
        path_and_filename_1 = os.path.join(local_path, file1)
        with open(path_and_filename_1, "w"):
            pass

        # add fake file to dir
        os.mkdir(f"{tmpdirname}/test_dir")
        _ = os.path.join(local_path, file2)
        with open(os.path.join(local_path, file2), "w"):
            pass

        # run function
        path_and_filename_3 = os.path.join(local_path, file3)
        upload_one_file(remote_filename=path_and_filename_3, local_filename=path_and_filename_1)

        # check the object are not there
        filenames = get_all_filenames_in_path(local_path)
        assert len(filenames) == 3
