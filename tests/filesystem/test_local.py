import os
import tempfile
from pathlib import Path
from nowcasting_dataset.filesystem.utils import delete_all_files_in_temp_path, check_path_exists


def test_check_file_exists():

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
        path_and_filename_2 = os.path.join(local_path, file2)
        with open(os.path.join(local_path, file2), "w"):
            pass

        # run function
        check_path_exists(path=f"{tmpdirname}/test_dir")


def test_delete_local_files():

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
        path_and_filename_2 = os.path.join(local_path, file2)
        with open(os.path.join(local_path, file2), "w"):
            pass

        # run function
        delete_all_files_in_temp_path(path=local_path)

        # check the object are not there
        assert not os.path.exists(path_and_filename_1)
        assert not os.path.exists(path_and_filename_2)
