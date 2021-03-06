""" copy a folder from local to gcp """
import logging
from concurrent import futures

from nowcasting_dataset.filesystem.utils import get_all_filenames_in_path, upload_one_file

logging.basicConfig()
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.DEBUG)

sets = ["train", "validation", "test"]
data_sources = ["gsp", "metadata", "nwp", "pv", "satellite", "sun", "topographic"]

GCP_PATH = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v10"
LOCAL_PATH = (
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
    "prepared_ML_training_data/v10"
)

all_filenames = {}
for dset in sets:
    for data_source in data_sources:
        dir = f"{LOCAL_PATH}/{dset}/{data_source}"
        gsp_dir = f"{GCP_PATH}/{dset}/{data_source}"
        files = get_all_filenames_in_path(dir)
        files = sorted(files)
        # get files already in gsp
        try:
            gsp_files_already = get_all_filenames_in_path(gsp_dir)
        except Exception:
            gsp_files_already = []
        # only get .nc files
        filenames = [file for file in files if ".nc" in file]
        gsp_files_already = [file for file in gsp_files_already if ".nc" in file]
        print(f"Already {len(gsp_files_already)} in gsp folder already: {gsp_dir}")

        # remove file if already in gsp
        filenames = [
            file
            for file in filenames
            if f'{gsp_dir.replace("gs://","")}/{file.split("/")[-1]}' not in gsp_files_already
        ]
        print(f"There are {len(filenames)} to upload")

        files_dict = {file: f'{gsp_dir}/{file.split("/")[-1]}' for file in filenames}
        if len(filenames) > 0:
            all_filenames = {**all_filenames, **files_dict}


def one_file(local_file, gsp_file):
    """Copy one file from local to gsp"""
    # can use this index, only to copy files after a certain number
    file_index = int(local_file.split(".")[0][-6:])
    if file_index > -1:
        print(gsp_file)
        upload_one_file(remote_filename=gsp_file, local_filename=local_file, overwrite=False)


# test to see if it works
one_file(list(all_filenames.keys())[0], all_filenames[list(all_filenames.keys())[0]])

# loop over files
with futures.ThreadPoolExecutor(max_workers=2) as executor:
    # Submit tasks to the executor.
    future_examples_per_source = []
    for k, v in all_filenames.items():
        task = executor.submit(one_file, local_file=k, gsp_file=v)
        future_examples_per_source.append(task)
