## copy a folder from gcp to aws
from pathlib import Path
from nowcasting_dataset.cloud.gcp import get_all_filenames_in_path
from nowcasting_dataset.cloud.utils import gcp_to_aws
import gcsfs
import os
import logging
from concurrent import futures

logging.basicConfig()
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.DEBUG)


GCP_PATH = 'gs://solar-pv-nowcasting-data/prepared_ML_training_data/v4/train/'
AWS_PATH = "prepared_ML_training_data/v4/train"
AWS_BUCKET = "solar-pv-nowcasting-data"

# get all gcp filenames
filenames = get_all_filenames_in_path(remote_path=GCP_PATH)

# only get .nc files
filenames = [file for file in filenames if '.nc' in file]

# sort list by (/.../.../yyyyyy_xxxx.nc) x
filenames.sort(key=lambda x: int(x.split('/')[-1][7:].split('.')[0]))

# make aws filenames
aws_files = {file: file.split('/')[-1] for file in filenames}

# get gcs system
gcs = gcsfs.GCSFileSystem()


def one_file(filename):
    aws_filename = os.path.join(AWS_PATH, aws_files[filename])

    # can use this index, only to copy files after a certain number
    file_index = int(aws_files[filename][7:].split('.')[0])

    if file_index > 18000:
        print(filename)
        gcp_to_aws(gcp_filename=filename, aws_filename=aws_filename, aws_bucket=AWS_BUCKET, gcs=gcs)


# loop over files
with futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Submit tasks to the executor.
    future_examples_per_source = []
    for filename in filenames:
        task = executor.submit(
            one_file,
            filename=filename)
        future_examples_per_source.append(task)

