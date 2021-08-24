## copy a folder from gcp to aws
from pathlib import Path
from nowcasting_dataset.cloud.gcp import get_all_filenames_in_path
from nowcasting_dataset.cloud.utils import gcp_to_aws
import gcsfs
import os
import logging

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

# make aws filenames
aws_files = {file:file.split('/')[-1] for file in filenames}

# get gcs system
gcs = gcsfs.GCSFileSystem()

# loop over files
for filename in filenames:
    aws_filename = os.path.join(AWS_PATH, aws_files[filename])

    file_index = int(aws_files[filename][7:].split('.')[0])

    if file_index > 0:
        print(filename)
        gcp_to_aws(gcp_filename=filename, aws_filename=aws_filename, aws_bucket=AWS_BUCKET, gcs=gcs)




