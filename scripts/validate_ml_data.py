import logging
import os
from pathlib import Path

import nowcasting_dataset
import torch
from nowcasting_dataset.config.load import load_configuration_from_gcs, load_yaml_configuration
from nowcasting_dataset.dataset.datasets import NetCDFDataset, worker_init_fn
from nowcasting_dataset.utils import get_maximum_batch_id_from_gcs

logging.basicConfig(format="%(asctime)s %(levelname)s %(pathname)s %(lineno)d %(message)s")
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.DEBUG)

logging.getLogger("nowcasting_dataset.data_source").setLevel(logging.WARNING)

# load configuration, this can be changed to a different filename as needed
filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")
config = load_configuration_from_gcs(gcp_dir="prepared_ML_training_data/v5/")
config = load_yaml_configuration(filename=filename)

DST_NETCDF4_PATH = config.output_data.filepath
DST_TRAIN_PATH = os.path.join(DST_NETCDF4_PATH, "train")
DST_VALIDATION_PATH = os.path.join(DST_NETCDF4_PATH, "validation")
DST_TEST_PATH = os.path.join(DST_NETCDF4_PATH, "test")
LOCAL_TEMP_PATH = Path("~/temp/").expanduser()

# find how many datasets there are
maximum_batch_id_train = get_maximum_batch_id_from_gcs(f"gs://{DST_TRAIN_PATH}")
maximum_batch_id_validation = get_maximum_batch_id_from_gcs(f"gs://{DST_VALIDATION_PATH}")
maximum_batch_id_test = get_maximum_batch_id_from_gcs(f"gs://{DST_TEST_PATH}")

dataloader_config = dict(
    pin_memory=True,
    num_workers=8,
    prefetch_factor=8,
    worker_init_fn=worker_init_fn,
    persistent_workers=True,
    # Disable automatic batching because dataset
    # returns complete batches.
    batch_size=None,
)

# train dataset
train_dataset = torch.utils.data.DataLoader(
    NetCDFDataset(
        maximum_batch_id_train,
        f"gs://{DST_TRAIN_PATH}",
        LOCAL_TEMP_PATH,
        cloud="gcp",
        configuration=config,
    ),
    **dataloader_config,
)

# validation dataset
validation_dataset = torch.utils.data.DataLoader(
    NetCDFDataset(
        maximum_batch_id_train,
        f"gs://{DST_VALIDATION_PATH}",
        LOCAL_TEMP_PATH,
        cloud="gcp",
        configuration=config,
    ),
    **dataloader_config,
)

test_dataset = torch.utils.data.DataLoader(
    NetCDFDataset(
        maximum_batch_id_train,
        f"gs://{DST_TEST_PATH}",
        LOCAL_TEMP_PATH,
        cloud="gcp",
        configuration=config,
    ),
    **dataloader_config,
)

# validate all datasets
train_dataset.validate()
validation_dataset.validate()
test_dataset.validate()

# check there is no overlaps in the datasets
train_datetimes = train_dataset.day_datetimes
validation_datetimes = validation_dataset.day_datetimes
test_datetimes = validation_dataset.day_datetimes

assert len(train_datetimes.join(validation_datetimes, how="inner")) == 0
assert len(train_datetimes.join(test_datetimes, how="inner")) == 0
assert len(validation_datetimes.join(test_datetimes, how="inner")) == 0

# TODO update configuiration file to say it has been validated
