import logging
import os
from pathlib import Path

import torch

import nowcasting_dataset
from nowcasting_dataset.cloud.utils import get_maximum_batch_id
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.dataset.datasets import NetCDFDataset, worker_init_fn
from nowcasting_dataset.dataset.validate import ValidatorDataset

logging.basicConfig(format="%(asctime)s %(levelname)s %(pathname)s %(lineno)d %(message)s")
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.INFO)

logging.getLogger("nowcasting_dataset.data_source").setLevel(logging.WARNING)

# load configuration, this can be changed to a different filename as needed
filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")
config = load_yaml_configuration(filename=filename)

DST_NETCDF4_PATH = config.output_data.filepath
DST_TRAIN_PATH = os.path.join(DST_NETCDF4_PATH, "train")
DST_VALIDATION_PATH = os.path.join(DST_NETCDF4_PATH, "validation")
DST_TEST_PATH = os.path.join(DST_NETCDF4_PATH, "test")
LOCAL_TEMP_PATH = Path("~/temp/").expanduser()

# find how many datasets there are
maximum_batch_id_train = get_maximum_batch_id(f"gs://{DST_TRAIN_PATH}")
maximum_batch_id_validation = get_maximum_batch_id(f"gs://{DST_VALIDATION_PATH}")
maximum_batch_id_test = get_maximum_batch_id(f"gs://{DST_TEST_PATH}")

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
        configuration=config,
    ),
    **dataloader_config,
)

test_dataset = torch.utils.data.DataLoader(
    NetCDFDataset(
        maximum_batch_id_train,
        f"gs://{DST_TEST_PATH}",
        LOCAL_TEMP_PATH,
        configuration=config,
    ),
    **dataloader_config,
)

v_train_dataset = ValidatorDataset(configuration=config, batches=train_dataset)
v_validation_dataset = ValidatorDataset(configuration=config, batches=validation_dataset)
v_test_dataset = ValidatorDataset(configuration=config, batches=test_dataset)

# check there is no overlaps in the datasets
train_datetimes = v_train_dataset.day_datetimes
validation_datetimes = v_validation_dataset.day_datetimes
test_datetimes = v_test_dataset.day_datetimes

assert len(train_datetimes.join(validation_datetimes, how="inner")) == 0
assert len(train_datetimes.join(test_datetimes, how="inner")) == 0
assert len(validation_datetimes.join(test_datetimes, how="inner")) == 0

# TODO update configuiration file to say it has been validated
