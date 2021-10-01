import os

import torch

import nowcasting_dataset
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.dataset.datasets import worker_init_fn
from nowcasting_dataset.dataset.validate import FakeDataset, ValidatorDataset


def test_validate():

    local_path = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "../")

    # load configuration, this can be changed to a different filename as needed
    filename = os.path.join(local_path, "tests", "config", "test.yaml")
    config = load_yaml_configuration(filename)

    train_dataset = FakeDataset(
        configuration=config,
        length=10,
    )

    dataloader_config = dict(
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        # Disable automatic batching because dataset
        # returns complete batches.
        batch_size=None,
    )

    train_dataset = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

    ValidatorDataset(configuration=config, batches=train_dataset)
