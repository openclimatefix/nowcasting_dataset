# Datasets

This folder contains the following files

## batch.py

'Batch' pydantic class, to hold batch data in.

## datamodule.py

Contains a class NowcastingDataModule - pl.LightningDataModule
This handles the
 - amalgamation of all different data sources,
 - making valid datetimes across all the sources,
 - splitting into train and validation datasets


## datasets.py

This file contains the following classes

NetCDFDataset - torch.utils.data.Dataset: Use for loading pre-made batches
NowcastingDataset - torch.utils.data.IterableDataset: Dataset for making batches


## validate.py

Contains a class that can validate the prepare ml dataset
