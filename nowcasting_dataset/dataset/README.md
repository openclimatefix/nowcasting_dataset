# Datasets

This folder contains the following files

## batch.py

Functions used to 'play with' batch data, where "batch data" is a List of Example objects; i.e. `List[Example]`.

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


## example.py

Main thing in here is a Typed Dictionary. This is used to store one element of data use for one step in the ML models.
There is also a validation function. See this file for documentation about exactly what data is available in each ML
training Example.
