# nowcasting_dataset
Multi-process data loader for PyTorch, optimised for Google Cloud

# Installation

From within the cloned `nowcasting_dataset` directory:

```shell
conda env create -f environment.yml
conda activate nowcasting_dataset
pip install -e .
```

# Testing

To test using the small amount of data stored in this repo: `py.test -s`

To test using the full dataset on Google Cloud, add the `--use_cloud_data` switch.
