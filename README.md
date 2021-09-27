# nowcasting_dataset
A multi-process data loader for PyTorch,
optimised for Google Cloud, which aligns three separate datasets:

* Satellite imagery (EUMETSAT SEVIRI RSS 5-minutely data of UK)
* Numerical Weather Predictions (NWPS.  UK Met Office UKV model)
* Solar PV power timeseries data (from PVOutput.org, downloaded using
  our [pvoutput Python
  code](https://github.com/openclimatefix/pvoutput).)

At the start of writing this code, the intention was to load and align
data from these three datasets on-the-fly during ML training.  And it
can still be used that way!  But it just isn't quite fast enough to
keep a modern GPU constantly fed with data when loading multiple
satellite channels and multiple NWP parameters.  So, now, this code is
used to pre-prepare thousands of batches, and save these batches to
disk, each as a separate NetCDF file.  These files can then be loaded
super-quickly at training time.  The end result is a 12x speedup in
training.

The script `scripts/prepare_ml_training_data.py` is used to
pre-compute the training and validation data (the script makes used of the
`nowcasting_dataset` library).
`nowcasting_dataset.dataset.datasets.NetCDFDataset` is a PyTorch Dataset, to
be used to train ML models.

This repo doesn't contain the ML models themselves.  The models are
in: https://github.com/openclimatefix/predict_pv_yield/ and
https://github.com/openclimatefix/satflow, and utils are in
https://github.com/openclimatefix/nowcasting_utils

# Installation

From within the cloned `nowcasting_dataset` directory:

```shell
conda env create -f environment.yml
conda activate nowcasting_dataset
pip install -e .
sudo apt install libgl1-mesa-glx  # For optical flow
```

A (probably older) version is also available through `pip install nowcasting-dataset`

To prevent PyTorch failing with an error like `RuntimeError: unable to open shared memory object </torch_2276740_2849291446> in read-write mode`, edit `/etc/security/limits.conf` as root and add this line: `*		 soft	 nofile		 512000` then log out and log back in again  (see [this issue](https://github.com/openclimatefix/nowcasting_dataset/issues/158) for more details).

## pip
You can install a local version with `pip install -e .`.
If you want to also install [PVLive](https://github.com/SheffieldSolar/PV_Live-API) then use `pip install git+https://github.com/SheffieldSolar/PV_Live-API
`

## Pre-commit

A pre commit hook has been installed which makes `black` run with every commit. You need to install
`black` and `pre-commit` (these will be installed by `conda` or `pip` when installing
`nowcasting_dataset`) and run `pre-commit install` in this repo.

# Testing

To test using the small amount of data stored in this repo: `py.test -s`

To test using the full dataset on Google Cloud, add the `--use_cloud_data` switch.

# Documentation

Please see the [`Example` class](https://github.com/openclimatefix/nowcasting_dataset/blob/main/nowcasting_dataset/dataset/example.py) for documentation about the different data fields in each example / batch.
