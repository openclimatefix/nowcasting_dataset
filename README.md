# nowcasting_dataset
Pre-prepare batches of data for use in machine learning training.

This code combines several data sources including:

* Satellite imagery (EUMETSAT SEVIRI RSS 5-minutely data of UK)
* Numerical Weather Predictions (NWPs.  UK Met Office UKV model from CEDA)
* Solar PV power timeseries data (from PVOutput.org, downloaded using
  our [pvoutput Python code](https://github.com/openclimatefix/pvoutput).)
* Topographic data.
* The Sun's azimuth and angle.

This repo doesn't contain the ML models themselves.  Please see [this
page for an overview](https://github.com/openclimatefix/nowcasting) of
the Open Climate Fix solar PV nowcasting project, and how our code
repositories fit together.


# User manual

## Installation

### `conda`

From within the cloned `nowcasting_dataset` directory:

```shell
conda env create -f environment.yml
conda activate nowcasting_dataset
pip install -e .
```

Note you can install the [pytorch](https://github.com/pytorch/pytorch)
and [pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning) using
```shell
pip install -e .[torch]
```
but it is only used to create a dataloader for machine learning models, and will not be necessary
soon (when the dataloader is moved to `nowcasting_dataloader`).


### `pip`

A (probably older) version is also available through `pip install nowcasting-dataset`


### `RuntimeError: unable to open shared memory object`

To prevent PyTorch failing with an error like `RuntimeError: unable to open shared memory object </torch_2276740_2849291446> in read-write mode`, edit `/etc/security/limits.conf` as root and add this line: `*		 soft	 nofile		 512000` then log out and log back in again  (see [this issue](https://github.com/openclimatefix/nowcasting_dataset/issues/158) for more details).


### PV Live API
If you want to also install [PVLive](https://github.com/SheffieldSolar/PV_Live-API) then use `pip install git+https://github.com/SheffieldSolar/PV_Live-API
`

### Pre-commit

A pre commit hook has been installed which makes `black` run with every commit. You need to install
`black` and `pre-commit` (these will be installed by `conda` or `pip` when installing
`nowcasting_dataset`) and run `pre-commit install` in this repo.


## Testing

To test using the small amount of data stored in this repo: `py.test -s`

To test using the full dataset on Google Cloud, add the `--use_cloud_data` switch.


## Downloading data

### Satellite data

Use [Satip](https://github.com/openclimatefix/Satip) to download
  native EUMETSAT SEVIRI RSS data from EUMETSAT's API and then convert
  to an intermediate file format.


### PV data from PVOutput.org

Download PV timeseries data from PVOutput.org using
our PVOutput code](https://github.com/openclimatefix/pvoutput).


### Numerical weather predictions from the UK Met Office

Request access to the [UK Met Office data on CEDA](https://catalogue.ceda.ac.uk/uuid/f47bc62786394626b665e23b658d385f).

Once you have a username and password, download using:

```shell
wget --user=<username> --password=<password> --recursive -nH --cut-dirs=5 --no-clobber \
--reject-regex "[[:digit:]]{8}(03|09|15|21)00.*\.grib$" \
--reject-regex "T120\.grib$" \
--reject-regex "Wholesale5.*\.grib$" \
ftp://ftp.ceda.ac.uk/badc/ukmo-nwp/data/ukv-grib
```

(You probably want to run this in a `gnu screen` session if you're SSH'ing into a VM or remote server).

What are all those `--reject-regex` instructions doing?

* `--reject-regex "[[:digit:]]{8}(03|09|15|21)00.*\.grib$"` rejects all NWPs initialised at
  3, 9, 15, or 21 hours (and so you end up with "only" four initialisations per day: 00, 06, 12, 18).
* `--reject-regex "T120\.grib$"` rejects the `T120` files, which contain forecast steps from
  2 days and 9 hours ahead, to 5 days ahead, in 3-hourly increments.  So we accept the
  `Wholesale[1234].grib` files (steps from 00:00 to 1 day and 12 hours ahead, in hourly increments)
  and `Wholesale[1234]T54.grib` files (step runs from 1 day and 13 hours ahead to 2 days and 6 hours
  ahead.  Hourly increments from 1 day and 13 hours ahead to 2 days ahead.
  Then 3-hourly increments).
* `--reject-regex "Wholesale5.*\.grib$"` rejects the `Wholesale5` files, which are just static
  topography data, so no need to download multiple copies of this data!

Detailed docs of the Met Office data is available [here](http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf).


### GSP-level estimates of PV outturn from PV Live Regional

TODO


### Topographical data

TODO


## Configure `nowcasting_dataset` to point to the downloaded data

Copy and modify one of the config yaml files in
[`nowcasting_dataset/config/`](https://github.com/openclimatefix/nowcasting_dataset/tree/main/nowcasting_dataset/config)


## Prepare ML batches

Run [`scripts/prepare_ml_data.py`](https://github.com/openclimatefix/nowcasting_dataset/blob/main/scripts/prepare_ml_data.py)


## Load prepared ML batches into an ML model

`nowcasting_dataset.dataset.datasets.NetCDFDataset` is a PyTorch
Dataset which loads the pre-prepared batches during ML training
(although this will soon be moved to a separate
[`nowcasting_dataloader`
repository](https://github.com/openclimatefix/nowcasting_dataloader)).


## What exactly is in each batch?

Please see the `data_sources/<modality>/<modality>_model.py` files
(where `<modality>` is one of {datetime, metadata, gsp, nwp, pv,
satellite, sun, topographic}) for documentation about the different
data fields in each example / batch.


# History of nowcasting_dataset
When we first started writing `nowcasting_dataset`, our intention was
to load and align data from these three datasets on-the-fly during ML
training.  But it just isn't quite fast enough to keep a modern GPU constantly fed
with data when loading multiple satellite channels and multiple NWP
parameters.  So, now, this code is used to pre-prepare thousands of
batches, and save these batches to disk, each as a separate NetCDF
file.  These files can then be loaded super-quickly at training time.
The end result is a 12x speedup in training.
