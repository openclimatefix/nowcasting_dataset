# nowcasting_dataset
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![codecov](https://codecov.io/gh/openclimatefix/nowcasting_dataset/branch/main/graph/badge.svg?token=X0P4KTHWVA)](https://codecov.io/gh/openclimatefix/nowcasting_dataset)


Pre-prepare batches of data for use in machine learning training.

This code combines several data sources including:

* Satellite imagery (EUMETSAT SEVIRI RSS 5-minutely data of UK)
* Numerical Weather Predictions (NWPs.  UK Met Office UKV model from CEDA)
* Solar PV power timeseries data (from PVOutput.org, downloaded using
  our [pvoutput Python code](https://github.com/openclimatefix/pvoutput).)
* Estimated total solar PV generation for each of the ~350 "grid supply points"
  (GSPs) in Britain from [Sheffield Solar's PV Live Regional API](https://www.solar.sheffield.ac.uk/pvlive/regional/).
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

### `pip`

A (probably older) version is also available through `pip install nowcasting-dataset`


### PV Live API
If you want to also install [PVLive](https://github.com/SheffieldSolar/PV_Live-API) then use `pip install git+https://github.com/SheffieldSolar/PV_Live-API
`

### Pre-commit

A pre commit hook has been installed which makes `black` run with every commit. You need to install
`black` and `pre-commit` (these will be installed by `conda` or `pip` when installing
`nowcasting_dataset`) and run `pre-commit install` in this repo.


## Testing

To test using the small amount of data stored in this repo: `py.test -s`

To output debug logs while running the tests then run `py.test --log-cli-level=10`

To test using the full dataset on Google Cloud, add the `--use_cloud_data` switch.


## Downloading data

### Satellite data

Use [Satip](https://github.com/openclimatefix/Satip) to download
  native EUMETSAT SEVIRI RSS data from EUMETSAT's API and then convert
  to an intermediate file format.


### PV data from PVOutput.org

Download PV timeseries data from PVOutput.org using
[our PVOutput code](https://github.com/openclimatefix/pvoutput).


### Numerical weather predictions from the UK Met Office

Please use our [`nwp`](https://github.com/openclimatefix/nwp) code to download UKV NWPs and convert to Zarr.


### GSP-level estimates of PV outturn from PV Live Regional

TODO - GSP


### Topographical data

1. Make an account at the [USGS EarthExplorer](https://earthexplorer.usgs.gov/) website
2. Create a region of the world to download data for, in our case, the spatial extant of the SEVIRI RSS image
3. Select the data products you want, in this case SRTM elevation maps
4. Download all the SRTM files that cover that area

There does not seem to be an automated way to do this selecting and downloading, so this might take awhile.


## Configure `nowcasting_dataset` to point to the downloaded data

Copy and modify one of the config yaml files in
[`nowcasting_dataset/config/`](https://github.com/openclimatefix/nowcasting_dataset/tree/main/nowcasting_dataset/config).


## Prepare ML batches

Run [`scripts/prepare_ml_data.py --help`](https://github.com/openclimatefix/nowcasting_dataset/blob/main/scripts/prepare_ml_data.py)
to learn how to run the `prepare_ml_data.py` script.


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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://jack-kelly.com"><img src="https://avatars.githubusercontent.com/u/460756?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jack Kelly</b></sub></a><br /><a href="https://github.com/openclimatefix/nowcasting_dataset/commits?author=JackKelly" title="Code">💻</a></td>
    <td align="center"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jacob Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/nowcasting_dataset/commits?author=jacobbieker" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/nowcasting_dataset/commits?author=peterdudfield" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/flowirtz"><img src="https://avatars.githubusercontent.com/u/6052785?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Flo</b></sub></a><br /><a href="https://github.com/openclimatefix/nowcasting_dataset/commits?author=flowirtz" title="Code">💻</a></td>
    <td align="center"><a href="https://rohancalum.github.io/"><img src="https://avatars.githubusercontent.com/u/42122330?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Rohan Nuttall</b></sub></a><br /><a href="https://github.com/openclimatefix/nowcasting_dataset/commits?author=rohancalum" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/lenassero"><img src="https://avatars.githubusercontent.com/u/21358816?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nasser Benabderrazik</b></sub></a><br /><a href="https://github.com/openclimatefix/nowcasting_dataset/commits?author=lenassero" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/vnshanmukh"><img src="https://avatars.githubusercontent.com/u/67438038?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Shanmukh Chava</b></sub></a><br /><a href="https://github.com/openclimatefix/nowcasting_dataset/commits?author=vnshanmukh" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
