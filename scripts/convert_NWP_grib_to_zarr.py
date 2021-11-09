"""Convert Numerical Weather Prediction grib files to Zarr.

Use the script download_UK_Met_Office_NWPs_from_CEDA.sh to download grib files.

For documentation about the UK Met Office grib files, please see:
http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf

Todo:
* Start single-threaded
* Use click to set source and target directories.
* Get list of all available grib files (using glob).  Sort by init datetime.
* If Zarr file already exists then start at the most recent grib file.
* Loop round each grib file (in chronological order)
* Load with stepType=instant and stepType=accum.
* Append to Zarr.
"""
