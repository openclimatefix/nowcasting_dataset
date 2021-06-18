#!/usr/bin/env python3

import xarray as xr
from pathlib import Path
import numcodecs
import gcsfs
import rechunker
from dask.diagnostics import ProgressBar


BUCKET = Path('solar-pv-nowcasting-data')
SAT_PATH = BUCKET / 'satellite/EUMETSAT/SEVIRI_RSS/OSGB36/'
SOURCE_SAT_FILENAME = 'gs://' + str(SAT_PATH / 'all_zarr_int16')
TARGET_SAT_FILENAME = SAT_PATH / 'all_zarr_int16_single_timestep.zarr'
TEMP_STORE_FILENAME = SAT_PATH / 'temp.zarr'


def main():
    source_sat_dataset = xr.open_zarr(SOURCE_SAT_FILENAME, consolidated=True)

    gcs = gcsfs.GCSFileSystem()
    target_store = gcs.get_mapper(TARGET_SAT_FILENAME)
    temp_store = gcs.get_mapper(TEMP_STORE_FILENAME)

    target_chunks = {
        'stacked_eumetsat_data': {
            "time": 1,
            "y": 704,
            "x": 548,
            "variable": 1}}

    encoding = {
        'stacked_eumetsat_data': {
            'compressor': numcodecs.Blosc(cname="zstd", clevel=5)}}

    rechunk_plan = rechunker.rechunk(
        source=source_sat_dataset,
        target_chunks=target_chunks,
        max_mem="10GB",
        target_store=target_store,
        target_options=encoding,
        temp_store=temp_store)

    rechunk_plan.execute()


if __name__ == '__main__':
    main()
