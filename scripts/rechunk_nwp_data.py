#!/usr/bin/env python3

import xarray as xr
from pathlib import Path
import numcodecs
import gcsfs
import rechunker
import zarr
from nowcasting_dataset.data_sources.nwp_data_source import open_nwp, NWP_VARIABLE_NAMES


BUCKET = Path('solar-pv-nowcasting-data')
NWP_PATH = BUCKET / 'NWP/UK_Met_Office/'
SOURCE_PATH = 'gs://' + str(BUCKET / 'NWP/UK_Met_Office/UKV_zarr')
TARGET_PATH = NWP_PATH / 'UKV_single_step_and_single_timestep_all_vars.zarr'
TEMP_STORE_FILENAME = NWP_PATH / 'temp3.zarr'


def main():
    source_dataset = open_nwp(base_path=SOURCE_PATH, consolidated=True)
    source_dataset = source_dataset[list(NWP_VARIABLE_NAMES)].to_array()
    source_dataset = source_dataset.isel(init_time=slice(0, 1000))
    source_dataset = source_dataset.to_dataset(name='UKV')
    print(source_dataset)

    gcs = gcsfs.GCSFileSystem()
    target_store = gcs.get_mapper(TARGET_PATH)
    temp_store = gcs.get_mapper(TEMP_STORE_FILENAME)

    target_chunks = {
        'UKV': {
            "variable": 10,
            "init_time": 1,
            "step": 1,
            "x": 548,
            "y": 704
        }
    }

    encoding = {
        'UKV': {
            'compressor': numcodecs.Blosc(cname="zstd", clevel=5)
        }
    }

    print('Rechunking...')
    rechunk_plan = rechunker.rechunk(
        source=source_dataset,
        target_chunks=target_chunks,
        max_mem="1GB",
        target_store=target_store,
        target_options=encoding,
        temp_store=temp_store)

    rechunk_plan.execute()

    print('Consolidating...')
    zarr.convenience.consolidate_metadata(target_store)

    print('Done!')

if __name__ == '__main__':
    main()
