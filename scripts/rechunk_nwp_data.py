#!/usr/bin/env python3

import xarray as xr
from pathlib import Path
import numcodecs
import gcsfs
import rechunker
import zarr
from nowcasting_dataset.data_sources.nwp_data_source import open_nwp, NWP_VARIABLE_NAMES
import os
import numpy as np
# from dask import distributed, diagnostics


BUCKET = Path('solar-pv-nowcasting-data')
NWP_PATH = BUCKET / 'NWP/UK_Met_Office/'
SOURCE_PATH = 'gs://' + str(BUCKET / 'NWP/UK_Met_Office/UKV_zarr')
TARGET_PATH = NWP_PATH / 'UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr'
TEMP_STORE_FILENAME = NWP_PATH / 'temp.zarr'


def open_nwp(zarr_store: str) -> xr.Dataset:
    full_dir = os.path.join(SOURCE_PATH, zarr_store)
    ds = xr.open_dataset(
        full_dir, engine='zarr', consolidated=True, mode='r', chunks={})
    ds = ds.rename({'time': 'init_time'})

    # The isobaricInhPa coordinates look messed up, especially in
    # the 2018_7-12 and 2019_7-12 Zarr stores.  So let's drop all
    # the variables with multiple vertical levels for now:
    del ds['isobaricInhPa'], ds['gh_p'], ds['r_p'], ds['t_p']
    del ds['wdir_p'], ds['ws_p']

    # There are a lot of doubled-up indicies from 2018-07-18 00:00
    # to 2018-08-27 09:00.  De-duplicate the index. Code adapted
    # from https://stackoverflow.com/a/51077784/732596
    if zarr_store == '2018_7-12':
        _, unique_index = np.unique(ds.init_time, return_index=True)
        ds = ds.isel(init_time=unique_index)

    # 2019-02-01T21 is in the wrong place! It comes after
    # 2019-02-03T15.  Oops!
    if zarr_store == '2019_1-6':
        sorted_init_time = np.sort(ds.init_time)
        ds = ds.reindex(init_time=sorted_init_time)

    return ds


def main():
    #cluster = distributed.LocalCluster(n_workers=8, threads_per_worker=4)
    #client = distributed.Client(cluster)
    #print(client)

    nwp_datasets = []
    for zarr_store in ['2018_1-6', '2018_7-12', '2019_1-6', '2019_7-12']:
        print('opening', zarr_store)
        nwp_datasets.append(open_nwp(zarr_store))
    
    print('concat...')
    nwp_concatenated = xr.concat(nwp_datasets, dim='init_time')
        
    # Convert to array so we can chunk along the 'variable' axis
    source_dataset = nwp_concatenated[list(NWP_VARIABLE_NAMES)].to_array()
    #source_dataset = source_dataset.isel(init_time=slice(0, 2))
    source_dataset = source_dataset.to_dataset(name='UKV')
    print('source_dataset:\n', source_dataset)

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
            'compressor': numcodecs.Blosc(cname="zstd", clevel=5),
            'dtype': 'float32'
        }
    }

    print('Rechunking...')
    rechunk_plan = rechunker.rechunk(
        source=source_dataset,
        target_chunks=target_chunks,
        max_mem="3GB",
        target_store=target_store,
        target_options=encoding,
        temp_store=temp_store)

    #with diagnostics.ProgressBar():
    rechunk_plan.execute()

    print('Consolidating...')
    zarr.convenience.consolidate_metadata(target_store)

    print('Done!')

if __name__ == '__main__':
    main()
