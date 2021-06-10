from typing import Union
from pathlib import Path
import fsspec
import gcsfs
import xarray as xr


def open_zarr_on_gcp(
        filename: Union[str, Path],
        consolidated: bool = True
) -> xr.DataArray:
    """Lazily opens the Zarr store on Google Cloud Storage (GCS)."""
    # Clear reference to the loop and thread.  This is necessary otherwise
    # gcsfs hangs in the ML training loop.  Only required for fsspec >= 0.9.0
    # See https://github.com/dask/gcsfs/issues/379#issuecomment-839929801
    # TODO: Try deleting this two lines to make sure this is still relevant.
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None

    gcs = gcsfs.GCSFileSystem(access='read_only')
    store = gcsfs.GCSMap(root=filename, gcs=gcs)
    dataset = xr.open_zarr(store, consolidated=consolidated)
    return dataset
