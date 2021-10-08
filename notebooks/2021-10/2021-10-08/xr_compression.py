import os

import numpy as np
import xarray as xr
from nowcasting_dataset.utils import coord_to_range


def get_satellite_xrarray_data_array(
    batch_size, seq_length_5, satellite_image_size_pixels, number_sat_channels=10
):

    r = np.random.randn(
        # self.batch_size,
        seq_length_5,
        satellite_image_size_pixels,
        satellite_image_size_pixels,
        number_sat_channels,
    )

    time = np.sort(np.random.randn(seq_length_5))

    x_coords = np.sort(np.random.randint(0, 1000, (satellite_image_size_pixels)))
    y_coords = np.sort(np.random.randint(0, 1000, (satellite_image_size_pixels)))[::-1].copy()

    sat_xr = xr.DataArray(
        data=r,
        dims=["time", "x", "y", "channels"],
        coords=dict(
            # batch=range(0,self.batch_size),
            x=list(x_coords),
            y=list(y_coords),
            time=list(time),
            channels=range(0, number_sat_channels),
        ),
        attrs=dict(
            description="Ambient temperature.",
            units="degC",
        ),
        name="sata_data",
    )

    return sat_xr


def sat_data_array_to_dataset(sat_xr):
    ds = sat_xr.to_dataset(name="sat_data")
    # ds["sat_data"] = ds["sat_data"].astype(np.int16)

    for dim in ["time", "x", "y"]:
        # This does seem like the right way to do it
        # https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Saving_Datasets_and_DataArrays_to_NetCDF.html
        ds = coord_to_range(ds, dim, prefix="sat")
    ds = ds.rename(
        {
            "channels": f"sat_channels",
            "x": f"sat_x",
            "y": f"sat_y",
        }
    )

    # ds["sat_x_coords"] = ds["sat_x_coords"].astype(np.int32)
    # ds["sat_y_coords"] = ds["sat_y_coords"].astype(np.int32)

    return ds


def to_netcdf(batch_xr, local_filename):
    encoding = {name: {"compression": "lzf"} for name in batch_xr.data_vars}
    batch_xr.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)


# 1. try to save netcdf files not using coord to range function
sat_xrs = [get_satellite_xrarray_data_array(4, 19, 32) for _ in range(0, 10)]

### error ###
# cant do this step as x/y index has duplicate values
sat_dataset = xr.merge(sat_xrs)
to_netcdf(sat_dataset, "test_no_alignment.nc")
###

# but can save it as separate files
os.mkdir("test_no_alignment")
[sat_xrs[i].to_netcdf(f"test_no_alignment/{i}.nc", engine="h5netcdf") for i in range(0, 10)]
# 10 files about 1.5MB

# 2.
sat_xrs = [get_satellite_xrarray_data_array(4, 19, 32) for _ in range(0, 10)]
sat_xrs = [sat_data_array_to_dataset(sat_xr) for sat_xr in sat_xrs]

sat_dataset = xr.concat(sat_xrs, dim="example")
to_netcdf(sat_dataset, "test_alignment.nc")
# this 15 MB


# conclusion
# no major improvement in compression by joining datasets together, buts by joining array together,
# it does make it easier to get array ready ML
