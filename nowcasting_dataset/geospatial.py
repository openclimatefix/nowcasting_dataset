""" Geospatial functions """
from datetime import datetime
from numbers import Number
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pvlib
import pyproj
import xarray as xr
from pyresample.area_config import load_area_from_string

# OSGB is also called "OSGB 1936 / British National Grid -- United
# Kingdom Ordnance Survey".  OSGB is used in many UK electricity
# system maps, and is used by the UK Met Office UKV model.  OSGB is a
# Transverse Mercator projection, using 'easting' and 'northing'
# coordinates which are in meters.  See https://epsg.io/27700
OSGB = 27700

# WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
# latitude and longitude.
WGS84 = 4326
WGS84_CRS = f"EPSG:{WGS84}"

EUMETSAT_MSG_SEVIRI_AREA_DEF = """
msg_seviri_rss_1km:
  description: MSG SEVIRI Rapid Scanning Service area definition with 1 km resolution
  projection:
    proj: geos
    lon_0: 9.5
    h: 35785831
    x_0: 0
    y_0: 0
    a: 6378169
    rf: 295.488065897014
    no_defs: null
    type: crs
  shape:
    height: 891
    width: 1843
  area_extent:
    lower_left_xy: [28503.828942775726, 5088183.4998726845]
    upper_right_xy: [-1814743.776023388, 4197063.795030117]
    units: m
"""

_T_NUM_OR_ARRAY = Union[Number, np.ndarray, Sequence[Number]]


class Transformers:
    """
    Class to store transformation from one Grid to another.

    Its good to make this only once, but need the
    option of updating them, due to out of data grids.
    """

    def __init__(self):
        """Init"""
        self._osgb_to_lat_lon = None
        self._lat_lon_to_osgb = None
        self._osgb_to_geostationary = None
        self.make_transformers()

    def make_transformers(self):
        """
        Make transformers

         Nice to only make these once, as it makes calling the functions below quicker
        """
        self._osgb_to_lat_lon = pyproj.Transformer.from_crs(crs_from=OSGB, crs_to=WGS84)
        self._lat_lon_to_osgb = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB)
        geostationary_area_def = load_area_from_string(EUMETSAT_MSG_SEVIRI_AREA_DEF)
        geostationary_crs = geostationary_area_def.crs
        self._osgb_to_geostationary = pyproj.Transformer.from_crs(
            crs_from=OSGB, crs_to=geostationary_crs
        )

    @property
    def osgb_to_lat_lon(self):
        """OSGB to lat-lon property"""
        return self._osgb_to_lat_lon

    @property
    def lat_lon_to_osgb(self):
        """lat-lon to OSGB property"""
        return self._lat_lon_to_osgb

    @property
    def osgb_to_geostationary(self):
        """Convert from OSGB to geostationary coordinates."""
        return self._osgb_to_geostationary


# make the transformers
transformers = Transformers()


def download_grids():
    """The transformer grid sometimes need updating"""
    pyproj.transformer.TransformerGroup(crs_from=OSGB, crs_to=WGS84).download_grids(verbose=True)
    pyproj.transformer.TransformerGroup(crs_from=WGS84, crs_to=OSGB).download_grids(verbose=True)

    transformers.make_transformers()


def osgb_to_lat_lon(x: Number, y: Number) -> Tuple[Number, Number]:
    """
    Change OSGB coordinates to lat, lon

    Args:
        x: osgb east-west
        y: osgb north-south

    Return: 2-tuple of latitude (north-south), longitude (east-west).

    """
    return transformers.osgb_to_lat_lon.transform(x, y)


def lat_lon_to_osgb(lat: Number, lon: Number) -> Tuple[Number, Number]:
    """
    Change lat, lon to a OSGB coordinates

    Args:
        lat: latitude
        lon: longitude

    Return: 2-tuple of x (east-west), y (north-south).

    """
    return transformers.lat_lon_to_osgb.transform(lat, lon)


def osgb_to_geostationary(
    x: _T_NUM_OR_ARRAY, y: _T_NUM_OR_ARRAY
) -> tuple[_T_NUM_OR_ARRAY, _T_NUM_OR_ARRAY]:
    """
    Change OSGB coordinates to geostationary.

    Args:
        x: OSGB east-west.
        y: OSGB north-south.

    Return: 2-tuple of geostationary x, y.
    """
    return transformers.osgb_to_geostationary.transform(x, y)


def calculate_azimuth_and_elevation_angle(
    latitude: float, longitude: float, datestamps: list[datetime]
) -> pd.DataFrame:
    """
    Calculation the azimuth angle, and the elevation angle for several datetamps.

    But for one specific osgb location

    More details see:
    https://www.celestis.com/resources/faq/what-are-the-azimuth-and-elevation-of-a-satellite/

    Args:
        latitude: latitude of the pv site
        longitude: longitude of the pv site
        datestamps: list of datestamps to calculate the sun angles. i.e the sun moves from east to
            west in the day.

    Returns: Pandas data frame with the index the same as 'datestamps', with columns of
    "elevation" and "azimuth" that have been calculate.

    """
    # get the solor position
    solpos = pvlib.solarposition.get_solarposition(datestamps, latitude, longitude)

    # extract the information we want
    return solpos[["elevation", "azimuth"]]


def get_osgb_center_from_list_of_x_and_y_osgb(
    x_osgb: Union[xr.DataArray, List[float], np.ndarray],
    y_osgb: Union[xr.DataArray, List[float], np.ndarray],
) -> (float, float):
    """
    Get the OSGB center from OSGB coords

    This gets the average of the x coordinates,
    and the average of the y coordinates.

    Args:
        x_osgb: list of x coords in OSGB
        y_osgb: list of y coords in OSGB

    Returns: x and y center [OSGB]

    """

    center_x_osgb = np.mean(x_osgb)
    center_y_osgb = np.mean(y_osgb)

    return center_x_osgb, center_y_osgb


def get_lat_lon_center_from_list_of_x_and_y_osgb(
    x_osgb: Union[xr.DataArray, List[float], np.ndarray],
    y_osgb: Union[xr.DataArray, List[float], np.ndarray],
) -> (float, float):
    """
    Get the center lat and lon coords from a list of OSGB coords

    Args:
        x_osgb: list of x coords in OSGB
        y_osgb: list of y coords in OSGB

    Returns: x and y center [lat, lon]

    """
    center_x_osgb, center_y_osgb = get_osgb_center_from_list_of_x_and_y_osgb(
        x_osgb=x_osgb, y_osgb=y_osgb
    )

    return osgb_to_lat_lon(center_x_osgb, center_y_osgb)
