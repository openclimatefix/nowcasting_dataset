import pandas as pd
import pyproj
from numbers import Number
from typing import Tuple
import datetime
import pvlib


# OSGB is also called "OSGB 1936 / British National Grid -- United
# Kingdom Ordnance Survey".  OSGB is used in many UK electricity
# system maps, and is used by the UK Met Office UKV model.  OSGB is a
# Transverse Mercator projection, using 'easting' and 'northing'
# coordinates which are in meters.  See https://epsg.io/27700
OSGB = 27700

# WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
# latitude and longitude.
WGS84 = 4326


class Transformers:
    """
    Class to store transformation from one Grid to another. Its good to make this only once, but need the
    option of updating them, due to out of data grids.
    """

    def __init__(self):

        self._osgb_to_lat_lon = None
        self._lat_lon_to_osgb = None
        self.make_transformers()

    def make_transformers(self):
        # Nice to only make these once, as it makes calling the functions below quicker
        self._osgb_to_lat_lon = pyproj.Transformer.from_crs(crs_from=OSGB, crs_to=WGS84)
        self._lat_lon_to_osgb = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB)

    @property
    def osgb_to_lat_lon(self):
        return self._osgb_to_lat_lon

    @property
    def lat_lon_to_osgb(self):
        return self._lat_lon_to_osgb


# make the transformers
transformers = Transformers()


def download_grids():
    """
    The transformer grid sometimes need updating
    """

    pyproj.transformer.TransformerGroup(crs_from=OSGB, crs_to=WGS84).download_grids(verbose=True)
    pyproj.transformer.TransformerGroup(crs_from=WGS84, crs_to=OSGB).download_grids(verbose=True)

    transformers.make_transformers()


def osgb_to_lat_lon(x: Number, y: Number) -> Tuple[Number, Number]:
    """Returns 2-tuple of latitude (north-south), longitude (east-west).

    Args:
      x, y: Location in Ordnance Survey GB 1936, also known as
        British National Grid, coordinates.
    """
    return transformers.osgb_to_lat_lon.transform(x, y)


def lat_lon_to_osgb(lat: Number, lon: Number) -> Tuple[Number, Number]:
    """Returns 2-tuple of x (east-west), y (north-south).

    Args:
      lat, lon: Location is WGS84 coordinates.
    """
    return transformers.lat_lon_to_osgb.transform(lat, lon)


def calculate_azimuth_and_elevation_angle(latitude: float, longitude: float, datestamps: [datetime]) -> pd.DataFrame:
    """
    Calculation the azimuth angle, and the elevation angle for several datetamps, but for one specific osgb location

    More details see: https://www.celestis.com/resources/faq/what-are-the-azimuth-and-elevation-of-a-satellite/

    Args:
        latitude: latitude of the pv site
        longitude: longitude of the pv site
        datestamps: list of datestamps to calculate the sun angles. i.e the sun moves from east to west in the day.

    Returns: Pandas data frame with the index the same as 'datestamps', with columns of "elevation" and "azimuth" that
    have been calculate.

    """

    # get the solor position
    solpos = pvlib.solarposition.get_solarposition(datestamps, latitude, longitude)

    # extract the information we want
    return solpos[["elevation", "azimuth"]]
