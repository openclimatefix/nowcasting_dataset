import pyproj
from numbers import Number
from typing import Tuple


# OSGB is also called "OSGB 1936 / British National Grid -- United
# Kingdom Ordnance Survey".  OSGB is used in many UK electricity
# system maps, and is used by the UK Met Office UKV model.  OSGB is a
# Transverse Mercator projection, using 'easting' and 'northing'
# coordinates which are in meters.  See https://epsg.io/27700
OSGB = 27700

# WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
# latitude and longitude.
WGS84 = 4326

_osgb_to_lat_lon = pyproj.Transformer.from_crs(crs_from=OSGB, crs_to=WGS84)
_lat_lon_to_osgb = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB)


def osgb_to_lat_lon(x: Number, y: Number) -> Tuple[Number, Number]:
    """Returns 2-tuple of latitude (north-south), longitude (east-west).

    Args:
      x, y: Location in Ordnance Survey GB 1936, also known as
        British National Grid, coordinates.
    """
    return _osgb_to_lat_lon.transform(x, y)


def lat_lon_to_osgb(lat: Number, lon: Number) -> Tuple[Number, Number]:
    """Returns 2-tuple of x (east-west), y (north-south).

    Args:
      lat, lon: Location is WGS84 coordinates.
    """
    return _lat_lon_to_osgb.transform(lat, lon)
