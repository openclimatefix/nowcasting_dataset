import pandas as pd


import urllib
import json
import pandas as pd

from pvlive_api import PVLive
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go


def get_gsp_metadata_from_eso() -> pd.DataFrame:
    """
    Get the metadata for the gsp, from ESO.
    @return:
    """

    # call ESO website
    url = "https://data.nationalgrideso.com/api/3/action/datastore_search?" \
          "resource_id=bbe2cc72-a6c6-46e6-8f4e-48b879467368&limit=400"
    fileobj = urllib.request.urlopen(url)
    d = json.loads(fileobj.read())

    # make dataframe
    results = d["result"]["records"]
    return pd.DataFrame(results)
