import logging

import pandas as pd
import xarray as xr

from typing import List, Union, Optional
from pathlib import Path
from datetime import datetime

from nowcasting_dataset.data_sources.gsp.eso import get_pv_gsp_metadata_from_eso

logger = logging.getLogger(__name__)


def get_list_of_gsp_ids(maximum_number_of_gsp: int) -> List[int]:
    """
    Get list of gsp ids from ESO metadata
    @param maximum_number_of_gsp: clib list by this amount.
    @return: list of gsp ids
    """
    # get a lit of gsp ids
    metadata = get_pv_gsp_metadata_from_eso()

    # get rid of nans, and duplicates
    metadata = metadata[~metadata['gsp_id'].isna()]
    metadata.drop_duplicates(subset=['gsp_id'], inplace=True)

    # make into list
    gsp_ids = metadata['gsp_id'].to_list()
    gsp_ids = [int(gsp_id) for gsp_id in gsp_ids]

    # adjust number of gsp_ids
    if maximum_number_of_gsp is None:
        maximum_number_of_gsp = len(metadata)
    if maximum_number_of_gsp > len(metadata):
        logging.warning(f'Only {len(metadata)} gsp available to load')
    if maximum_number_of_gsp < len(metadata):
        gsp_ids = gsp_ids[0: maximum_number_of_gsp]

    return gsp_ids


def load_solar_pv_gsp_data_from_gcs(
        filename: Union[str, Path],
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None) -> pd.DataFrame:
    """
    Load solar pv gsp data from gcs (although there is an option to load from local - for testing)
    @param filename: filename of file to be loaded, can put 'gs://' files in here too
    @param start_dt: the start datetime, which to trim the data to
    @param end_dt: the end datetime, which to trim the data to
    @return: dataframe of pv data
    """
    logger.debug('Loading Solar PV GCP Data from GCS')
    # Open data - it maye be quicker to open byte file first, but decided just to keep it like this at the moment
    pv_power = xr.open_zarr(filename)

    pv_power = pv_power.sel(datetime_gmt=slice(start_dt, end_dt))
    pv_power_df = pv_power.to_dataframe()

    # Save memory
    del pv_power

    # Process the data a little
    pv_power_df = pv_power_df.dropna(axis='columns', how='all')
    pv_power_df = pv_power_df.clip(lower=0, upper=5E7)

    # make column names ints, not strings
    pv_power_df.columns = [int(col) for col in pv_power_df.columns]

    return pv_power_df
