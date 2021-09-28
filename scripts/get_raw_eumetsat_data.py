############
# Pull raw pv satellite data from EUMetSat
#
# 2021-09-28
# Jacob Bieker
#
# The data is about 1MB for a month of data
############
from datetime import datetime
import pytz
import yaml
import os
import numcodecs

from satip import eumetsat
from satip.eumetsat import compress_downloaded_files

from satflow.data.utils.utils import eumetsat_name_to_datetime, eumetsat_filename_to_datetime
from datetime import datetime, timedelta

from pathlib import Path
from nowcasting_dataset.cloud.local import delete_all_files_in_temp_path
from nowcasting_dataset.cloud.gcp import gcp_upload_and_delete_local_files
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

start = datetime(2018, 1, 1, tzinfo=pytz.utc)
end = datetime(2021, 1, 1, tzinfo=pytz.utc)
gcp_path = "gs://solar-pv-nowcasting-data/PV/GSP/v1"

config = {"start": start, "end": end, "gcp_path": gcp_path}
dm = eumetsat.DownloadManager(user_key, user_secret, data_dir, metadata_db_fp, debug_fp)

for day in range(0, 31):
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        for year in range(2010, 2022):
            curr_date = f"{year}/{month}/{day}"
            curr_date = datetime.strptime(curr_date, "%Y/%m/%d")
            # make_day((os.path.join(data_dir, curr_date.strftime("%Y/%m/%d/")), curr_date, 0))
            # continue
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:05",
                product_id="EO:EUM:DAT:MSG:RSS-CLM",
            )
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:05",
                product_id="EO:EUM:DAT:MSG:MSG15-RSS",
            )
            dm.download_date_range(f"{year}-{month}-{day} 07:59", f"{year}-{month}-{day} 20:05")

# format local temp folder
LOCAL_TEMP_PATH = Path("~/temp/").expanduser()
delete_all_files_in_temp_path(path=LOCAL_TEMP_PATH)

# get data
data_df = load_pv_gsp_raw_data_from_pvlive(start=start, end=end)

# upload to gcp
gcp_upload_and_delete_local_files(dst_path=gcp_path, local_path=LOCAL_TEMP_PATH)
