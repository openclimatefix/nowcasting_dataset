""" Script to adjust metadata data, ready for nowcasting"""
import pandas as pd

"""
need to make sure the metadata has the follow columns
- system_id
- longitude
- latitude
"""


dir = "gs://solar-pv-nowcasting-data/PV/Passive/20211027_Passiv_PV_Data"
output_dir = "gs://solar-pv-nowcasting-data/PV/Passive/ocf_formatted/v0"

# ### metadata
filename = f"{dir}/system_metadata.csv"
passive_metadata = pd.read_csv(filename)

filename = f"{dir}/llsoa_centroids.csv"
passive_llsoacd = pd.read_csv(filename)

# join llsoacdf data
passive_metadata = passive_metadata.merge(passive_llsoacd, on="llsoacd", how="left")

passive_metadata.rename(columns={"ss_id": "system_id"})

assert "system_id" in passive_metadata.columns
assert "longitude" in passive_metadata.columns
assert "latitude" in passive_metadata.columns

passive_metadata.to_csv(f"{output_dir}/system_metadata.csv")

# ### metadata OCF only

filename = f"{dir}/system_metadata_OCF_ONLY.csv"

passive_metadata_ocf = pd.read_csv(filename)

passive_metadata.rename(
    columns={"ss_id": "system_id", "longitude": "longitude_rounded", "latitude": "latitude_rounded"}
)

assert "system_id" in passive_metadata_ocf.columns
assert "longitude" in passive_metadata_ocf.columns
assert "latitude" in passive_metadata_ocf.columns

passive_metadata.to_csv(f"{output_dir}/system_metadata_OCF_ONLY.csv")
