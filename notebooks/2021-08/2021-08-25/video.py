import os

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nowcasting_dataset.dataset import SAT_MEAN, SAT_STD, NetCDFDataset

DATA_PATH = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v4/"
TEMP_PATH = ""

# set up data generator
train_dataset = NetCDFDataset(
    24_900, os.path.join(DATA_PATH, "train"), os.path.join(TEMP_PATH, "train")
)

train_dataset.per_worker_init(1)
train_dataset_iterator = iter(train_dataset)

# get batch of data, this may take a few seconds to run
data = next(train_dataset_iterator)

# get the timestamp of the image
batch_index = 6
sat_datetimes = pd.to_datetime(data["sat_datetime_index"][batch_index], unit="s")
print(sat_datetimes)

x_coordinates = data["sat_x_coords"][6]
x_max = np.max(x_coordinates)
x_min = np.min(x_coordinates)
print(x_coordinates)
print(x_min, x_max)

y_coordinates = data["sat_y_coords"][6]
y_max = np.max(y_coordinates)
y_min = np.min(y_coordinates)
print(y_coordinates)
print(y_min, y_max)

# get satellite image, currently from https://github.com/openclimatefix/py-staticmaps
import staticmaps

from nowcasting_dataset.geospatial import osgb_to_lat_lon

bottom_left = osgb_to_lat_lon(x=x_min, y=y_min)
top_right = osgb_to_lat_lon(x=x_max, y=y_max)

bottom_left = staticmaps.create_latlng(bottom_left[0], bottom_left[1])
top_right = staticmaps.create_latlng(top_right[0], top_right[1])

context = staticmaps.Context()
map = context.make_clean_map_from_bounding_box(
    width=640, height=640, bottom_left=bottom_left, top_right=top_right
)
map.save("map.png")
map = np.array(map)
# height, weight, 4
trace_map = go.Image(z=map)

# make rgb image sequence from satelight image
# want to use IR_016, VIS006 and VIS008 for rgb
# https://www.britannica.com/science/color/The-visible-spectrum,
# wavelengths
# red 650
# blue 450
# green 550
channel_indexes = [1, 8, 9]
satellite_data = []
for channel_index in channel_indexes:

    # renormalize
    satellite_data.append(
        data["sat_data"][batch_index, :, :, :, channel_index] * SAT_STD.values[channel_index]
        + SAT_MEAN.values[channel_index]
    )
    # which is shape seq_length, width, height

satellite_rgb_data = np.stack(satellite_data, axis=3).astype(np.uint8)
# which is shape seq_length, width, height, 3

satellite_reszied_data = []
for i in range(0, satellite_rgb_data.shape[0]):
    # resize
    satellite_reszied_data.append(cv2.resize(satellite_rgb_data[i], (map.shape[1], map.shape[0])))
satellite_rgb_data = np.stack(satellite_reszied_data, axis=0).astype(np.uint8)


# overlay satellite images on top of static map
alpha = 0.5  # the amount of background map


frames = []
for i in range(0, satellite_rgb_data.shape[0]):
    z = satellite_rgb_data[i] * alpha + map[:, :, 0:3] * (1 - alpha)
    frames.append(go.Frame(data=[go.Image(z=z)], layout=go.Layout(title=str(sat_datetimes[i]))))

fig = go.Figure(
    frames=frames,
    data=trace_map,
    layout=go.Layout(
        updatemenus=[
            dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])
        ],
    ),
)
fig.show(renderer="browser")
