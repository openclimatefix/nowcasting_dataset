import os

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# get satellite image, currently from https://github.com/openclimatefix/py-staticmaps
import staticmaps
from plotly.subplots import make_subplots

from nowcasting_dataset.dataset import SAT_MEAN, SAT_STD, NetCDFDataset
from nowcasting_dataset.geospatial import osgb_to_lat_lon

##############
# Setup data generator
##############

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
batch_index = 4
sat_datetimes = pd.to_datetime(data["sat_datetime_index"][batch_index], unit="s")
print(sat_datetimes)

x_coordinates = data["sat_x_coords"][batch_index]
x_max = np.max(x_coordinates)
x_min = np.min(x_coordinates)
print(x_coordinates)
print(x_min, x_max)

y_coordinates = data["sat_y_coords"][batch_index]
y_max = np.max(y_coordinates)
y_min = np.min(y_coordinates)
print(y_coordinates)
print(y_min, y_max)

pv_x = data["pv_system_x_coords"][batch_index, 0]
pv_y = data["pv_system_y_coords"][batch_index, 0]

##############
# get static map
##############

bottom_left = osgb_to_lat_lon(x=x_min, y=y_min)
top_right = osgb_to_lat_lon(x=x_max, y=y_max)
pv = osgb_to_lat_lon(x=pv_x, y=pv_y)

bottom_left = staticmaps.create_latlng(bottom_left[0], bottom_left[1])
top_right = staticmaps.create_latlng(top_right[0], top_right[1])

context = staticmaps.Context()

context.add_object(staticmaps.Marker(staticmaps.create_latlng(pv[0], pv[1]), size=5))

map = context.make_clean_map_from_bounding_box(
    width=640, height=640, bottom_left=bottom_left, top_right=top_right
)
map.save("map.png")
map = np.array(map)
# height, weight, 4
trace_map = go.Image(z=map)


##############
# Process satellite data
##############

# make rgb image sequence from satelight image
# want to use IR_016, VIS006 and VIS008 for rgb
# https://www.britannica.com/science/color/The-visible-spectrum,
# wavelengths
# red 650
# green 550
# blue 450
channel_indexes = [1, 9, 8]
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

satellite_rgb_data = np.expand_dims(satellite_rgb_data.mean(axis=3), axis=3).repeat(3, axis=3)

satellite_reszied_data = []
for i in range(0, satellite_rgb_data.shape[0]):
    # resize
    satellite_reszied_data.append(cv2.resize(satellite_rgb_data[i], (map.shape[1], map.shape[0])))
satellite_rgb_data = np.stack(satellite_reszied_data, axis=0).astype(np.uint8)


# pv yield
pv_yield = data["pv_yield"][batch_index][:, 0]

##############
# Plot one animation
##############

# overlay satellite images on top of static map
alpha = 0.5  # the amount of background map

# slider_steps = []
frames = []
for i in range(0, satellite_rgb_data.shape[0]):
    z = satellite_rgb_data[i] * alpha + map[:, :, 0:3] * (1 - alpha)
    frames.append(
        dict(
            data=go.Image(z=z),
            # traces=[0, 1],
            layout=go.Layout(title=str(sat_datetimes[i])),
        )
    )

# One plot
fig = go.Figure()
fig.add_trace(trace_map)
fig.update(frames=frames)


buttons = [
    dict(
        label="Play",
        method="animate",
        args=[
            None,
            dict(
                frame=dict(duration=300, redraw=True),  # MODIFY JUST THIS redraw to True
                transition={"duration": 50, "easing": "elastic"},
                easing="linear",
                fromcurrent=True,
                mode="immediate",
            ),
        ],
    ),
]
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=0,
            x=1.05,
            xanchor="left",
            yanchor="bottom",
            buttons=buttons,
        )
    ],
    width=800,
    height=500,
)


fig.update_yaxes(showticklabels=False)
fig.update_xaxes(showticklabels=False)
fig.show(renderer="browser")

##############
# Plot one animation, plus pv yeild on the side
##############

frames = []
for i in range(0, satellite_rgb_data.shape[0]):
    z = satellite_rgb_data[i] * alpha + map[:, :, 0:3] * (1 - alpha)
    frames.append(
        dict(
            data=[go.Image(z=z)],
            traces=[0],
            layout=go.Layout(title=str(sat_datetimes[i])),
            name=str(sat_datetimes[i]),
        )
    )

fig = make_subplots(
    rows=1, cols=2, subplot_titles=("Satellite", "PV Yield"), horizontal_spacing=0.051
)

# fig["data"] = [trace_map, go.Scatter(x=sat_datetimes, y=pv_yield)]
fig.add_trace(trace_map, row=1, col=1)
fig.add_trace(go.Scatter(x=sat_datetimes, y=pv_yield), row=1, col=2)
fig.update(frames=frames)


buttons = [
    dict(
        label="Play",
        method="animate",
        args=[
            None,
            dict(
                frame=dict(duration=300, redraw=True),  # MODIFY JUST THIS redraw to True
                transition={"duration": 50, "easing": "elastic"},
                easing="linear",
                fromcurrent=True,
                mode="immediate",
            ),
        ],
    ),
    dict(
        label="Pause",
        method="animate",
        args=[
            [None],
            dict(
                frame=dict(duration=0, redraw=True),  # MODIFY JUST THIS redraw to True
                transition={"duration": 0},
                mode="immediate",
            ),
        ],
    ),
]
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=0,
            x=1.05,
            xanchor="left",
            yanchor="bottom",
            buttons=buttons,
        )
    ],
    width=800,
    height=500,
    sliders=[
        {
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 500.0, "easing": "linear"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(sat_datetimes[i])],
                        {
                            "frame": {"duration": 500.0, "easing": "linear", "redraw": True},
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": str(sat_datetimes[i]),
                    "method": "animate",
                }
                for i in range(satellite_rgb_data.shape[0])
            ],
        }
    ],
)


fig.update_yaxes(showticklabels=False, row=1, col=1)
fig.update_xaxes(showticklabels=False, row=1, col=1)
fig.show(renderer="browser")
# plotly.offline.plot(fig, filename="filename_1.html", auto_open=False)
