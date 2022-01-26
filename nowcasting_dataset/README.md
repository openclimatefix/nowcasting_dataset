# nowcasting_dataset

This main dir contains the following dirs and files

## Dirs

### config

Defined, load and save configurations. Also stores two example configurations.

#### data_ources

Functions on how to load and manipulate each different data source

### dataset

Functions and methods for a 'batch' which is a collection of examples from the different data sources

### filesystems

Utility functions on how to access local and cloud files

## Files

### consts.py

Constant strings and variables that are needed throughout the repo

### geospatial.py

Geospatial functions that are used to transform coordinates. For example latitude and longitude to OSGB.

### square.py

Methods to mask things in bounding boxes ( or squares)
.
### time.py

Time utility functions for time related functions e.g night time filtering

### utils.py

General util functions (TODO #170 probably needs some tidying up)
