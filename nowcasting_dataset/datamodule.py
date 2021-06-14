import pytorch_lightning as pl
from nowcasting_dataset import data_sources
from nowcasting_dataset import time as nd_time


class NowcastingDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size

    def prepare_data(self):
        self.sat_data_source = data_sources.SatelliteDataSource()
        self.data_sources = [self.sat_data_source]

    def setup(self):
        """Split data, etc.

        ## Selecting daytime data.

        We're interested in forecasting solar power generation, so we
        don't care about nighttime data :)

        In the UK in summer, the sun rises first in the north east, and
        sets last in the north west [1].  In summer, the north gets more
        hours of sunshine per day.

        In the UK in winter, the sun rises first in the south east, and
        sets last in the south west [2].  In winter, the south gets more
        hours of sunshine per day.

        |                        | Summer | Winter |
        |           ---:         |  :---: |  :---: |
        | Sun rises first in     | N.E.   | S.E.   |
        | Sun sets last in       | N.W.   | S.W.   |
        | Most hours of sunlight | North  | South  |

        Before training, we select timesteps which have at least some
        sunlight.  We do this by computing the clearsky global horizontal
        irradiance (GHI) for the four corners of the satellite imagery,
        and for all the timesteps in the dataset.  We only use timesteps
        where the maximum global horizontal irradiance across all four
        corners is above some threshold.

        The 'clearsky solar irradiance' is the amount of sunlight we'd
        expect on a clear day at a specific time and location. The SI unit
        of irradiance is watt per square meter.  The 'global horizontal
        irradiance' (GHI) is the total sunlight that would hit a
        horizontal surface on the surface of the Earth.  The GHI is the
        sum of the direct irradiance (sunlight which takes a direct path
        from the Sun to the Earth's surface) and the diffuse horizontal
        irradiance (the sunlight scattered from the atmosphere).  For more
        info, see: https://en.wikipedia.org/wiki/Solar_irradiance

        References:
          1. [Video of June 2019](https://www.youtube.com/watch?v=IOp-tj-IJpk)
          2. [Video of Jan 2019](https://www.youtube.com/watch?v=CJ4prUVa2nQ)
        """
        available_timestamps = [
            data_source.available_timestamps()
            for data_source in self.data_sources]
        datetime_index = nd_time.intersection_of_datetimeindexes(
            available_timestamps)

        # TODO: Continue... Get boundaries from sat data.
        # Use that with nd_time.daylight to get daylight hours.
        # Then split into train and val date ranges.
