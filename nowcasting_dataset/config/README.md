# Configuration

Configuration for the data set.


Decided to go for a 'Pydantic' data class. It's slightly more complicated that just having yaml files, but the
'Pydantic' feature I think outweigh this. There is a load from yaml file also.

# Example

```
# import the load function
from nowcasting_dataset.config.load import load_yaml_configuration

# load the configuration
confgiruation = load_yaml_configuration(filename)

# get the batch size
batch_size = confgiruation.process.batch_size
```
