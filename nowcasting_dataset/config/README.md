# Configuration

Configuration for the data set.

Decided to go for a 'Pydantic' data class. It's slightly more complicated that just having yaml files, but the
'Pydantic' feature I think outweigh this. There is a load from yaml file also.

See `model.py` for documentation of the expected configuration fields.

See either `gcp.yaml` or `on_premises.yaml` for example config files.

All paths must include the protocol prefix.  For local files,
it's sufficient to just start with a '/'.  For aws, start with 's3://',
for gcp start with 'gs://'.


# Example

```python
# import the load function
from nowcasting_dataset.config.load import load_yaml_configuration

# load the configuration
confgiruation = load_yaml_configuration(filename)

# get the batch size
batch_size = configuration.process.batch_size
```
