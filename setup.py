from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nowcasting_dataset",
    version="0.1.2",
    license="MIT",
    description="Nowcasting Dataset",
    author="Jack Kelly, Peter Dudfield, Jacob Bieker",
    author_email="info@openclimatefix.org",
    company="Open Climate Fix Ltd",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "zarr",
        "xarray",
        "ipykernel",
        "h5netcdf",
        "gcsfs",
        "scikit-image",
        "torch",
        "pytorch-lightning",
        "dask",
        "pvlib",
        "pyproj",
        "flake8",
        "jedi",
        "mypy",
        "tables",
        "boto3",
        "moto",
        "neptune-client",
        "pydantic",
        "pytest-cov",
        "plotly",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
)
