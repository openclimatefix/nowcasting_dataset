"""Configure PyTest"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--use_cloud_data", action="store_true", default=False,
        help="Use large datasets on GCP instead of local test datasets.")


@pytest.fixture
def use_cloud_data(request):
    return request.config.getoption("--use_cloud_data")
