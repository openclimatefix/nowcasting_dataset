""" Manager util functions """
import logging

logger = logging.getLogger(__name__)


def callback(result, data_source_name, split_name):
    """Create callback for 'pool.apply_async'"""
    logger.info(f"{data_source_name} has finished created batches for {split_name}!")


def error_callback(exception, an_error_has_occured, data_source_name, split_name):
    """Create error callback for 'pool.apply_async'

    Need to pass in data_source_name rather than rely on data_source_name
    in the outer scope, because otherwise the error message will contain
    the wrong data_source_name (due to stuff happening concurrently!)
    """
    logger.exception(
        f"Exception raised by {data_source_name} whilst creating batches for"
        f" {split_name}\n{exception.__class__.__name__}: {exception}"
    )
    an_error_has_occured.set()
