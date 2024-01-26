"""
This file contains various utility functions related to logging with Google Cloud Logging.
"""

# =====
# SETUP
# =====
# The code below will set up the rest of the file.

# Import statements
import logging
import os
from google.cloud import logging as cloud_logging
from utils.settings import GBQ_PROJECT_ID

# Define the level of logging we're interested in based on the LOG_LEVEL environment variable
options = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOGGING_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Instantiate a Google Cloud Logging client
client = cloud_logging.Client(project=GBQ_PROJECT_ID)

# Retrieves a Cloud Logging handler based on the environment
client.setup_logging(log_level=LOGGING_LEVEL)


def get_logger(name, log_to_console=False):
    """
    This function sets up a logger with a particular name.
    If a logger with the same name already exists, it is deleted before creation.
    """

    # Define the formatter string
    formatter_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Check if a logger with the same name already exists
    if name in logging.root.manager.loggerDict:
        # Delete the existing logger to ensure a fresh setup
        del logging.root.manager.loggerDict[name]

    # Get the logger
    logger = logging.getLogger(name)

    # Ensure that at least one handler is attached
    if not logger.handlers:
        # If no handler is present, add a NullHandler to avoid 'No handlers could be found' warning
        logger.addHandler(logging.NullHandler())

    # Optionally, add a console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOGGING_LEVEL)
        logger.addHandler(console_handler)

    # Set up the formatter and apply it to all handlers
    formatter = logging.Formatter(formatter_string)
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger


def get_dummy_logger():
    """
    Create a dummy logger that does nothing with log messages.
    """
    logger = logging.getLogger("dummy")
    logger.addHandler(logging.NullHandler())
    return logger
