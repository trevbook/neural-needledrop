"""
This file contains various utility functions related to logging with Google Cloud Logging.
"""

# =====
# SETUP
# =====
# The code below will set up the rest of the file.

# Import statements
import logging
from google.cloud import logging as cloud_logging

# Get the GBQ_PROJECT_ID
# TODO: I'll eventually have to grab this from an environment variable or something
GBQ_PROJECT_ID = "neural-needledrop"

# Define the level of logging we're interested in
LOGGING_LEVEL = logging.INFO

# Instantiate a Google Cloud Logging client
client = cloud_logging.Client(project=GBQ_PROJECT_ID)

# Retrieves a Cloud Logging handler based on the environment
client.setup_logging(
    log_level=LOGGING_LEVEL,
)


# Set up a particular logger
def get_logger(name, log_to_console=False):
    """
    This function sets up a logger with a particular name.
    """

    # Define the formatter string
    formatter_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get the logger
    logger = logging.getLogger(name)

    # Optionall,y add a console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOGGING_LEVEL)
        console_handler.setFormatter(logging.Formatter(formatter_string))
        logger.addHandler(console_handler)

    # Set up the formatter
    formatter = logging.Formatter(formatter_string)

    # Set the formatter
    logger.handlers[0].setFormatter(formatter)

    # Return the logger
    return logger
