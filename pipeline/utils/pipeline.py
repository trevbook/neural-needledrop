"""
This file contains utilities related to running the pipeline. 
"""

# =====
# SETUP
# =====
# The code below will help to set up the rest of this file

# General import statements
import traceback

# Importing custom modules
from utils.logging import get_logger

# Set up the logger for the pipeline
logger = get_logger(name="pipeline")

# ===============
# GENERAL METHODS
# ===============
# Below, I've defined some general methods that will help with the
# running of the pipeline.


def pipeline_job_wrapper(job_function, *args, **kwargs):
    """
    This method will wrap the pipeline job in a try/except block,
    which will allow the job to be run without crashing the entire
    pipeline.
    """

    # Try to run the job
    try:
        # Run the job
        job_function(*args, **kwargs)
    except Exception as e:
        # Log the error
        logger.error(
            f"Terminal error encountered in the function {job_function.__name__}: '{e}'\nTraceback is as follows:\n{traceback.format_exc()}"
        )

        # Pass, so that the pipeline can continue
        pass
