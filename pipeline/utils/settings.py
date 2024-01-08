"""
This file contains various settings for the pipeline. 
"""

# Import statements
import os

# Indicate what GBQ project we're using
GBQ_PROJECT_ID = os.getenv("GBQ_PROJECT_ID", "neural-needledrop")
GBQ_DATASET_ID = os.getenv("GBQ_DATASET_ID", "backend_data")

# Determine whether logging should print to console
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE") is not None

# Determine whether the tqdm progress bar should be enabled
TQDM_ENABLED = os.getenv("TQDM_ENABLED") is not None
