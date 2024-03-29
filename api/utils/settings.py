"""
This file contains a number of overall settings pertaining to the API. 
"""

# Import statements
import os

# Indicate what GBQ project we're using
GBQ_PROJECT_ID = os.getenv("GBQ_PROJECT_ID", "neural-needledrop")
GBQ_DATASET_ID = os.getenv("GBQ_DATASET_ID", "backend_data")

# Determine whether logging should print to console
LOG_TO_CONSOLE = True

# Determine whether the tqdm progress bar should be enabled
TQDM_ENABLED = os.getenv("TQDM_ENABLED") is not None

# Indicate some different settings for the Postgres database
POSTGRES_USER = os.getenv("POSTGRES_USER", "my_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "my_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "neural_needledrop_data")
