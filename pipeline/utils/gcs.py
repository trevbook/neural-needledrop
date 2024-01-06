"""
These utility functions are all for interacting with Google Cloud Storage.
They primarily use the `google-cloud-storage` package.
"""

# =====
# SETUP
# =====
# The code below will set up the file.

# General import statements
import pandas as pd
from pytubefix import YouTube, Channel
from google.cloud import bigquery
import traceback
import time
import random
from tqdm import tqdm
import pandas_gbq
import datetime
import uuid
from datetime import timedelta
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound

# Importing custom utility functions
import utils.gbq as gbq_utils
import utils.youtube as youtube_utils

# Indicate whether or not we want tqdm progress bars
tqdm_enabled = True

# Set some constants for the project
GBQ_PROJECT_ID = "neural-needledrop"
GBQ_DATASET_ID = "backend_data"

# Set the pandas_gbq context to the project ID
# pandas_gbq.context.project = GBQ_PROJECT_ID

# ===================
# GENERAL GCS METHODS
# ===================
# These are all general purpose GCS methods.


def delete_gcs_bucket(bucket_name, project_id, gcs_client=None, delete_if_exists=False):
    """
    This is a helper method for deleting a bucket.
    """
    if gcs_client is None:
        gcs_client = storage.Client(project=project_id)
    bucket = gcs_client.get_bucket(bucket_name)

    # Then delete the bucket
    bucket.delete(force=delete_if_exists)
    print(f"Bucket {project_id}.{bucket_name} deleted")


def create_bucket(bucket_name, project_id, gcs_client=None, delete_if_exists=False):
    """
    This is a helper method for creating a bucket.
    """
    if gcs_client is None:
        gcs_client = storage.Client(project=project_id)

    # Check if the bucket exists
    try:
        gcs_client.get_bucket(bucket_name)
        if delete_if_exists:
            delete_gcs_bucket(
                bucket_name, project_id, gcs_client, delete_if_exists=delete_if_exists
            )
        else:
            print(f"Bucket {project_id}.{bucket_name} already exists.")
            return
    except NotFound:
        pass

    # Create a new bucket
    bucket = gcs_client.create_bucket(bucket_name)
    print(f"Bucket {project_id}.{bucket_name} created")


def upload_file_to_bucket(file_path, bucket_name, project_id, gcs_client=None):
    """
    This is a helper method for uploading a file to a bucket.
    The file is defined by whatever is at `file_path`.
    """
    # We'll wrap this in a try/except block in case we run into an error
    try:
        if gcs_client is None:
            gcs_client = storage.Client(project=project_id)
        bucket = gcs_client.bucket(bucket_name)
        print(f"Found a bucket named {bucket_name} in project {project_id}.")
        
        # Extract the name of the file
        file_name = Path(file_path).name
        print(f"Uploading file {file_name} to {project_id}.{bucket_name}...")
        blob = bucket.blob(file_name)
        
        # Upload the file using the file path
        blob.upload_from_filename(file_path)
        
        print(f"File {file_path} uploaded to {project_id}.{bucket_name}")
    except Exception as e:
        print(f"Error uploading file {file_path} to {project_id}.{bucket_name}: {e}")
