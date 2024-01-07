"""
These utility functions are all for interacting with Google Cloud Storage.
They primarily use the `google-cloud-storage` package.
"""

# =====
# SETUP
# =====
# The code below will set up the file.

# General import statements
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound

# Importing custom utility functions
from utils.logging import get_dummy_logger

# ===================
# GENERAL GCS METHODS
# ===================
# These are all general purpose GCS methods.


def delete_gcs_bucket(
    bucket_name, project_id, gcs_client=None, delete_if_exists=False, logger=None
):
    """
    This is a helper method for deleting a bucket.
    """
    logger = logger or get_dummy_logger()
    if gcs_client is None:
        gcs_client = storage.Client(project=project_id)
    bucket = gcs_client.get_bucket(bucket_name)

    # Then delete the bucket
    bucket.delete(force=delete_if_exists)
    logger.debug(f"Bucket {project_id}.{bucket_name} deleted")


def create_bucket(
    bucket_name, project_id, gcs_client=None, delete_if_exists=False, logger=None
):
    """
    This is a helper method for creating a bucket.
    """
    logger = logger or get_dummy_logger()

    if gcs_client is None:
        gcs_client = storage.Client(project=project_id)

    # Check if the bucket exists
    try:
        gcs_client.get_bucket(bucket_name)
        if delete_if_exists:
            delete_gcs_bucket(
                bucket_name,
                project_id,
                gcs_client,
                delete_if_exists=delete_if_exists,
                logger=logger,
            )
        else:
            logger.debug(f"Bucket {project_id}.{bucket_name} already exists")
            return
    except NotFound:
        pass

    # Create a new bucket
    gcs_client.create_bucket(bucket_name)
    logger.debug(f"Bucket {project_id}.{bucket_name} created")


def upload_file_to_bucket(
    file_path, bucket_name, project_id, gcs_client=None, logger=None
):
    """
    This is a helper method for uploading a file to a bucket.
    The file is defined by whatever is at `file_path`.
    """
    logger = logger or get_dummy_logger()
    # We'll wrap this in a try/except block in case we run into an error
    try:
        if gcs_client is None:
            gcs_client = storage.Client(project=project_id)
        bucket = gcs_client.bucket(bucket_name)

        # Extract the name of the file
        file_name = Path(file_path).name
        blob = bucket.blob(file_name)

        # Upload the file using the file path
        blob.upload_from_filename(file_path)
        logger.debug(f"File {file_name} uploaded to {project_id}.{bucket_name}")
    except Exception as e:
        logger.error(
            f"Error uploading file {file_name} to {project_id}.{bucket_name}: '{e}'"
        )


def download_file_from_bucket(
    bucket_name, file_name, destination_folder, project_id, gcs_client=None, logger=None
):
    """
    This is a helper method for downloading a file from a bucket.
    The file is defined by whatever is at `file_path`.
    """
    logger = logger or get_dummy_logger()

    # We'll wrap this in a try/except block in case we run into an error
    try:
        # Check to see if the destination folder exists
        if not Path(destination_folder).exists():
            Path(destination_folder).mkdir(parents=True, exist_ok=True)

        if gcs_client is None:
            gcs_client = storage.Client(project=project_id)
        bucket = gcs_client.bucket(bucket_name)

        # Download the file using the file path
        blob = bucket.blob(file_name)
        blob.download_to_filename(destination_folder + file_name)
        logger.debug(f"File {file_name} downloaded from {project_id}.{bucket_name}")
    except Exception as e:
        logger.error(
            f"Error downloading file {file_name} from {project_id}.{bucket_name}: '{e}'"
        )
