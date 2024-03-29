"""
This job will initialize the cloud resources for the pipeline. 
This includes GBQ tables and GCS buckets. If the resources are 
already initialized, this job will do nothing.
"""

# =====
# SETUP
# =====
# The code below will set up the rest of this file

# Import statements
from google.cloud import bigquery, storage

# Importing custom utility functions
from utils.settings import GBQ_PROJECT_ID, GBQ_DATASET_ID, LOG_TO_CONSOLE
import utils.gbq as gbq_utils
import utils.gcs as gcs_utils
import utils.logging as logging_utils

# ========
# MAIN JOB
# ========
# Below, I'm going to define the function that will be called
# when this job is run.


def run_initialize_cloud_resources_job(
    delete_existing_tables=False,
    delete_existing_buckets=False,
    tables_to_create=[
        "video_metadata",
        "audio",
        "transcriptions",
        "enriched_video_metadata",
        "embeddings",
        "age_restricted",
    ],
    buckets_to_create=[
        "neural-needledrop-audio",
        "neural-needledrop-embeddings",
    ],
    gbq_client=None,
    gcs_client=None,
):
    """
    This method will generate all of the necessary cloud resources
    for the pipeline. You can alter which resources are created by
    modifying the `tables_to_create` and `buckets_to_create` arguments.
    If you want to delete the existing tables before creating the new
    ones, set `delete_existing_tables` to True.
    """

    # Set up a logger
    logger = logging_utils.get_logger(
        name="pipeline.initialize_resources", log_to_console=LOG_TO_CONSOLE
    )

    # Log that we're starting the job
    logger.info("Starting the INITIALIZE RESOURCES job.")

    # If the GBQ client isn't provided, create it
    if gbq_client is None:
        gbq_client = bigquery.Client(project=GBQ_PROJECT_ID)

    # If the GCS client isn't provided, create it
    if gcs_client is None:
        gcs_client = storage.Client(project=GBQ_PROJECT_ID)

    # Start by creating the GBQ dataset
    gbq_utils.create_dataset(
        project_id=GBQ_PROJECT_ID,
        dataset_id=GBQ_DATASET_ID,
        gbq_client=gbq_client,
        logger=logger,
    )

    # ==========
    # GBQ TABLES
    # ==========
    # We're going to iterate through each of the tables we want to create and
    # make sure that they exist. If they don't, we'll create them. If they do,
    # we'll skip them unless `delete_existing_tables` is set to True.

    for table_name in tables_to_create:
        # Run the table generation method from the gbq_utils module
        gbq_utils.run_table_generation_method(
            table_name=table_name,
            project_id=GBQ_PROJECT_ID,
            dataset_id=GBQ_DATASET_ID,
            gbq_client=gbq_client,
            delete_if_exists=delete_existing_tables,
            logger=logger,
        )

    # ===========
    # GCS BUCKETS
    # ===========
    # We're going to iterate through each of the buckets we want to create and
    # make sure that they exist. If they don't, we'll create them. If they do,
    # we'll skip them (unless `delete_existing_buckets` is set to True).

    # Iterate through the different buckets and make sure they exist
    for bucket_name in buckets_to_create:
        gcs_utils.create_bucket(
            bucket_name=bucket_name,
            project_id=GBQ_PROJECT_ID,
            gcs_client=gcs_client,
            delete_if_exists=delete_existing_buckets,
            logger=logger,
        )

    # Log success
    logger.info("Finished the INITIALIZE RESOURCES job.")
