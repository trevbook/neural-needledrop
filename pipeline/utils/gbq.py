"""
This file contains some utility functions for interacting with Google BigQuery.
"""
# =====
# SETUP
# =====
# The code below will set up the file.

# Import packages
import uuid
from google.cloud import bigquery

# Import custom utils
from utils.logging import get_dummy_logger

# ===============
# GENERAL METHODS
# ===============
# These are some general purpose GBQ methods.


def delete_dataset(project_id, dataset_id, gbq_client=None):
    """
    Delete a dataset from BigQuery.

    Args:
        project_id (str): The project ID for the project where the dataset resides.
        dataset_id (str): The dataset ID for the dataset to be deleted.
        gbq_client (google.cloud.bigquery.client.Client): A BigQuery client object. If None, a new client will be created.
    """
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)
    dataset_ref = gbq_client.dataset(dataset_id)
    gbq_client.delete_dataset(dataset_ref)
    print("Dataset {} deleted.".format(dataset_id))


def create_dataset(
    project_id, dataset_id, gbq_client=None, delete_if_exists=False, logger=None
):
    """
    Create a dataset in BigQuery.

    Args:
        project_id (str): The project ID for the project where the dataset resides.
        dataset_id (str): The dataset ID for the dataset to be created.
        gbq_client (google.cloud.bigquery.client.Client): A BigQuery client object. If None, a new client will be created.
        delete_if_exists (bool): If True, delete the dataset if it already exists.
        logger (logging.Logger): A logger object. If None, no logging will be performed.
    """
    logger = logger or get_dummy_logger()
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)
    dataset_ref = gbq_client.dataset(dataset_id)
    if delete_if_exists:
        try:
            delete_dataset(project_id, dataset_id)
            logger.debug(f"Deleted dataset {project_id}.{dataset_id}")
        except Exception:
            pass
    dataset = bigquery.Dataset(dataset_ref)
    try:
        dataset = gbq_client.create_dataset(dataset)
        logger.debug(f"Created dataset {project_id}.{dataset_id}")
    except Exception:
        logger.debug(f"Dataset {project_id}.{dataset_id} already exists.")


def delete_table(project_id, dataset_id, table_id, gbq_client=None):
    """
    Delete a table from BigQuery.

    Args:
        project_id (str): The project ID for the project where the table resides.
        dataset_id (str): The dataset ID for the dataset where the table resides.
        table_id (str): The table ID for the table to be deleted.
        gbq_client (google.cloud.bigquery.client.Client): A BigQuery client object. If None, a new client will be created.
    """
    try:
        if gbq_client is None:
            gbq_client = bigquery.Client(project=project_id)
        table_ref = gbq_client.dataset(dataset_id).table(table_id)
        gbq_client.delete_table(table_ref)
        print("Table {}:{} deleted.".format(dataset_id, table_id))
    except Exception:
        print("Table {}:{} does not exist.".format(dataset_id, table_id))


def create_table(
    project_id, dataset_id, table_id, schema, gbq_client=None, delete_if_exists=False,
    logger=None
):
    """
    Create a table in BigQuery.

    Args:
        project_id (str): The project ID for the project where the table resides.
        dataset_id (str): The dataset ID for the dataset where the table resides.
        table_id (str): The table ID for the table to be created.
        schema (list): A list of dictionaries containing the schema for the table.
        gbq_client (google.cloud.bigquery.client.Client): A BigQuery client object. If None, a new client will be created.
        delete_if_exists (bool): If True, delete the table if it already exists.
        logger (logging.Logger): A logger object. If None, no logging will be performed.
    """
    logger = logger or get_dummy_logger()
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)
    dataset_ref = gbq_client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    if delete_if_exists:
        try:
            delete_table(project_id, dataset_id, table_id)
        except Exception:
            pass
    table = bigquery.Table(table_ref, schema=schema)
    try:
        table = gbq_client.create_table(table)
        logger.debug(f"Created table {project_id}.{dataset_id}.{table_id}")
    except Exception:
        logger.debug(f"Table {project_id}.{dataset_id}.{table_id} already exists.")


def add_rows_to_table(project_id, dataset_id, table_id, rows, gbq_client=None):
    """
    Add rows to a table in BigQuery.

    Args:
        project_id (str): The project ID for the project where the table resides.
        dataset_id (str): The dataset ID for the dataset where the table resides.
        table_id (str): The table ID for the table to be created.
        rows (list): A list of dictionaries containing the rows to be added to the table.
        gbq_client (google.cloud.bigquery.client.Client): A BigQuery client object. If None, a new client will be created.
    """
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)
    dataset_ref = gbq_client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    table = gbq_client.get_table(table_ref)
    errors = gbq_client.insert_rows(table, rows)
    if errors == []:
        print("Loaded {} row(s) into {}:{}.".format(len(rows), dataset_id, table_id))
    else:
        print("Errors: {}".format(errors))


def create_temporary_table_in_gbq(
    dataframe,
    project_id,
    dataset_name,
    table_name=None,
    if_exists="fail",
    gbq_client=None,
):
    # If the GBQ client is not provided, then we're going to create a new one
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)

    # Generate a random table name if not provided
    if not table_name:
        table_name = str(uuid.uuid4()).replace("-", "_")

    full_table_name = f"{dataset_name}.{table_name}"

    # Use pandas-gbq to upload the DataFrame
    dataframe.to_gbq(
        destination_table=full_table_name,
        project_id=project_id,
        if_exists=if_exists,
    )

    # Return the full table name
    return f"{project_id}.{full_table_name}"


# ======================
# TABLE-SPECIFIC METHODS
# ======================
# These are some methods for working with specific tables.


def generate_video_metadata_table(
    project_id="neural-needledrop",
    dataset_id="backend_data",
    gbq_client=None,
    delete_if_exists=False,
    logger=None,
):
    """
    This helper theme will initialize the `video_metadata` table in the `backend_data` dataset, if it doesn't already exist.
    """
    
    # Set up the logger
    logger = logger or get_dummy_logger()

    # If the gbq_client is not provided, create one
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)

    # Define the table schema
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("length", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("channel_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("channel_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("short_description", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("view_ct", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("small_thumbnail_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("large_thumbnail_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("publish_date", "DATETIME", mode="NULLABLE"),
        bigquery.SchemaField("scrape_date", "DATETIME", mode="REQUIRED"),
    ]

    # Define the table_id
    create_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id="video_metadata",
        schema=schema,
        gbq_client=gbq_client,
        delete_if_exists=delete_if_exists,
        logger=logger
    )


def generate_audio_table(
    project_id="neural-needledrop",
    dataset_id="backend_data",
    gbq_client=None,
    delete_if_exists=False,
    logger=None,
):
    """
    This helper method will initialize the `audio` table in the `backend_data` dataset, if it doesn't already exist.
    """
    
    # Set up the logger
    logger = logger or get_dummy_logger()

    # If the gbq_client is not provided, create one
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)

    # Define the table schema
    schema = [
        bigquery.SchemaField("video_url", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("audio_gcr_uri", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("scrape_date", "DATETIME", mode="REQUIRED"),
    ]

    # Define the table_id
    create_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id="audio",
        schema=schema,
        gbq_client=gbq_client,
        delete_if_exists=delete_if_exists,
        logger=logger
    )


def generate_transcriptions_table(
    project_id="neural-needledrop",
    dataset_id="backend_data",
    gbq_client=None,
    delete_if_exists=False,
    logger=None,
):
    """
    This method will initialize the `transcriptions` table in the `backend_data` dataset, if it doesn't already exist.
    """
    
    # Set up the logger
    logger = logger or get_dummy_logger()

    # If the gbq_client is not provided, create one
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)

    # Define the table schema
    schema = [
        bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("language", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "DATETIME", mode="REQUIRED"),
        bigquery.SchemaField("segment_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("segment_id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("segment_seek", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("segment_start", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("segment_end", "FLOAT", mode="REQUIRED"),
    ]

    # Create the table
    create_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id="transcriptions",
        schema=schema,
        gbq_client=gbq_client,
        delete_if_exists=delete_if_exists,
        logger=logger
    )


def generate_enriched_video_metadata_table(
    project_id="neural-needledrop",
    dataset_id="backend_data",
    gbq_client=None,
    delete_if_exists=False,
    logger=None,
):
    """
    This method will initialize the `enriched_video_metadata` table in the `backend_data` dataset, if it doesn't already exist.
    """
    
    # Set up the logger
    logger = logger or get_dummy_logger()

    # If the gbq_client is not provided, create one
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)

    # Define the table schema
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("video_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("review_score", "INTEGER", mode="NULLABLE"),
    ]

    # Create the table
    create_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id="enriched_video_metadata",
        schema=schema,
        gbq_client=gbq_client,
        delete_if_exists=delete_if_exists,
        logger=logger
    )


def generate_embeddings_table(
    project_id="neural-needledrop",
    dataset_id="backend_data",
    gbq_client=None,
    delete_if_exists=False,
    logger=None,
):
    """
    This method will initialize the `embeddings` table in the `backend_data` dataset, if it doesn't already exist.
    """
    
    # Set up the logger
    logger = logger or get_dummy_logger()

    # If the gbq_client is not provided, create one
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)

    # Define the table schema
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("video_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("gcs_uri", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("embedding_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("start_segment", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("end_segment", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("segment_length", "INTEGER", mode="REQUIRED"),
    ]

    # Create the table
    create_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id="embeddings",
        schema=schema,
        gbq_client=gbq_client,
        delete_if_exists=delete_if_exists,
        logger=logger
    )


def run_table_generation_method(
    table_name,
    project_id,
    dataset_id,
    gbq_client=None,
    delete_if_exists=False,
    logger=None,
):
    """
    This method will run the appropriate table generation method based on the provided table name.
    """
    # If the gbq_client is not provided, create one
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)

    # If the logger is not provided, create one
    if logger is None:
        logger = get_dummy_logger()
    
    logger.debug(f"Running table generation method for {table_name}")

    # Define a mapping between table names and table generation methods
    table_name_to_generation_method = {
        "video_metadata": generate_video_metadata_table,
        "audio": generate_audio_table,
        "transcriptions": generate_transcriptions_table,
        "enriched_video_metadata": generate_enriched_video_metadata_table,
        "embeddings": generate_embeddings_table,
    }

    # Get the appropriate table generation method
    table_generation_method = table_name_to_generation_method[table_name]

    # Run the table generation method
    table_generation_method(
        project_id=project_id,
        dataset_id=dataset_id,
        gbq_client=gbq_client,
        delete_if_exists=delete_if_exists,
        logger=logger
    )
