"""
This file contains some utility functions for interacting with Google BigQuery.
"""
# =====
# SETUP
# =====
# The code below will set up the file.

# Import packages
from google.cloud import bigquery

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


def create_dataset(project_id, dataset_id, gbq_client=None, delete_if_exists=False):
    """
    Create a dataset in BigQuery.

    Args:
        project_id (str): The project ID for the project where the dataset resides.
        dataset_id (str): The dataset ID for the dataset to be created.
        gbq_client (google.cloud.bigquery.client.Client): A BigQuery client object. If None, a new client will be created.
        delete_if_exists (bool): If True, delete the dataset if it already exists.
    """
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)
    dataset_ref = gbq_client.dataset(dataset_id)
    if delete_if_exists:
        try:
            delete_dataset(project_id, dataset_id)
        except Exception:
            pass
    dataset = bigquery.Dataset(dataset_ref)
    try:
        dataset = gbq_client.create_dataset(dataset)
        print("Created dataset {}.".format(dataset_id))
    except Exception:
        print("Dataset {} already exists.".format(dataset_id))


def delete_table(project_id, dataset_id, table_id, gbq_client=None):
    """
    Delete a table from BigQuery.

    Args:
        project_id (str): The project ID for the project where the table resides.
        dataset_id (str): The dataset ID for the dataset where the table resides.
        table_id (str): The table ID for the table to be deleted.
        gbq_client (google.cloud.bigquery.client.Client): A BigQuery client object. If None, a new client will be created.
    """
    if gbq_client is None:
        gbq_client = bigquery.Client(project=project_id)
    table_ref = gbq_client.dataset(dataset_id).table(table_id)
    gbq_client.delete_table(table_ref)
    print("Table {}:{} deleted.".format(dataset_id, table_id))


def create_table(
    project_id, dataset_id, table_id, schema, gbq_client=None, delete_if_exists=False
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
    """
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
        print("Created table {}:{}.".format(dataset_id, table_id))
    except Exception:
        print("Table {}:{} already exists.".format(dataset_id, table_id))


# ======================
# TABLE-SPECIFIC METHODS
# ======================
# These are some methods for working with specific tables.


def generate_video_metadata_table(
    project_id="neural-needledrop",
    dataset_id="backend_data",
    gbq_client=None,
    delete_if_exists=False,
):
    """
    This helper theme will initialize the `video_metadata` table in the `backend_data` dataset, if it doesn't already exist.
    """

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
        bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("view_ct", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("thumbnail_url", "STRING", mode="NULLABLE"),
    ]

    # Define the table_id
    create_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id="video_metadata",
        schema=schema,
        gbq_client=gbq_client,
        delete_if_exists=delete_if_exists,
    )
