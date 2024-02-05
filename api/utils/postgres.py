"""
This file contains various utility functions for interacting with the Postgres database.
"""

# =====
# SETUP
# =====
# The code below will set up the file.

# Import statements
import psycopg2
import pandas as pd
from sqlalchemy.sql import text
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Date,
    Float,
    Boolean,
    DateTime,
)

# Importing other custom modules
from utils.logging import get_dummy_logger

# =======
# METHODS
# =======
# Below are different methods for interacting with the Postgres database.


def delete_table(table_name, engine, logger=None):
    """
    Deletes a table from the database.
    """

    # If the logger is not provided, create a new one
    if logger is None:
        logger = get_dummy_logger()

    # We're going to try to delete the table. If we can't, we'll rollback the changes.

    with engine.connect() as conn:
        try:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()
            logger.debug(f"Successfully deleted table '{table_name}'")
        except Exception as e:
            logger.error(f"Error deleting table '{table_name}': {e}")
            conn.rollback()


def create_table(table_name, schema, engine, metadata, logger=None):
    """
    This function will create a particular table in our Postgres database. It'll use the
    schema provided by `schema`.
    """

    # If the logger is not provided, create a new one
    if logger is None:
        logger = get_dummy_logger()

    # Create the table
    try:
        # Define the table
        table = Table(table_name, metadata, *schema)

        # Create the table
        metadata.create_all(engine)

        # Log success
        logger.debug(f"Successfully created table '{table_name}'")

    except Exception as e:
        logger.error(f"Error creating table '{table_name}': {e}")
        raise e


def query_postgres(query, engine, logger=None):
    """
    This function will execute a particular query in our Postgres database.
    It will return a pandas DataFrame with the results.
    """

    # If the logger is not provided, create a new one
    if logger is None:
        logger = get_dummy_logger()

    # Execute the query
    with engine.connect() as conn:
        try:
            if query.upper().startswith(
                ("SET", "CREATE", "DROP", "INSERT", "UPDATE", "DELETE")
            ):
                # If it's a command that modifies the database, use the execute method
                conn.begin()
                conn.execute(text(query))
                conn.commit()
                logger.debug(f"Successfully executed command: {query}")
            else:
                # If it's a SELECT query, use the read_sql_query method
                result = pd.read_sql_query(query, conn)
                logger.debug(f"Successfully executed query: {query}")
                return result
        except Exception as e:
            logger.error(f"Error executing query '{query}': {e}")
            conn.rollback()
            raise e


def upload_to_table(data_frame, table, engine, logger=None):
    """
    This function will upload the contents of a pandas DataFrame to a particular table in our Postgres database.
    """

    # If the logger is not provided, create a new one
    if logger is None:
        logger = get_dummy_logger()

    # Upload the DataFrame
    try:
        data_frame.to_sql(table, engine, if_exists="append", index=False)
        logger.debug(f"Successfully uploaded data to table '{table}'")
    except Exception as e:
        logger.error(f"Error uploading data to table '{table}': {e}")
        raise e
