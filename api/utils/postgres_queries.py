"""
This file contains a number of methods that're used to fetch particular 
pieces of data from the Postgres database.
"""

# =====
# SETUP
# =====
# The code below will set up the rest of the file.

# General import statements
import pandas as pd
from pandas_gbq import read_gbq
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, declarative_base
from tqdm import tqdm
from pathlib import Path
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Importing custom modules
from utils.openai import embed_text, embed_text_list
from utils.settings import (
    GBQ_PROJECT_ID,
    GBQ_DATASET_ID,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    LOG_TO_CONSOLE,
)
from utils.logging import get_logger
from utils.postgres import query_postgres, upload_to_table

# Set up a logger for this notebook
logger = get_logger("postgres_notebook", log_to_console=LOG_TO_CONSOLE)

# =======
# METHODS
# =======
# The code below contains a number of methods that will be used to fetch data from the database.


def most_similar_embeddings(embedding, engine, n=10, logger=None):
    """
    This helper method will determine the most similar embeddings to the input `embedding`.
    The `n` parameter determines how many similar embeddings to return.
    """

    # Create the query to find the most similar embeddings
    most_similar_emb_query = f"""
    SELECT *, 1 - (embedding <-> '{embedding}') AS cos_sim
    FROM embeddings ORDER BY cos_sim DESC LIMIT {n};
    """

    # Query the database
    most_similar_embeddings_df = query_postgres(
        most_similar_emb_query, engine, logger=logger
    )

    # Return the dataframe
    return most_similar_embeddings_df


def most_similar_embeddings_to_text(text, engine, n=10, logger=None):
    """
    This method will determine the most similar embeddings to the input `text`.
    The `n` parameter determines how many similar embeddings to return.
    """

    # Embed the text
    text_embedding = embed_text(text)

    # Use the most_similar_embeddings method to find the most similar embeddings
    most_similar_embeddings_df = most_similar_embeddings(
        text_embedding, engine, n, logger
    )

    # Return the dataframe
    return most_similar_embeddings_df


def retrieve_video_transcript(video_id, engine, logger=None):
    """
    This method retrieves the transcript of a video from the database using the video's ID.
    """

    # Define the video URL
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Define the SQL query
    sql_query = f"""
        SELECT 
            transcription.*
        FROM transcriptions  transcription
        WHERE
            url = '{video_url}'
        ORDER BY 
            segment_id ASC
    """

    # Query the database
    video_transcript_df = query_postgres(sql_query, engine, logger=logger)

    # Return the dataframe
    return video_transcript_df


def retrieve_multiple_video_transcripts(video_ids, engine, logger=None):
    """
    This method retrieves the transcripts of multiple videos from the database using their IDs.
    """

    # Make all of the video IDs into URLs
    video_urls = [
        f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids
    ]

    # Create or replace a temporary table in Postgres for joining in video_urls
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS temp_video_urls"))
        conn.execute(text("CREATE TEMP TABLE temp_video_urls (url text)"))
        for url in video_urls:
            conn.execute(text(f"INSERT INTO temp_video_urls VALUES ('{url}')"))
        conn.commit()

    # Define the SQL query
    sql_query = f"""
        SELECT 
            transcription.*
        FROM transcriptions transcription
        INNER JOIN temp_video_urls
        ON transcription.url = temp_video_urls.url
        ORDER BY 
            segment_id ASC
    """

    # Query the database
    all_transcripts_df = query_postgres(sql_query, engine, logger=logger)

    all_transcripts_df['video_id'] = all_transcripts_df['url'].apply(lambda x: x.split('=')[-1])

    # Return the dataframe containing all transcripts
    return all_transcripts_df


def retrieve_video_metadata(video_id, engine, logger=None):
    """
    This method retrieves the metadata of a video from the database using the video's ID.
    """

    # Define the video URL
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Define the SQL query
    sql_query = f"""
        SELECT 
            metadata.*
        FROM video_metadata metadata
        WHERE
            url = '{video_url}'
    """

    # Query the database
    video_metadata_df = query_postgres(sql_query, engine, logger=logger)

    # Return the dataframe
    return video_metadata_df


def retrieve_multiple_video_metadata(video_ids, engine, logger=None):
    """
    This method retrieves the metadata of multiple videos from the database using their IDs.
    """

    # Create a temporary table in Postgres for joining in video_ids
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS temp_video_ids"))
        conn.execute(text("CREATE TEMP TABLE temp_video_ids (id text)"))
        for video_id in video_ids:
            conn.execute(text(f"INSERT INTO temp_video_ids VALUES ('{video_id}')"))
        conn.commit()

    # Define the SQL query
    sql_query = f"""
        SELECT 
            metadata.*
        FROM video_metadata metadata
        INNER JOIN temp_video_ids
        ON metadata.id = temp_video_ids.id
    """

    # Query the database
    video_metadata_df = query_postgres(sql_query, engine, logger=logger)

    # Return the dataframe
    return video_metadata_df
