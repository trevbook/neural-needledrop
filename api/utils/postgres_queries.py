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
    SELECT *, 1 - (embedding <=> '{embedding}') AS cos_sim
    FROM embeddings ORDER BY cos_sim LIMIT {n};
    """
    
    print(most_similar_emb_query)

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

    all_transcripts_df["video_id"] = all_transcripts_df["url"].apply(
        lambda x: x.split("=")[-1]
    )

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


def fetch_text_for_segments(input_df, engine):
    # Create a list of the video segments we need to fetch
    video_segments_to_fetch_df_records = []
    for row in input_df.itertuples():
        start_segment = row.start_segment
        end_segment = row.end_segment
        for cur_segment in range(start_segment, end_segment):
            video_segments_to_fetch_df_records.append(
                {
                    "video_url": row.url,
                    "embedding_id": row.id,
                    "video_segment": cur_segment,
                }
            )

    # Create a DataFrame from this list of records
    video_segments_to_fetch_df = pd.DataFrame(video_segments_to_fetch_df_records)

    # Create a temporary table in Postgres for the video segments to fetch
    from sqlalchemy.sql import text

    # Drop the table if it exists
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS text_segments_to_fetch"))
        conn.commit()

    # Upload the DataFrame to Postgres
    with engine.connect() as conn:
        video_segments_to_fetch_df.to_sql(
            "text_segments_to_fetch", conn, index=False, if_exists="replace"
        )
        conn.commit()

    # Now, we're going to query all of the text
    fetched_text_segments_df = query_postgres(
        """
        SELECT
        transcriptions.*,
        text_segments_to_fetch.embedding_id
        FROM
        transcriptions
        JOIN
        text_segments_to_fetch
        ON
        text_segments_to_fetch.video_url = transcriptions.url
        """,
        engine,
    )

    # Combine each of the text segments
    combined_text_segment_df_records = []
    for embedding_id in input_df["id"].unique():
        # Determine the start and end segments for this embedding
        embedding_df = input_df[input_df["id"] == embedding_id]
        start_segment = embedding_df["start_segment"].iloc[0]
        end_segment = embedding_df["end_segment"].iloc[0]

        # Filter the fetched text segments to only those for this embedding
        embedding_text_segments_df = fetched_text_segments_df[
            fetched_text_segments_df["embedding_id"] == embedding_id
        ]

        # Now, filter the text segments to only those that are within the start and end segments
        embedding_text_segments_df = embedding_text_segments_df[
            (embedding_text_segments_df["segment_id"] >= start_segment)
            & (embedding_text_segments_df["segment_id"] < end_segment)
        ]

        # Sort by the segment ID, and deduplicate the text segments
        embedding_text_segments_df = embedding_text_segments_df.sort_values(
            "segment_id"
        ).drop_duplicates("segment_id")

        # Now, combine the text segments into a single string
        combined_text = " ".join(
            [x.strip() for x in embedding_text_segments_df["text"].values]
        )

        # Add the combined text to the list of records
        combined_text_segment_df_records.append(
            {"id": embedding_id, "text": combined_text}
        )

    # Make a DataFrame from this list of records
    combined_text_segment_df = pd.DataFrame(combined_text_segment_df_records)

    # Now, delete the temporary table
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS text_segments_to_fetch"))
        conn.commit()

    # Merge in the combined text segments
    input_df = input_df.merge(combined_text_segment_df, on="id", how="left")

    # Return the input DataFrame with the combined text segments
    return input_df


def most_similar_embeddings_filtered(
    embedding,
    engine,
    n=10,
    nearest_neighbors_to_screen=10000,
    release_date_filter=None,
    video_type_filter=None,
    review_score_filter=None,
    include_text=False,
    logger=None,
):
    """
    This helper method will determine the most similar embeddings to the input `embedding`.
    The `n` parameter determines how many similar embeddings to return.
    The `release_date_filter`, `video_type_filter`, and `review_score_filter` parameters are used to filter the results.
    If the `include_text` parameter is set to True, then the text of the video will be included in the results.
    """

    # If nearest_neighbors_to_screen is None, we're not going to include it in the query
    if nearest_neighbors_to_screen is None:
        limit_clause = ""
    else:
        limit_clause = f"LIMIT {nearest_neighbors_to_screen}"

    # Create the base query to find the most similar embeddings
    most_similar_emb_query = f"""
    WITH 
      nearest_embeddings AS (
          SELECT
              embeddings.id
          FROM
                embeddings
          ORDER BY
                (embedding <=> '{embedding}')
          {limit_clause}
      )
    
    SELECT 
        embeddings.id,
        embeddings.url,
        embeddings.start_segment,
        embeddings.end_segment,
        1 - (embedding <=> '{embedding}') AS cos_sim
    FROM
    nearest_embeddings
    JOIN 
    embeddings
    ON
    nearest_embeddings.id = embeddings.id
    LEFT JOIN video_metadata
    ON embeddings.url = video_metadata.url
    """

    # If the release date filter is not None, then we need to add it to the query
    if release_date_filter is not None:
        if release_date_filter[0] is None and release_date_filter[1] is not None:
            # Filter for videos released before the second element of the list
            release_date_query = f"publish_date <= '{release_date_filter[1]}'"
        elif release_date_filter[0] is not None and release_date_filter[1] is not None:
            # Filter for videos released between the two elements of the list
            release_date_query = f"publish_date BETWEEN '{release_date_filter[0]}' AND '{release_date_filter[1]}'"
        elif release_date_filter[0] is not None and release_date_filter[1] is None:
            # Filter for videos released after the first element of the list
            release_date_query = f"publish_date >= '{release_date_filter[0]}'"

    # If the video_type_filter is not None, then we'll need to add it to the query.
    # This video_type_filter will be a list, and we need to filter to the `video_type` column
    if video_type_filter is not None:
        video_type_query = f"video_type IN {tuple(video_type_filter)}"

    # If the review_score_filter is not None, then we'll need to add it to the query.
    if review_score_filter is not None:
        if review_score_filter[0] is None and review_score_filter[1] is not None:
            # Filter for videos with review score at most the second element of the list
            review_score_query = f"review_score <= {review_score_filter[1]}"
        elif review_score_filter[0] is not None and review_score_filter[1] is not None:
            # Filter for videos with review score between the two elements of the list
            review_score_query = f"review_score BETWEEN {review_score_filter[0]} AND {review_score_filter[1]}"
        elif review_score_filter[0] is not None and review_score_filter[1] is None:
            # Filter for videos with review score at least the first element of the list
            review_score_query = f"review_score >= {review_score_filter[0]}"

    # Add the filters to the query
    filters = []
    if release_date_filter is not None:
        filters.append(release_date_query)
    if video_type_filter is not None:
        filters.append(video_type_query)
    if review_score_filter is not None:
        filters.append(review_score_query)

    if filters:
        most_similar_emb_query += " WHERE " + " AND ".join(filters)

    # Add the order and limit to the query
    most_similar_emb_query += f" LIMIT {n};"
    
    # Print the EXPLAIN of the query
    print("\nEXPLAIN of the query:\n")
    print(query_postgres(
        f"EXPLAIN {most_similar_emb_query}", engine, logger=logger
    ))

    # Query the database
    most_similar_embeddings_filtered_df = query_postgres(
        most_similar_emb_query, engine, logger=logger
    )

    # Order the resulting DataFrame by the cosine similarity
    most_similar_embeddings_filtered_df = (
        most_similar_embeddings_filtered_df.sort_values("cos_sim", ascending=False)
    )

    # If we want to include the text, then we'll need to query the database again
    if include_text:
        most_similar_embeddings_filtered_df = fetch_text_for_segments(
            most_similar_embeddings_filtered_df, engine
        )

    # Return the dataframe
    return most_similar_embeddings_filtered_df


def most_similar_embeddings_to_text_filtered(
    text,
    engine,
    n=5,
    nearest_neighbors_to_screen=10000,
    release_date_filter=None,
    video_type_filter=None,
    review_score_filter=None,
    model="text-embedding-3-small",
    include_text=False,
    logger=None,
):
    # Get the embedding of the text
    query_embedding = embed_text(
        text,
        model=model,
    )

    # Call the most_similar_embeddings method
    most_similar_embeddings_df = most_similar_embeddings_filtered(
        query_embedding,
        engine,
        n=n,
        release_date_filter=release_date_filter,
        video_type_filter=video_type_filter,
        review_score_filter=review_score_filter,
        include_text=include_text,
        nearest_neighbors_to_screen=nearest_neighbors_to_screen,
        logger=logger,
    )

    return most_similar_embeddings_df
