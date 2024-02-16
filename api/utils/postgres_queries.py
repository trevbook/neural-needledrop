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
import math
from pandas_gbq import read_gbq
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, declarative_base
from tqdm import tqdm
from pathlib import Path
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import random
import string
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

# Ensure you have the stopwords dataset downloaded
nltk.download("stopwords", quiet=True)

# Extract the stopwords
additional_stop_words = [
    "like"
]
stop_words = set(stopwords.words("english")) | set(additional_stop_words)

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


def generate_tsquery(input_query, max_n=3):
    # Tokenize the input string into words
    words = input_query.split()

    # Filter out stopwords
    filtered_words = [
        word for word in words if word.lower() not in stop_words
    ]

    # Generate n-grams for each n from 1 to max_n
    phrases = []
    for n in range(1, max_n + 1):
        for ngram in ngrams(filtered_words, n):
            phrase = " <-> ".join(ngram)  # Use <-> for phrase search in tsquery
            phrases.append(phrase)

    # Combine all phrases with the OR operator for tsquery
    tsquery = " | ".join(phrases)

    return tsquery


def generate_random_string(n):
    """
    This function generates a random n-digit string.
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


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

    # Query the database
    most_similar_embeddings_df = query_postgres(
        most_similar_emb_query, engine, logger=logger
    )

    # Return the dataframe
    return most_similar_embeddings_df


def most_similar_embeddings_by_url(embedding, urls, engine, n_per_url=5, logger=None):
    """
    This helper method is similar to the most_similar_embeddings method, but it will
    return the most similar embeddings for each URL in the provided list.
    """

    # Create a temporary table with the URLs
    pd.DataFrame([{"url": url} for url in urls]).to_sql(
        "temp_urls", engine, if_exists="replace", index=False
    )

    # Create the query to find the most similar embeddings for each URL
    most_similar_emb_query = f"""
    -- Create a CTE with all of the embeddings for the URLs
    WITH
    url_embeddings AS (
        SELECT
            embeddings.id,
            embeddings.url,
            embeddings.start_segment,
            embeddings.end_segment,
            embeddings.segment_length,
            embeddings_to_text.segment_start,
            embeddings_to_text.segment_end,
            embeddings_to_text.text,
            1 - (embeddings.embedding <=> '{embedding}') AS cos_sim
        FROM
            embeddings
        JOIN    
            temp_urls
        ON  
            embeddings.url = temp_urls.url
        JOIN
            embeddings_to_text
        ON
            embeddings_to_text.url = embeddings.url
            AND
            embeddings.id = embeddings_to_text.id
    ),
    ranked_url_embeddings AS (
        SELECT
            url_embeddings.*,
            ROW_NUMBER() OVER (PARTITION BY url_embeddings.url ORDER BY cos_sim DESC) AS rn
        FROM
            url_embeddings
    )
    SELECT
        ranked_url_embeddings.*
    FROM
        ranked_url_embeddings
    WHERE
        ranked_url_embeddings.rn <= {n_per_url}
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


def most_similar_embeddings_to_text_by_url(
    text, urls, engine, n_per_url=5, logger=None
):
    """
    This method will determine the most similar embeddings to the input `text` for the given `urls`.
    The `n_per_url` parameter determines how many similar embeddings to return for each url.
    """

    # Embed the text
    text_embedding = embed_text(text)

    # Use the most_similar_embeddings_by_url method to find the most similar embeddings
    most_similar_embeddings_df = most_similar_embeddings_by_url(
        text_embedding, urls, engine, n_per_url, logger
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

    # If this DataFrame is empty, return the input DataFrame as is
    if video_segments_to_fetch_df.empty:
        input_df["text"] = None
        return input_df

    # Create a temporary table in Postgres for the video segments to fetch
    from sqlalchemy.sql import text

    # Generate a name for the temporary table
    temp_table_name = (
        f"temp_text_segments_to_fetch_{generate_random_string(10)}".lower()
    )

    # Upload the DataFrame to Postgres
    with engine.connect() as conn:
        video_segments_to_fetch_df.to_sql(
            temp_table_name, conn, index=False, if_exists="replace"
        )
        conn.commit()

    # Now, we're going to query all of the text
    fetched_text_segments_df = query_postgres(
        f"""
        SELECT
        transcriptions.*,
        text_segments_to_fetch.embedding_id
        FROM
        transcriptions
        JOIN
        {temp_table_name} text_segments_to_fetch
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
        conn.execute(text(f"DROP TABLE IF EXISTS {temp_table_name}"))
        conn.commit()

    # Merge in the combined text segments
    input_df = input_df.merge(combined_text_segment_df, on="id", how="left")

    # Return the input DataFrame with the combined text segments
    return input_df


def create_filters(
    release_date_filter=None, video_type_filter=None, review_score_filter=None
):
    """
    This method will create a string of filters to be used in a SQL query.
    """
    filters = []

    if release_date_filter is not None:
        if release_date_filter[0] is None and release_date_filter[1] is not None:
            release_date_query = f"publish_date <= '{release_date_filter[1]}'"
        elif release_date_filter[0] is not None and release_date_filter[1] is not None:
            release_date_query = f"publish_date BETWEEN '{release_date_filter[0]}' AND '{release_date_filter[1]}'"
        elif release_date_filter[0] is not None and release_date_filter[1] is None:
            release_date_query = f"publish_date >= '{release_date_filter[0]}'"
        filters.append(release_date_query)

    if video_type_filter is not None:
        video_type_filter_string = (
            "(" + " OR ".join([f"video_type = '{x}'" for x in video_type_filter]) + ")"
        )
        filters.append(video_type_filter_string)

    if review_score_filter is not None:
        if review_score_filter[0] is None and review_score_filter[1] is not None:
            review_score_query = f"review_score <= {review_score_filter[1]}"
        elif review_score_filter[0] is not None and review_score_filter[1] is not None:
            review_score_query = f"review_score BETWEEN {review_score_filter[0]} AND {review_score_filter[1]}"
        elif review_score_filter[0] is not None and review_score_filter[1] is None:
            review_score_query = f"review_score >= {review_score_filter[0]}"
        filters.append(review_score_query)

    return filters


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

    # Create the filters
    filters = create_filters(
        release_date_filter, video_type_filter, review_score_filter
    )

    # If filters are provided, add them to the query
    if filters:
        most_similar_emb_query += " WHERE " + " AND ".join(filters)

    # Add the order and limit to the query
    most_similar_emb_query += f" LIMIT {n};"

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
        try:
            most_similar_embeddings_filtered_df = fetch_text_for_segments(
                most_similar_embeddings_filtered_df, engine
            )
        except Exception as e:
            print(f"RAN INTO AN ERROR ('{e}') WHILE FETCHING TEXT FOR SEGMENTS")
            most_similar_embeddings_filtered_df["text"] = None

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


def get_most_similar_transcriptions_filtered(
    query,
    engine,
    n_results=10,
    release_date_filter=None,
    video_type_filter=None,
    review_score_filter=None,
    include_text=False,
):
    """
    This is a helper method to retrieve the most similar
    transcriptions to a given text, with optional filters
    applied to the results.
    """
    
    # Convert the query into a tsquery
    tsquery = generate_tsquery(query)

    most_similar_transcription_segments_query = f"""
    SELECT
    video_metadata.title,
    transcriptions.url,
    {f"transcriptions.text," if include_text else ""}
    transcriptions.segment_id AS id,
    transcriptions.segment_seek,
    transcriptions.segment_start,
    transcriptions.segment_end,
    ts_rank_cd(ts_vec, to_tsquery('english', '{tsquery}')) AS rank
    FROM
    transcriptions
    LEFT JOIN
    video_metadata
    ON
    video_metadata.url = transcriptions.url
    WHERE
    ts_vec @@ to_tsquery('english', '{tsquery}')
    """
    
    print(f"The most_similar_transcription_segments_query is:\n\n{most_similar_transcription_segments_query}")

    # Create the filters
    filters = create_filters(
        release_date_filter, video_type_filter, review_score_filter
    )
    if filters:
        most_similar_transcription_segments_query += f" AND {' AND '.join(filters)}"

    # Add the ordering and limit clause
    most_similar_transcription_segments_query += f"""
    ORDER BY rank DESC
    """

    # If n_results is not None, add it to the query
    if n_results is not None:
        most_similar_transcription_segments_query += f" LIMIT {n_results}"

    # Execute the query
    most_similar_transcription_segments = query_postgres(
        most_similar_transcription_segments_query, engine
    )

    # Return the results
    return most_similar_transcription_segments


def get_most_similar_transcriptions(query, engine, n_results=10):
    """
    This is a helper method to retrieve the most similar
    transcriptions to a given text without applying any filters.
    """

    most_similar_transcription_segments_query = f"""
    SELECT
    transcriptions.url,
    transcriptions.segment_id AS id,
    transcriptions.segment_seek,
    transcriptions.segment_start,
    transcriptions.segment_end,
    ts_rank_cd(ts_vec, phraseto_tsquery('english', '{query}')) AS rank
    FROM
    transcriptions
    WHERE
    ts_vec @@ phraseto_tsquery('english', '{query}')
    ORDER BY rank DESC
    LIMIT {n_results}
    """

    # Execute the query
    most_similar_transcription_segments = query_postgres(
        most_similar_transcription_segments_query, engine
    )

    # Return the results
    return most_similar_transcription_segments


def get_most_similar_transcriptions_by_url(query, urls, engine, n_per_url=5):
    """
    This helper method is similar to the get_most_similar_transcriptions method, but it will
    return the most similar transcriptions for each URL in the provided list.
    """

    # Create a temporary table with the URLs
    pd.DataFrame([{"url": url} for url in urls]).to_sql(
        "temp_urls_transcriptions", engine, if_exists="replace", index=False
    )

    # Create the query to find the most similar transcriptions for each URL
    most_similar_trans_query = f"""
    -- Create a CTE with all of the transcriptions for the URLs
    WITH
    url_transcriptions AS (
        SELECT
            transcriptions.url,
            embeddings_to_text.id AS id,
            transcriptions.segment_seek,
            embeddings_to_text.segment_start AS segment_start,
            embeddings_to_text.segment_end AS segment_end,
            embeddings_to_text.text,
            ts_rank_cd(ts_vec, phraseto_tsquery('english', '{query}')) AS rank
        FROM
            transcriptions
        JOIN    
            temp_urls_transcriptions
        ON  
            transcriptions.url = temp_urls_transcriptions.url
        JOIN
            embeddings_to_text
        ON
            embeddings_to_text.url = transcriptions.url
            AND
            transcriptions.segment_id >= embeddings_to_text.start_segment
            AND
            transcriptions.segment_id < embeddings_to_text.end_segment
    ),
    ranked_url_transcriptions AS (
        SELECT
            url_transcriptions.*,
            ROW_NUMBER() OVER (PARTITION BY url_transcriptions.url ORDER BY rank DESC) AS rn
        FROM
            url_transcriptions
        WHERE
            rank > 0
    )
    SELECT
        ranked_url_transcriptions.*,
        NULL AS cos_sim
    FROM
        ranked_url_transcriptions
    WHERE
        ranked_url_transcriptions.rn <= {n_per_url}
    """

    # Query the database
    most_similar_transcriptions_df = query_postgres(most_similar_trans_query, engine)

    # Return the dataframe
    return most_similar_transcriptions_df


def get_most_similar_transcriptions_by_url_hybrid(
    text, urls, engine, n_per_url=5, k=60,
    neural_weight=0.8, keyword_weight=1
):
    """
    This method will determine the most similar transcriptions to the input `text` for the given `urls`.
    It'll use both neural embeddings and text search to find the most similar transcriptions.
    It'll combine the results from both methods with reciprocal rank fusion, and then return the top `n_per_url` results for each URL.
    """

    # Parameterize the meeting
    n_chunks_multiplier = 2

    # We're going to get the most similar chunks for both the neural and keyword searches in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        neural_similar_chunks_future = executor.submit(
            most_similar_embeddings_to_text_by_url,
            text=text,
            urls=urls,
            engine=engine,
            n_per_url=math.ceil(n_per_url * n_chunks_multiplier),
        )
        keyword_similar_chunks_future = executor.submit(
            get_most_similar_transcriptions_by_url,
            query=text,
            urls=urls,
            engine=engine,
            n_per_url=math.ceil(n_per_url * n_chunks_multiplier),
        )

    # Get the results
    neural_similar_chunks_df = neural_similar_chunks_future.result()
    keyword_similar_chunks_df = keyword_similar_chunks_future.result()

    # Modify the resulting DataFrames a bit
    neural_similar_chunks_df = neural_similar_chunks_df.rename(
        columns={"rn": "neural_rank"}
    )
    keyword_similar_chunks_df = keyword_similar_chunks_df.rename(
        columns={"rn": "keyword_rank"}
    )

    # Determine all of the unique (url, id, text) combinations
    unique_segment_chunks_df = (
        pd.concat(
            [
                neural_similar_chunks_df[["url", "id", "text"]],
                keyword_similar_chunks_df[["url", "id", "text"]],
            ]
        )
        .drop_duplicates()
        .copy()
    )

    # Merge in the ranks from both the neural and keyword searches
    unique_segment_chunks_df = unique_segment_chunks_df.merge(
        neural_similar_chunks_df[["url", "id", "neural_rank", "text", "cos_sim"]],
        on=["url", "id", "text"],
        how="left",
    ).merge(
        keyword_similar_chunks_df[["url", "id", "keyword_rank", "text", "rank"]],
        on=["url", "id", "text"],
        how="left",
    )

    # Add the "neural score" and "keyword score" columns
    unique_segment_chunks_df["neural_score"] = unique_segment_chunks_df[
        "neural_rank"
    ].apply(lambda x: 1 / (x + k))
    unique_segment_chunks_df["neural_score"] = unique_segment_chunks_df[
        "neural_score"
    ].fillna(0)

    unique_segment_chunks_df["keyword_score"] = unique_segment_chunks_df[
        "keyword_rank"
    ].apply(lambda x: 1 / (x + k))
    unique_segment_chunks_df["keyword_score"] = unique_segment_chunks_df[
        "keyword_score"
    ].fillna(0)

    # Now, add a "fused score" column
    unique_segment_chunks_df["fused_score"] = (
        unique_segment_chunks_df["neural_score"] * neural_weight
        + unique_segment_chunks_df["keyword_score"] * keyword_weight
    )

    # Add a fused rank column
    unique_segment_chunks_df["rn"] = unique_segment_chunks_df.groupby("url")[
        "fused_score"
    ].rank("dense", ascending=False)

    # Only take the top n_top_segment_chunks chunks per
    filtered_segment_chunks_df = (
        unique_segment_chunks_df.sort_values(
            ["fused_score", "keyword_score", "neural_score"],
            ascending=[False, False, False],
        )
        .groupby("url")
        .head(n_per_url)
    ).sort_values(
        ["url", "fused_score", "keyword_score", "neural_score"],
        ascending=[True, False, False, False],
    )

    # Merge back in the segment start and end times
    filtered_segment_chunks_df = filtered_segment_chunks_df.merge(
        keyword_similar_chunks_df[["url", "id", "segment_start", "segment_end"]],
        on=["url", "id"],
        how="left",
    ).merge(
        neural_similar_chunks_df[["url", "id", "segment_start", "segment_end"]],
        on=["url", "id"],
        suffixes=("_keyword", "_neural"),
        how="left",
    )

    # Take the minimum and maximum segment start and end times
    filtered_segment_chunks_df["segment_start"] = filtered_segment_chunks_df[
        ["segment_start_keyword", "segment_start_neural"]
    ].min(axis=1)
    filtered_segment_chunks_df["segment_end"] = filtered_segment_chunks_df[
        ["segment_end_keyword", "segment_end_neural"]
    ].max(axis=1)

    # Drop the unnecessary columns
    filtered_segment_chunks_df = filtered_segment_chunks_df.drop(
        columns=[
            "segment_start_keyword",
            "segment_end_keyword",
            "segment_start_neural",
            "segment_end_neural",
            "neural_rank",
            "keyword_rank",
            "neural_score",
            "keyword_score",
        ]
    )

    return filtered_segment_chunks_df
