"""
This file contains some functions that're related to searching for particular videos. 
"""

# =====
# SETUP
# =====
# The code below contains some imports that're necessary for the functions in this file to work.

# General import statements
import pandas as pd
import datetime
from time import time

# Importing custom modules
from utils.settings import (
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
)
import utils.postgres_queries as pg_queries
import utils.postgres as postgres
from sqlalchemy import create_engine

# Create the connection string to the database
postgres_connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create the connection engine
engine = create_engine(postgres_connection_string)

# =======
# METHODS
# =======
# The code below contains the functions that're used to search for particular videos.


def neural_search(
    query,
    release_date_filter=None,
    video_type_filter=None,
    review_score_filter=None,
    n_chunks_to_consider_initially=1500,
    n_most_similar_chunks_per_video=5,
    n_videos_to_return=10,
    n_segment_chunks_to_showcase=3,
    min_similar_chunks=2,
    nearest_neighbors_to_screen=5000,
    ivfflat_probes=50,
):
    """
    This method will run some neural search on the database. It will return a list of videos that are most similar to the query,
    with some of the most similar text chunks from those videos.
    """

    # First, we're going to set the probes
    postgres.query_postgres(f"SET ivfflat.probes = {ivfflat_probes}", engine=engine)
    postgres.query_postgres(
        f"SET hnsw.efSearch = {nearest_neighbors_to_screen}", engine=engine
    )

    # Run the query
    start_time = time()
    similar_chunks_df = pg_queries.most_similar_embeddings_to_text_filtered(
        text=query,
        engine=engine,
        n=n_chunks_to_consider_initially,
        nearest_neighbors_to_screen=nearest_neighbors_to_screen,
        release_date_filter=release_date_filter,
        video_type_filter=video_type_filter,
        review_score_filter=review_score_filter,
        include_text=False,
    )
    total_time = time() - start_time
    print(f"Total time to get similar chunks: {total_time} seconds.")

    # Groupby `url`, and take the top `n_most_similar_chunks_per_video` chunks per video
    start_time = time()
    aggregated_similar_chunks_df = similar_chunks_df.groupby("url").head(
        n_most_similar_chunks_per_video
    )

    # Aggregate the similarity statistics
    aggregated_similar_chunks_df = (
        aggregated_similar_chunks_df.groupby("url")
        .agg(
            median_similarity=("cos_sim", "median"),
            average_similarity=("cos_sim", "mean"),
            n_similar_chunks=("cos_sim", "count"),
            max_similarity=("cos_sim", "max"),
        )
        .reset_index()
    )

    # Add a z-score column for the maximum similarity
    aggregated_similar_chunks_df["cos_sim_z_score"] = (
        aggregated_similar_chunks_df["max_similarity"]
        - aggregated_similar_chunks_df["max_similarity"].mean()
    ) / aggregated_similar_chunks_df["max_similarity"].std()

    # Add a weighted z-score median similarity column
    aggregated_similar_chunks_df["weighted_z_score"] = (
        aggregated_similar_chunks_df["cos_sim_z_score"]
        * aggregated_similar_chunks_df["n_similar_chunks"]
    )

    # Add a weighted median similarity column
    aggregated_similar_chunks_df["weighted_median_similarity"] = (
        aggregated_similar_chunks_df["median_similarity"]
        * aggregated_similar_chunks_df["n_similar_chunks"]
    )

    # Sort by the weighted median similarity
    aggregated_similar_chunks_df = (
        aggregated_similar_chunks_df.sort_values("weighted_z_score", ascending=False)
        .query("n_similar_chunks >= @min_similar_chunks")
        .head(n_videos_to_return)
    )
    total_time = time() - start_time
    print(f"Total time to aggregate and sort: {total_time} seconds.")

    # Create a temporary table called `temp_similar_chunks` that is the aggregated_similar_chunks_df DataFrame
    start_time = time()
    with engine.connect() as conn:
        aggregated_similar_chunks_df.to_sql(
            "temp_similar_chunks", conn, if_exists="replace", index=False
        )

    # Now, select the entire `video_metadata` table for each of the videos in the `temp_similar_chunks` table
    similar_chunks_video_metadata_df = postgres.query_postgres(
        """
        SELECT 
            video_metadata.*, 
            temp_similar_chunks.median_similarity, 
            temp_similar_chunks.n_similar_chunks, 
            temp_similar_chunks.weighted_median_similarity,
            temp_similar_chunks.weighted_z_score
        FROM video_metadata
        JOIN temp_similar_chunks
        ON video_metadata.url = temp_similar_chunks.url
        ORDER BY temp_similar_chunks.weighted_median_similarity DESC
        """,
        engine=engine,
    )

    # Create a DataFrame containing the segment chunks I want to showcase
    segment_chunks_to_showcase_df = (
        (
            pg_queries.fetch_text_for_segments(
                similar_chunks_df[
                    similar_chunks_df["url"].isin(
                        similar_chunks_video_metadata_df["url"].unique()
                    )
                ],
                engine,
            )
            .sort_values("cos_sim", ascending=False)
            .groupby("url")
            .head(n_segment_chunks_to_showcase)
            .sort_values(["url", "cos_sim"], ascending=False)
        )
        .groupby("url")
        .agg(
            top_segment_chunks=("text", lambda x: list(x)),
        )
        .reset_index()
    )

    # Merge this DataFrame with the video metadata
    segment_chunks_to_showcase_df = segment_chunks_to_showcase_df.merge(
        similar_chunks_video_metadata_df, on="url"
    ).sort_values("weighted_z_score", ascending=False)
    total_time = time() - start_time
    print(f"Total time to get video metadata and text: {total_time} seconds.")

    # Return the DataFrame as a list of dictionaries
    return segment_chunks_to_showcase_df.to_json(orient="records")
