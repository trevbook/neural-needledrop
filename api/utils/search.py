"""
This file contains some functions that're related to searching for particular videos. 
"""

# =====
# SETUP
# =====
# The code below contains some imports that're necessary for the functions in this file to work.

# General import statements
from time import time
import json
from concurrent.futures import ThreadPoolExecutor

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

# General import statements
import pandas as pd
from sqlalchemy import create_engine
from utils.postgres_queries import (
    get_most_similar_transcriptions_filtered,
)

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
    # print(f"Total time to get similar chunks: {total_time} seconds.")

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

    # Add a z-score column for the number of similar chunks
    aggregated_similar_chunks_df["n_similar_chunks_z_score"] = (
        aggregated_similar_chunks_df["n_similar_chunks"]
        - aggregated_similar_chunks_df["n_similar_chunks"].mean()
    ) / aggregated_similar_chunks_df["n_similar_chunks"].std()

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
    # print(f"Total time to aggregate and sort: {total_time} seconds.")

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
            temp_similar_chunks.n_similar_chunks AS n_segment_matches, 
            temp_similar_chunks.weighted_z_score AS neural_search_score
        FROM video_metadata
        JOIN temp_similar_chunks
        ON video_metadata.url = temp_similar_chunks.url
        ORDER BY temp_similar_chunks.weighted_median_similarity DESC
        """,
        engine=engine,
    )

    # Add a column indicating the z-score of the neural search score
    similar_chunks_video_metadata_df["neural_search_score_z_score"] = (
        similar_chunks_video_metadata_df["neural_search_score"]
        - similar_chunks_video_metadata_df["neural_search_score"].mean()
    ) / similar_chunks_video_metadata_df["neural_search_score"].std()

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
    ).sort_values("neural_search_score", ascending=False)
    total_time = time() - start_time
    # print(f"Total time to get video metadata and text: {total_time} seconds.")

    # Return the DataFrame as a list of dictionaries
    return segment_chunks_to_showcase_df.to_json(orient="records")


def keyword_search(
    query,
    release_date_filter=None,
    video_type_filter=None,
    review_score_filter=None,
    n_results_to_consider=None,
    n_most_similar_videos=5,
    n_top_segments_per_video=3,
):
    """
    This method will run a keyword search on the database.
    """

    # Get the most similar transcriptions to the query
    most_similar_transcriptions_df = get_most_similar_transcriptions_filtered(
        query=query,
        engine=engine,
        release_date_filter=release_date_filter,
        review_score_filter=review_score_filter,
        video_type_filter=video_type_filter,
        n_results=n_results_to_consider,
        include_text=False,
    )

    # Aggregate the transcription stats
    aggregated_transcription_stats_df = most_similar_transcriptions_df.groupby(
        "url"
    ).agg(
        median_rank=("rank", "median"),
        mean_rank=("rank", "mean"),
        n_results=("rank", "count"),
    )

    # Determine the z-score of the count of results
    aggregated_transcription_stats_df["count_z_score"] = (
        aggregated_transcription_stats_df["n_results"]
        - aggregated_transcription_stats_df["n_results"].mean()
    ) / aggregated_transcription_stats_df["n_results"].std()

    # Now, make a "weighted median", which is the median rank weighted by the z-score of the count of results
    aggregated_transcription_stats_df["keyword_search_score"] = (
        aggregated_transcription_stats_df["median_rank"]
        # * aggregated_transcription_stats_df["count_z_score"]
    )

    # Add a z-score column for the keyword search score
    aggregated_transcription_stats_df["keyword_search_score_z_score"] = (
        aggregated_transcription_stats_df["keyword_search_score"]
        - aggregated_transcription_stats_df["keyword_search_score"].mean()
    ) / aggregated_transcription_stats_df["keyword_search_score"].std()

    # Sort the dataframe by the weighted median rank
    aggregated_transcription_stats_df = (
        aggregated_transcription_stats_df.sort_values(
            "keyword_search_score", ascending=False
        )
        .head(n_most_similar_videos)
        .reset_index()
        .copy()
    )

    # Uplaod the dataframe to the database
    aggregated_transcription_stats_df.to_sql(
        "most_similar_transcriptions_temp", engine, if_exists="replace"
    )

    # Create a query to get the video metadata
    video_metadata_query = """
    SELECT
        video.*
    FROM
        video_metadata video
    JOIN
        most_similar_transcriptions_temp transcriptions
    ON
        video.url = transcriptions.url
    """

    # Execute the query
    most_similar_videos_metadata_df = postgres.query_postgres(
        video_metadata_query, engine
    )

    # Re-join the aggreageted transcription stats with the video metadata
    aggregated_transcription_stats_with_metadata_df = (
        aggregated_transcription_stats_df.merge(
            most_similar_videos_metadata_df, on="url"
        )
    )

    transcriptions_to_fetch_df = most_similar_transcriptions_df.merge(
        aggregated_transcription_stats_with_metadata_df[["url"]],
        on="url",
    )

    if len(transcriptions_to_fetch_df) == 0:
        return json.dumps([])

    # Upload a temporary table to the database
    transcriptions_to_fetch_df.to_sql(
        "transcriptions_to_fetch_temp", engine, if_exists="replace"
    )

    # Now, we'll fetch the text for the segments
    fetch_text_for_segments_query = """
    SELECT
        transcriptions.text,
        transcriptions.url,
        transcriptions.segment_id AS id
    FROM
        transcriptions_to_fetch_temp segments
    LEFT JOIN
        transcriptions
    ON
        transcriptions.url = segments.url
        AND transcriptions.segment_id = segments.id
    """

    # Execute the query
    transcriptions_with_text_df = postgres.query_postgres(
        fetch_text_for_segments_query, engine
    )

    # Aggregate this into the transcriptions_to_fetch_df dataframe
    transcriptions_to_fetch_df = transcriptions_to_fetch_df.merge(
        transcriptions_with_text_df, on=["url", "id"]
    )

    # Aggregate the text into a list
    transcriptions_to_fetch_df = (
        transcriptions_to_fetch_df.sort_values("rank", ascending=False)
        .groupby("url")
        .agg(top_segment_chunks=("text", lambda x: list(x)[:n_top_segments_per_video]))
        .reset_index()
    )

    # Add this to the aggregated_transcription_stats_with_metadata_df dataframe
    aggregated_transcription_stats_with_metadata_and_text_df = (
        aggregated_transcription_stats_with_metadata_df.merge(
            transcriptions_to_fetch_df, on="url"
        )
    )

    # Fix this DataFrame's columns
    aggregated_transcription_stats_with_metadata_and_text_df = (
        aggregated_transcription_stats_with_metadata_and_text_df.drop(
            columns=["median_rank", "mean_rank", "count_z_score"]
        ).rename(
            columns={
                "n_results": "n_segment_matches",
            }
        )
    )

    # If there's only one result, we'll set the keyword search score to 1, and the z-score to 2
    if len(aggregated_transcription_stats_with_metadata_and_text_df) == 1:
        aggregated_transcription_stats_with_metadata_and_text_df[
            "keyword_search_score"
        ] = 1
        aggregated_transcription_stats_with_metadata_and_text_df[
            "keyword_search_score_z_score"
        ] = 0.5

    # Return the DataFrame as a JSON
    return aggregated_transcription_stats_with_metadata_and_text_df.to_json(
        orient="records"
    )


def rerank_segment_chunks_for_urls(
    query: str, urls: list, search_method: str = "hybrid", n_top_segment_chunks: int = 3,
    neural_weight: float = 0.6, keyword_weight: float = 1
):
    """
    This method will "rerank" the segment chunks for a given list of URLs. This will consider
    all of the segments for each video, and will rank them based on their relevance to the
    query. This will be done according to the `search_method` parameter, which can be either
    "neural", "keyword", or "hybrid". The `n_top_segment_chunks` parameter will determine the
    number of top segment chunks to return for each video.
    """

    # First, we're going to create a temporary table in the database to store the results
    table_name = "temp_search_results_for_reranking"
    pd.DataFrame({"url": urls}).to_sql(
        table_name, engine, if_exists="replace", index=False
    )

    # If the search method is "neural", we're going to identify the segments that are most relevant to the query
    if search_method == "neural":
        most_similar_segment_chunks_df = (
            pg_queries.most_similar_embeddings_to_text_by_url(
                text=query, urls=urls, engine=engine, n_per_url=n_top_segment_chunks
            )
        )

    # If the search method is "keyword", we're going to identify the segments that are most relevant to the query
    if search_method == "keyword":
        most_similar_segment_chunks_df = (
            pg_queries.get_most_similar_transcriptions_by_url(
                query=query, urls=urls, engine=engine, n_per_url=n_top_segment_chunks
            )
        )
        most_similar_segment_chunks_df["cos_sim"] = None

    # Finally: if the search method is "hybrid", we'll run both the keyword and neural searches and aggregate with RRF
    if search_method == "hybrid":
        most_similar_segment_chunks_df = (
            pg_queries.get_most_similar_transcriptions_by_url_hybrid(
                text=query,
                urls=urls,
                engine=engine,
                n_per_url=n_top_segment_chunks,
                neural_weight=neural_weight,
                keyword_weight=keyword_weight,
            )
        )

    # Aggregate the segment chunks into lists
    aggregated_segment_chunks_df = (
        most_similar_segment_chunks_df.groupby("url")
        .agg(
            top_segment_chunks=("text", lambda x: x.tolist()),
            cos_sim=("cos_sim", lambda x: x.tolist()),
        )
        .reset_index()
    )

    # Strip any spaces from the segment chunks
    aggregated_segment_chunks_df["top_segment_chunks"] = aggregated_segment_chunks_df[
        "top_segment_chunks"
    ].apply(lambda text_list: [x.strip() for x in text_list])

    def clean_cos_sim_list(cos_sim_list):
        """
        This function will clean the cosine similarity list. It will convert any None or NaN values to 0.
        """
        return [0 if pd.isna(x) else x for x in cos_sim_list]

    # Clean the cosine similarity list
    aggregated_segment_chunks_df["cos_sim"] = aggregated_segment_chunks_df[
        "cos_sim"
    ].apply(clean_cos_sim_list)

    # Return the fused DataFrame
    return aggregated_segment_chunks_df


def hybrid_search(
    query: str,
    release_date_filter: list = None,
    video_type_filter: list = None,
    review_score_filter: list = None,
    max_video_per_search_method: int = 5,
    max_results=10,
    keyword_weight: float = 1,
    neural_weight: float = 1,
):
    """
    This method combines both the `keyword_search` and `neural_search` methods to create a hybrid search.
    The results from each search are merged via Reciprocal Rank Fusion (RRF).
    """

    # Run the neural search and keyword search in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_neural = executor.submit(
            neural_search,
            query=query,
            release_date_filter=release_date_filter,
            video_type_filter=video_type_filter,
            review_score_filter=review_score_filter,
            n_videos_to_return=max_video_per_search_method,
        )
        future_keyword = executor.submit(
            keyword_search,
            query=query,
            release_date_filter=release_date_filter,
            video_type_filter=video_type_filter,
            review_score_filter=review_score_filter,
            n_most_similar_videos=max_video_per_search_method,
        )

    # Get the results
    neural_results_json_str = future_neural.result()
    keyword_results_json_str = future_keyword.result()

    # Get DataFrame from the results
    neural_results_df = pd.DataFrame(json.loads(neural_results_json_str)).reset_index(
        drop=True
    )
    keyword_results_df = pd.DataFrame(json.loads(keyword_results_json_str)).reset_index(
        drop=True
    )

    print(
        f"Retrieved {len(neural_results_df)} neural results and {len(keyword_results_df)} keyword results."
    )

    if len(keyword_results_df) == 0:
        return neural_results_json_str

    # Create a DataFrame with the metadata of the resulting videos' metadata
    all_resulting_videos_metadata_df = (
        pd.concat(
            [
                neural_results_df.drop(
                    columns=[
                        "top_segment_chunks",
                        "n_segment_matches",
                        "neural_search_score",
                        "neural_search_score_z_score",
                    ]
                ).copy(),
                keyword_results_df.drop(
                    columns=[
                        "top_segment_chunks",
                        "n_segment_matches",
                        "keyword_search_score",
                        "keyword_search_score_z_score",
                    ]
                ).copy(),
            ]
        )
        .drop_duplicates(subset=["url"])
        .reset_index()
    )

    # Do some reciprocal rank fusion
    k = 60
    neural_results_df["neural_score"] = k / (neural_results_df.index + 1 + k)
    keyword_results_df["keyword_score"] = k / (keyword_results_df.index + 1 + k)
    fused_df = (
        neural_results_df[["url", "neural_score", "neural_search_score_z_score"]]
        .copy()
        .merge(
            keyword_results_df[
                ["url", "keyword_score", "keyword_search_score_z_score"]
            ],
            on="url",
            how="outer",
        )
        .fillna(0)
    )
    fused_df["fused_score"] = (
        fused_df["neural_score"] * neural_weight
        + fused_df["keyword_score"] * keyword_weight
    )
    fused_df["avg_z_score"] = (
        fused_df["neural_search_score_z_score"] * neural_weight
        + fused_df["keyword_search_score_z_score"] * keyword_weight
    ) / (neural_weight + keyword_weight)
    fused_df = (
        fused_df.sort_values(
            [
                "fused_score",
                "avg_z_score",
            ],
            ascending=[False, False],
        )
        .reset_index(drop=True)
        .head(max_results)
    )

    # Run the reranking of the segment chunks
    reranked_segment_chunks_df = rerank_segment_chunks_for_urls(
        query=query,
        urls=fused_df["url"].tolist(),
        search_method="hybrid",
        n_top_segment_chunks=3,
    )

    # Merge the reranked segment chunks with the fused_df
    fused_with_text_df = fused_df.merge(
        reranked_segment_chunks_df,
        on="url",
        how="left",
    )

    # Create a DataFrame with the metadata of the resulting videos' metadata
    result_df = fused_with_text_df.copy().merge(
        all_resulting_videos_metadata_df, on="url", how="left"
    )

    # Return the result as a JSON
    return result_df.to_json(orient="records")
