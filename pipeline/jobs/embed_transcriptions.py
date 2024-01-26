"""
This job will use OpenAI's API to embed transcriptions of YouTube videos.
"""

# =====
# SETUP
# =====
# The code below will set up the rest of this file

# General import statements
import pandas as pd
from pytubefix import YouTube, Channel
from google.cloud import bigquery, storage
import traceback
import time
import random
from tqdm import tqdm
import pandas_gbq
import datetime
from pathlib import Path
from google.cloud.exceptions import NotFound
import whisper
import numpy as np

# Importing custom utility functions
import utils.gbq as gbq_utils
import utils.gcs as gcs_utils
import utils.openai as openai_utils
import utils.logging as logging_utils
from utils.settings import GBQ_PROJECT_ID, GBQ_DATASET_ID, TQDM_ENABLED, LOG_TO_CONSOLE
import utils.miscellaneous as misc_utils

# ========
# MAIN JOB
# ========
# The code below will define the main function that will be called when this job is run


def run_embed_transcriptions_job(
    gcs_client=None,
    gbq_client=None,
    temp_download_directory="temp_embedding_data/",
    max_parallel_embedding_workers=4,
    max_parallel_upload_workers=8,
    max_videos_to_embed=100,
    embedding_model="text-embedding-3-small",
):
    """
    This method will use the OpenAI API to embed transcriptions of YouTube videos.
    """

    # Set up a logger
    logger = logging_utils.get_logger(
        name="pipeline.embed_transcriptions", log_to_console=LOG_TO_CONSOLE
    )

    # If the GCS client is None, then instantiate it
    if gcs_client is None:
        gcs_client = storage.Client(project=GBQ_PROJECT_ID)

    # If the GBQ client is None, then instantiate it
    if gbq_client is None:
        gbq_client = bigquery.Client(project=GBQ_PROJECT_ID)

    # Make sure the temp_download_directory exists
    Path(temp_download_directory).mkdir(parents=True, exist_ok=True)

    # ===================================
    # DETERMINING TRANSCRIPTIONS TO EMBED
    # ===================================
    # The code below will determine which transcriptions to embed

    # The query below will determine which transcriptions we need to embed
    transcriptions_to_embed_query = f"""
    
    WITH all_transcriptions_to_embed AS (
        SELECT
        transcript.*
        FROM
        `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.transcriptions` transcript
        LEFT JOIN
        `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.embeddings` embedding
        ON
        embedding.video_url = transcript.url
        WHERE
        embedding.id IS NULL
    ),
    
    videos_to_embed AS (
        SELECT
        DISTINCT url
        FROM
        all_transcriptions_to_embed
        LIMIT {max_videos_to_embed}
    )
    
    SELECT
    all_transcriptions_to_embed.*
    FROM
    all_transcriptions_to_embed
    INNER JOIN
    videos_to_embed
    ON
    videos_to_embed.url = all_transcriptions_to_embed.url
    """

    # Execute the query
    transcriptions_to_embed_df = pd.read_gbq(
        transcriptions_to_embed_query, project_id=GBQ_PROJECT_ID
    )

    logger.info(
        f"Found {len(transcriptions_to_embed_df):,} individual transcription segments to embed."
    )

    # =======================
    # CHUNKING TRANSCRIPTIONS
    # =======================
    # Next up: I'm going to chunk the transcription segments together. This ensures
    # that there's a decent amount of context for the model to work with in each embedding.

    # Define the segment chunk sizes
    segment_chunk_sizes = [4, 8]

    # Initialize a list to hold the segment_chunk rows
    segment_chunks = []

    # Iterate through each unique URL in the DataFrame
    for url in transcriptions_to_embed_df["url"].unique():
        # Filter the DataFrame for the current URL
        url_df = transcriptions_to_embed_df[transcriptions_to_embed_df["url"] == url]

        # Extract the video ID from the URL
        video_id = url.split("watch?v=")[-1]

        # Sort the DataFrame by the segment_start
        url_df = url_df.sort_values(by=["segment_start"])

        # Iterate through the defined chunk sizes
        for chunk_size in segment_chunk_sizes:
            # Iterate through the segments in steps of chunk_size
            for i in range(0, len(url_df), chunk_size):
                # Get the chunk of segments
                segment_chunk_df = url_df.iloc[i : i + chunk_size]

                # Concatenate the text of the segments to form the chunk text
                chunk_text = " ".join(segment_chunk_df["text"].tolist())

                # Determine a new ID for this particular segment chunk
                segment_chunk_id = f"{video_id}_{i}_{i+chunk_size}"

                # Create a dictionary for the segment_chunk row
                segment_chunk_row = {
                    "id": segment_chunk_id,
                    "video_url": url,
                    "embedding_type": "segment_chunk",
                    "start_segment": i,
                    "end_segment": i + chunk_size,
                    "segment_length": chunk_size,
                    "text": chunk_text.strip(),
                }

                # Append the segment_chunk_row to the list of segment_chunks
                segment_chunks.append(segment_chunk_row)

    # Create a DataFrame that has the segment_chunks
    segment_chunks_df = pd.DataFrame(segment_chunks)

    logger.info(
        f"Groupped the transcription segments into {len(segment_chunks_df):,} segment chunks."
    )

    # ==========================
    # EMDED TRANSCRIPTION CHUNKS
    # ==========================
    # Next, we'll actually run through the embedding process for each of the transcription chunks.

    logger.info(f"Embedding the transcription chunks...")

    # We're going to collect all of the embeddings in this list
    embeddings_col = openai_utils.embed_text_list(
        list(segment_chunks_df["text"]),
        show_progress=TQDM_ENABLED,
        max_workers=max_parallel_embedding_workers,
        model=embedding_model,
    )

    # Add this column containing all of the embeddings to the segment_chunks_df
    segment_chunks_df["embedding"] = embeddings_col

    logger.info(f"Finished embedding the transcription chunks.")

    # ==============
    # STORING IN GCS
    # ==============
    # Now that we've got all of the embeddings, we'll store them in GCS.

    logger.info(f"Storing the embeddings in GCS...")

    # We're going to keep a list of the file paths of the embeddings we save
    embedding_file_paths = []
    for row in segment_chunks_df.itertuples():
        video_id = row.id
        emb_filename = Path(temp_download_directory) / f"/{video_id}.npy"
        emb = row.embedding
        openai_utils.save_as_npy(
            embedding=emb,
            file_name=emb_filename,
        )
        embedding_file_paths.append(emb_filename)

    # Use the bulk upload method to upload all .npy files to GCS
    gcs_utils.upload_files_to_bucket(
        file_path_list=embedding_file_paths,
        bucket_name="neural-needledrop-embeddings",
        project_id=GBQ_PROJECT_ID,
        gcs_client=gcs_client,
        logger=logger,
        show_progress=TQDM_ENABLED,
        max_workers=max_parallel_upload_workers,
    )

    # Remove the files after uploading
    for file_path in embedding_file_paths:
        Path(file_path).unlink()

    # Delete the temp_embeddings folder
    Path(temp_download_directory).rmdir()

    # ============
    # UPDATING GBQ
    # ============
    # Finally, we'll update GBQ that we've embedded these transcription chunks

    # Define the rows_to_upload list
    rows_to_upload = []

    # Iterate through the segment_chunks_df and extract the rows
    for row in segment_chunks_df.itertuples():
        # Extract the row
        row_dict = {
            "id": row.id,
            "video_url": row.video_url,
            "gcs_uri": f"gs://neural-needledrop-embeddings/{row.id}.npy",
            "embedding_type": row.embedding_type,
            "start_segment": row.start_segment,
            "end_segment": row.end_segment,
            "segment_length": row.segment_length,
        }

        # Append the row to the rows_to_upload
        rows_to_upload.append(row_dict)

    # Add the rows to the `backend_data.embeddings` table
    gbq_utils.add_rows_to_table(
        project_id=GBQ_PROJECT_ID,
        dataset_id=GBQ_DATASET_ID,
        table_id="embeddings",
        rows=rows_to_upload,
        gbq_client=gbq_client,
        logger=logger,
    )
