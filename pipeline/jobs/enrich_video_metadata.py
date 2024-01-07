"""
This job will enrich the data in the `video_metadata` table with additional
information about the videos. 
"""

# =====
# SETUP
# =====
# The code below will set up the rest of this file

# General import statements
import pandas as pd
import numpy as np

# Importing custom utility functions
import utils.gbq as gbq_utils
import utils.logging as logging_utils
import utils.enriching_data as enrichment_utils
from utils.settings import GBQ_PROJECT_ID, GBQ_DATASET_ID, LOG_TO_CONSOLE

# ========
# MAIN JOB
# ========
# Below, I'm going to define the function that will be called
# when this job is run.


def run_enrich_video_metadata_job(max_n_videos_to_enrich=1000, gbq_client=None):
    """
    This method will enrich the data in the `video_metadata` table with additional
    information about the videos.
    """

    # Set up a logger
    logger = logging_utils.get_logger(
        name="pipeline.enrich_video_metadata", log_to_console=LOG_TO_CONSOLE
    )

    # Log that we're starting the job
    logger.info("Starting the ENRICH VIDEO METADATA job.")

    # The query below will define the videos that we need to enrich
    videos_to_enrich_query = f"""
    SELECT
    metadata.id,
    metadata.url,
    metadata.title,
    metadata.description
    FROM
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.video_metadata` metadata
    WHERE NOT EXISTS (
    SELECT 1
    FROM `backend_data.enriched_video_metadata` enriched_metadata
    WHERE enriched_metadata.url = metadata.url
    )
    LIMIT {max_n_videos_to_enrich}
    """

    # Execute the query
    videos_to_enrich_df = pd.read_gbq(videos_to_enrich_query, project_id=GBQ_PROJECT_ID)

    # If there are no videos to enrich, log that and return
    if len(videos_to_enrich_df) == 0:
        logger.info("No videos to enrich. Exiting the job...")
        return

    # Otherwise, log how many videos we're enriching
    logger.info(f"Found {len(videos_to_enrich_df)} videos to enrich")

    # Add a column containing the video type
    logger.info("Adding `video_type` enrichment.")
    videos_to_enrich_df["video_type"] = videos_to_enrich_df["title"].apply(
        lambda x: enrichment_utils.classify_video_type(x)
    )

    # Add a column containing the review score
    logger.info("Adding `review_score` enrichment.")
    videos_to_enrich_df["review_score"] = videos_to_enrich_df["description"].apply(
        lambda x: enrichment_utils.extract_review_score(x)
    )

    # Make a copy of the DataFrame that we're going to upload
    enriched_metadata_to_upload_df = videos_to_enrich_df.copy()

    # Drop the title and description columns
    enriched_metadata_to_upload_df.drop(columns=["title", "description"], inplace=True)

    # Replace the NaN values with None
    enriched_metadata_to_upload_df["review_score"] = enriched_metadata_to_upload_df[
        "review_score"
    ].replace({np.nan: None})

    # Add the rows to the `backend_data.enriched_video_metadata` table
    gbq_utils.add_rows_to_table(
        project_id=GBQ_PROJECT_ID,
        dataset_id=GBQ_DATASET_ID,
        table_id="enriched_video_metadata",
        rows=enriched_metadata_to_upload_df.to_dict(orient="records"),
        logger=logger,
        gbq_client=gbq_client,
    )

    # Log that we've finished the job
    logger.info("Finished the ENRICH VIDEO METADATA job.")
