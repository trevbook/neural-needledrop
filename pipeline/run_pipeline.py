"""
This script helps to run the entire data pipeline for the 
Neural Needledrop project.
"""

# =====
# SETUP
# =====

# General import statements
from google.cloud import bigquery, storage

# Importing the different jobs that we'll run
from jobs.initialize_cloud_resources import run_initialize_cloud_resources_job
from jobs.download_video_metadata import run_download_video_metadata_job
from jobs.enrich_video_metadata import run_enrich_video_metadata_job
from jobs.download_audio import run_download_audio_job
from jobs.transcribe_audio import run_transcribe_audio_job
from jobs.embed_transcriptions import run_embed_transcriptions_job

# Importing other custom utilities
from utils.pipeline import pipeline_job_wrapper, logger
from utils.settings import GBQ_PROJECT_ID

# ====================
# RUNNING THE PIPELINE
# ====================

# Below, I'll define a method that will run the entire pipeline.
# This method will be called at the bottom of this script.


def neural_needledrop_pipeline():
    """
    This method will run through each of the necessary steps for the
    Neural Needledrop data pipeline.
    """

    # Log that we're starting the pipeline
    logger.info("STARTING NEURAL NEEDLEDROP PIPELINE.")

    # Set up both a GBQ and GCS client
    gbq_client = bigquery.Client(project=GBQ_PROJECT_ID)
    gcs_client = storage.Client(project=GBQ_PROJECT_ID)

    # Initialize the cloud resources
    pipeline_job_wrapper(
        run_initialize_cloud_resources_job, gbq_client=gbq_client, gcs_client=gcs_client
    )

    # Download the video metadata
    pipeline_job_wrapper(
        run_download_video_metadata_job,
        channel_url="https://www.youtube.com/c/theneedledrop",
        video_limit=10,
        stop_at_most_recent_video=True,
        video_parse_step_size=20,
        time_to_sleep_between_requests=2,
        sleep_time_multiplier=2.25,
        gbq_client=gbq_client,
        n_days_to_not_scrape=3,
    )

    # Enrich the video metadata
    pipeline_job_wrapper(run_enrich_video_metadata_job, gbq_client=gbq_client)

    # Download the audio
    pipeline_job_wrapper(
        run_download_audio_job,
        n_max_videos_to_download=5,
        time_to_sleep_between_requests=10,
        sleep_multiplier=3.25,
        gbq_client=gbq_client,
        gcs_client=gcs_client,
    )

    # Transcribe the audio
    pipeline_job_wrapper(
        run_transcribe_audio_job,
        n_max_to_transcribe=3,
        gbq_client=gbq_client,
        gcs_client=gcs_client,
    )

    # Embed the transcriptions
    pipeline_job_wrapper(
        run_embed_transcriptions_job,
        gbq_client=gbq_client,
        gcs_client=gcs_client,
        max_videos_to_embed=3,
    )

    # Indicate that the pipeline is complete
    logger.info("NEURAL NEEDLEDROP PIPELINE COMPLETE.")


# If this script is being run from the command line, then run the
# pipeline
if __name__ == "__main__":
    neural_needledrop_pipeline()
