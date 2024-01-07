"""
This job will download audio from YouTube videos. 
"""

# =====
# SETUP
# =====
# The code below will set up the rest of this file

# General import statements
import pandas as pd
from google.cloud import bigquery, storage
import traceback
from tqdm import tqdm
import datetime
from pathlib import Path

# Importing custom utility functions
import utils.gbq as gbq_utils
import utils.youtube as youtube_utils
import utils.gcs as gcs_utils
import utils.logging as logging_utils
from utils.settings import GBQ_PROJECT_ID, GBQ_DATASET_ID, TQDM_ENABLED, LOG_TO_CONSOLE
import utils.miscellaneous as misc_utils

# ========
# MAIN JOB
# ========
# This is the main function that will be called when this job is run


def run_download_audio_job(
    n_max_videos_to_download=100,
    time_to_sleep_between_requests=10,
    sleep_multiplier=2.5,
    temp_download_directory="temp_audio_data/",
    gbq_client=None,
    gcs_client=None,
):
    """
    This method will download the audio streams for the videos in `video_metadata` that haven't been downloaded.
    The audio streams will be saved to the `neural-needledrop-audio` bucket in GCS.
    """

    # Set up a logger
    logger = logging_utils.get_logger(
        name="pipeline.download_audio", log_to_console=LOG_TO_CONSOLE
    )

    # Log that we're starting the job
    logger.info("Starting the DOWNLOAD AUDIO job.")

    # If the GCS client is None, then instantiate it
    if gcs_client is None:
        gcs_client = storage.Client(project=GBQ_PROJECT_ID)

    # If the GBQ client is None, then instantiate it
    if gbq_client is None:
        gbq_client = bigquery.Client(project=GBQ_PROJECT_ID)

    # Make the temp_download_directory a Path object
    temp_download_directory = Path(temp_download_directory)

    # =============================
    # DETERMINING AUDIO TO DOWNLOAD
    # =============================
    # The code below will determine which videos we need to download audio for.
    # The query below will determine which videos we need to download audio for
    videos_for_audio_parsing_query = f"""
    SELECT
    video.url
    FROM
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.video_metadata` video
    LEFT JOIN
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.audio` audio
    ON
    audio.video_url = video.url
    WHERE
    audio.audio_gcr_uri IS NULL
    LIMIT {n_max_videos_to_download}
    """

    # Execute the query
    videos_for_audio_parsing_df = pd.read_gbq(
        videos_for_audio_parsing_query, project_id=GBQ_PROJECT_ID
    )

    # If there aren't any videos to download, log that and return
    if len(videos_for_audio_parsing_df) == 0:
        logger.info("No videos to download audio for. Exiting job...")
        return
    logger.info(f"Downloading audio for {len(videos_for_audio_parsing_df):,} videos.")

    # =================
    # DOWNLOADING AUDIO
    # =================
    # Next, we're going to iterate through each of the videos and download their audio.

    # Iterate through the videos and download their audio
    for video_url in tqdm(videos_for_audio_parsing_df["url"], disable=not TQDM_ENABLED):
        # We'll wrap this in a try/except block so that we can catch any errors that occur
        try:
            logger.debug(f"Downloading audio for {video_url}")

            # Download the audio from the video
            youtube_utils.download_audio_from_video(
                video_url=video_url, data_folder_path=temp_download_directory
            )
            misc_utils.sleep_random_time(
                time_to_sleep_between_requests,
                sleep_multiplier,
            )

        # If we run into an Exception, then we'll print out the traceback
        except Exception as e:
            logger.error(
                f"Error downloading audio for {video_url}: '{e}'\nThe traceback is:\n{traceback.format_exc()}"
            )

    # ======================
    # ULPAODING AUDIO TO GCS
    # ======================
    # The code below will upload the audio to GCS

    # Iterate through all of the video urls in the videos_for_audio_parsing_df
    for row in tqdm(
        list(videos_for_audio_parsing_df.itertuples()), disable=not TQDM_ENABLED
    ):
        # We'll wrap this in a try/except block so that we can catch any errors that occur
        try:
            # Get the video url
            video_url = row.url

            # Get the video id
            video_id = video_url.split("watch?v=")[-1]

            # Get the path to the audio file
            audio_file_path = temp_download_directory / f"{video_id}.m4a"

            # Check to see if this file exists
            if not Path(audio_file_path).exists():
                # If it doesn't exist, then we'll continue. Print out a warning
                logger.warning(
                    f"When trying to upload audio for {video_url}, we couldn't find the audio file at {audio_file_path}. Skipping..."
                )
                continue

            # Get the GCS URI
            gcs_uri = f"neural-needledrop-audio"

            # Upload the audio file to GCS
            audio_file_path_str = str(audio_file_path)

            # Convert the audio file to .mp3 using youtube_utils
            youtube_utils.convert_m4a_to_mp3(
                input_file_path=audio_file_path_str,
                output_file_path=audio_file_path_str.replace(".m4a", ".mp3"),
                logger=logger,
            )

            # Remove the .m4a file
            audio_file_path.unlink()

            # Update the audio_file_path_str
            audio_file_path = Path(audio_file_path_str.replace(".m4a", ".mp3"))
            audio_file_path_str = str(audio_file_path)

            gcs_utils.upload_file_to_bucket(
                file_path=audio_file_path_str,
                bucket_name=gcs_uri,
                project_id=GBQ_PROJECT_ID,
                gcs_client=gcs_client,
                logger=logger,
            )

            # Create a dictionary to store the audio metadata
            audio_metadata_dict = {
                "video_url": video_url,
                "audio_gcr_uri": f"gs://{gcs_uri}/{audio_file_path.name}",
                "scrape_date": datetime.datetime.now(),
            }

            # Add the audio metadata to the table
            gbq_utils.add_rows_to_table(
                project_id=GBQ_PROJECT_ID,
                dataset_id=GBQ_DATASET_ID,
                table_id="audio",
                rows=[audio_metadata_dict],
                gbq_client=gbq_client,
                logger=logger,
            )

            # Delete the audio file if delete_files_after_upload
            audio_file_path.unlink()

        # If we run into an Exception, then we'll print out the traceback
        except Exception as e:
            logger.error(
                f"While uploading audio for {video_url}, we ran into an error: '{e}'\nThe traceback is:\n{traceback.format_exc()}"
            )

    # Make sure that all of the files have been deleted
    for file_path in temp_download_directory.iterdir():
        try:
            file_path.unlink()
        except:
            pass

    # Remove the temp_download_directory
    Path(temp_download_directory).rmdir()
