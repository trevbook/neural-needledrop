"""
This job will use OpenAI's Whisper model to transcribe
the audio from YouTube videos.
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
import whisper

# Importing custom utility functions
import utils.gbq as gbq_utils
import utils.gcs as gcs_utils
import utils.logging as logging_utils
from utils.settings import GBQ_PROJECT_ID, GBQ_DATASET_ID, TQDM_ENABLED, LOG_TO_CONSOLE

# ========
# MAIN JOB
# ========
# The code below will define the main function that will be called when this job is run


def run_transcribe_audio_job(
    n_max_to_transcribe=50,
    gcs_client=None,
    gbq_client=None,
    temp_download_directory="temp_audio_data/",
    whisper_model_size="tiny",
):
    """
    This method will run the transcribe audio job.
    """

    # Initialize the whisper model
    whisper_model = whisper.load_model(whisper_model_size)

    # Set up a logger
    logger = logging_utils.get_logger(
        name="pipeline.transcribe_audio", log_to_console=LOG_TO_CONSOLE
    )

    # Log that we're starting the job
    logger.info("Starting the TRANSCRIBE AUDIO job.")

    # If the GCS client is None, then instantiate it
    if gcs_client is None:
        gcs_client = storage.Client(project=GBQ_PROJECT_ID)

    # If the GBQ client is None, then instantiate it
    if gbq_client is None:
        gbq_client = bigquery.Client(project=GBQ_PROJECT_ID)

    # Make sure that the temp_download_directory exists
    Path(temp_download_directory).mkdir(exist_ok=True, parents=True)

    # ===============================
    # DETERMINING AUDIO TO TRANSCRIBE
    # ===============================
    # The code below will determine which audio we need to transcribe

    # This query will determine all of the videos we need to transcribe
    videos_for_transcription_query = f"""
    SELECT
    DISTINCT(audio.video_url) AS url,
    audio.audio_gcr_uri
    FROM
    `backend_data.audio` audio 
    LEFT JOIN
    `backend_data.transcriptions` transcript
    ON
    audio.video_url = transcript.url
    WHERE
    transcript.created_at IS NULL
    LIMIT {n_max_to_transcribe}
    """

    # Execute the query
    videos_for_transcription_df = pd.read_gbq(
        videos_for_transcription_query, project_id=GBQ_PROJECT_ID
    )

    # If there aren't any rows in this dataframe, then log that and return
    if len(videos_for_transcription_df) == 0:
        logger.info("No videos to transcribe. Exiting job...")
        return

    # Otherwise, log some info about the videos we're transcribing
    logger.info(
        f"Found {len(videos_for_transcription_df):,} videos whose audio we need to transcribe."
    )

    # =================
    # DOWNLOADING AUDIO
    # =================
    # Next up: we're going to download all of the audio for the
    # videos we need to transcribe. We'll assume that the audio
    # is saved in the GCS bucket `neural-needledrop-audio`.

    # Iterate through all of the video urls in the videos_for_transcription_df
    for row in tqdm(
        list(videos_for_transcription_df.itertuples()), disable=not TQDM_ENABLED
    ):
        # Parse the GCS URI
        split_gcs_uri = row.audio_gcr_uri.split("gs://")[-1]
        bucket_name, file_name = split_gcs_uri.split("/")[0], "/".join(
            split_gcs_uri.split("/")[1:]
        )

        # Download the audio
        gcs_utils.download_file_from_bucket(
            bucket_name=bucket_name,
            file_name=file_name,
            destination_folder=temp_download_directory,
            project_id=GBQ_PROJECT_ID,
            gcs_client=gcs_client,
            logger=logger,
        )

    logger.info("Finished downloading the audio of the videos we need to transcribe.")

    # ==================
    # TRANSCRIBING AUDIO
    # ==================
    # Next: we're going to transcribe the audio with Whisper.

    logger.info(f"Starting the transcription process.")

    # We'll store the audio metadata in a dictionary
    audio_metadata_dict_by_video_url = {}

    # Iterate through each of the files in the `temp_data` directory and transcribe them
    for child_file in tqdm(
        list(Path(temp_download_directory).iterdir()), disable=not TQDM_ENABLED
    ):
        
        # We're going to wrap this in a try/except block so that we can catch any errors that occur
        try:
            if child_file.suffix != ".mp3":
                continue

            # Extract some data about the file
            video_url = f"https://www.youtube.com/watch?v={child_file.stem}"

            # Use whisper to transcribe the audio
            logger.debug(f"Transcribing audio for video {video_url}")
            whisper_transcription = whisper_model.transcribe(
                str(child_file), fp16=False
            )

            # Store the transcription in the audio_metadata_dict_by_video_url
            audio_metadata_dict_by_video_url[video_url] = whisper_transcription

        # If we run into an error, we'll log it and continue
        except Exception as e:
            logger.error(
                f"Error transcribing audio for video {video_url}: '{e}'\nTraceback is as follows:\n{traceback.format_exc()}"
            )
            continue

    # Log that we're done transcribing
    logger.info(f"Finished transcribing the audio.")

    # ===========================
    # TRANSFORMING TRANSCRIPTIONS
    # ===========================
    # Before I upload the transcriptions, I'll transform them a bit.

    # Create a DataFrame from the audio_metadata_dict_by_video_url
    audio_metadata_df = pd.DataFrame.from_dict(
        audio_metadata_dict_by_video_url, orient="index"
    )

    # Reset the index into a "url" column
    audio_metadata_df.reset_index(inplace=True, names=["url"])

    # Explode the "segments" column
    audio_metadata_df = audio_metadata_df.explode("segments")

    # Rename the "segment" column to "segment" in the audio_metadata_df
    audio_metadata_df = audio_metadata_df.rename(columns={"segments": "segment"})

    # Add a "created_at" column to the audio_metadata_df
    audio_metadata_df["created_at"] = datetime.datetime.now()

    # Alter the "text" column so that it's extracted from the "segment" column
    audio_metadata_df["text"] = audio_metadata_df["segment"].apply(
        lambda x: x.get("text", None)
    )

    # Add a "segment_type" column to the audio_metadata_df
    audio_metadata_df["segment_type"] = "small_segment"

    # We're going to extract some columns from the `segment` dictionary
    segment_columns_to_keep = ["id", "seek", "start", "end"]
    normalized_segments_df = pd.json_normalize(audio_metadata_df["segment"])
    normalized_segments_df = normalized_segments_df[segment_columns_to_keep]

    # Rename all of the columns so that they have "segment_" prepended to them
    normalized_segments_df = normalized_segments_df.rename(
        columns={col: f"segment_{col}" for col in normalized_segments_df.columns}
    )

    # Make the final_transcription_df
    final_transcription_df = pd.concat(
        [
            audio_metadata_df.drop(columns=["segment"]).reset_index(drop=True),
            normalized_segments_df.reset_index(drop=True),
        ],
        axis=1,
    ).copy()

    # ===============================
    # UPLOADING TRANSCRIPTIONS TO GBQ
    # ===============================
    # Finally, I'm going to upload all of the transcriptions to GBQ.

    # Define the name of the table we're going to create
    table_name = "temp_transcriptions"

    # Create the table
    gbq_utils.create_temporary_table_in_gbq(
        dataframe=final_transcription_df,
        project_id=GBQ_PROJECT_ID,
        dataset_name=GBQ_DATASET_ID,
        table_name=table_name,
        if_exists="replace",
        gbq_client=gbq_client,
        logger=logger,
    )

    # The following query will determine which transcripts we need to upload
    transcripts_to_upload_query = f"""
    SELECT
    DISTINCT(temp_transcript.url)
    FROM
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.{table_name}` temp_transcript
    LEFT JOIN
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.transcriptions` transcript
    ON
    transcript.url = temp_transcript.url
    WHERE
    transcript.created_at IS NULL
    """

    # Execute the query
    transcripts_to_upload_df = pd.read_gbq(
        transcripts_to_upload_query, project_id=GBQ_PROJECT_ID
    )

    # Create a DataFrame containing the transcripts that we need to upload
    final_transcriptions_to_upload_df = final_transcription_df.merge(
        transcripts_to_upload_df, on="url"
    )

    # Use the gbq_utils to add rows to the `backend_data.transcriptions` table
    gbq_utils.add_rows_to_table(
        project_id=GBQ_PROJECT_ID,
        dataset_id=GBQ_DATASET_ID,
        table_id="transcriptions",
        rows=final_transcription_df.to_dict(orient="records"),
        gbq_client=gbq_client,
        logger=logger,
    )

    # Delete the temporary table
    gbq_utils.delete_table(
        project_id=GBQ_PROJECT_ID,
        dataset_id=GBQ_DATASET_ID,
        table_id=table_name,
    )

    # Delete the temp_data directory and everything in it
    for child_file in Path(temp_download_directory).iterdir():
        child_file.unlink()
    Path(temp_download_directory).rmdir()

    # Log that we're done
    logger.info("Finished the TRANSCRIBE AUDIO job.")
