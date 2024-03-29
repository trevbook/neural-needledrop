"""
This job will download metadata about YouTube videos.
"""

# =====
# SETUP
# =====
# The code below will set up the rest of this file

# General import statements
import pandas as pd
from pytubefix import Channel
from google.cloud import bigquery
import traceback
from tqdm import tqdm

# Importing custom utility functions
import utils.gbq as gbq_utils
import utils.youtube as youtube_utils
import utils.logging as logging_utils
import utils.miscellaneous as misc_utils
from utils.settings import GBQ_PROJECT_ID, GBQ_DATASET_ID, LOG_TO_CONSOLE, TQDM_ENABLED

# ========
# MAIN JOB
# ========
# Below, I'm going to define the function that will be called
# when this job is run.


def run_download_video_metadata_job(
    channel_url,
    delete_existing_tables=False,
    video_limit=1000,
    stop_at_most_recent_video=None,
    video_parse_step_size=200,
    gbq_client=None,
    time_to_sleep_between_requests=3,
    sleep_time_multiplier=2.5,
    n_days_to_not_scrape=1,
):
    """
    This method will download metadata about YouTube videos
    that we haven't already downloaded.
    """

    # Set up a logger
    logger = logging_utils.get_logger(
        name="pipeline.download_video_metadata", log_to_console=LOG_TO_CONSOLE
    )

    # If the GBQ client isn't provided, create it
    if gbq_client is None:
        gbq_client = bigquery.Client(project=GBQ_PROJECT_ID)

    # =============================
    # DETERMINING WHETHER TO SCRAPE
    # =============================
    # Below, we'll determine whether or not we should scrape this channel.

    # Log that we're starting the job, as well as some starting information
    logger.info("Determining whether or not to scrape this channel.")

    # Determine the most recent `scrape_date` field in the `video_metadata` table
    try:
        most_recent_date = (
            pd.read_gbq(
                f"SELECT scrape_date FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.video_metadata` ORDER BY scrape_date DESC LIMIT 1"
            )
            .iloc[0]
            .scrape_date
        )
        
        if n_days_to_not_scrape is None:
            skip_scraping = False

        # If today's date is within the timedelta, skip scraping
        if (pd.Timestamp.now() - most_recent_date) < pd.Timedelta(
            days=n_days_to_not_scrape
        ):
            skip_scraping = True
        else:
            skip_scraping = False

    except Exception as e:
        skip_scraping = False

    # If we're skipping scraping, then we'll log that and exit
    if skip_scraping:
        logger.info(
            f"Skipping scraping for channel {channel_url} because we've already scraped it within the last {n_days_to_not_scrape} days."
        )
        return

    # =================================
    # DETERMINING MOST RECENT VIDEO URL
    # =================================
    # Below, we'll figure out the most recent video that we've got data for
    # for this channel. We'll use this to determine which videos we need to
    # scrape.

    # Log that we're starting the job, as well as some starting information
    logger.info("Starting the DOWNLOAD VIDEO METADATA job.")
    logger.info(f"Crawling channel {channel_url}.")

    # If we don't want to stop at the most recent video, then we'll set the most recent video to None
    if stop_at_most_recent_video is False:
        most_recent_video_url = None

    # Otherwise, we'll determine the most recent video
    else:
        # Define the query that'll grab the most recent video url
        most_recent_video_url_query = """
        SELECT
        metadata.url
        FROM
        `neural-needledrop.backend_data.video_metadata` metadata
        ORDER BY
        publish_date DESC, scrape_date DESC
        LIMIT 1
        """

        # Use pandas-gbq to run the query
        most_recent_video_url_df = pd.read_gbq(
            most_recent_video_url_query, project_id=GBQ_PROJECT_ID
        )

        # If the length of the dataframe is zero, then we need to set the url to None
        if len(most_recent_video_url_df) == 0:
            most_recent_video_url = None

        # Otherwise, we can just grab the url from the dataframe
        else:
            most_recent_video_url = most_recent_video_url_df.iloc[0]["url"]

    # Log some information about the most recent video url
    if most_recent_video_url is None:
        logger.info(f"No most recent video url found.")
    else:
        logger.info(f"Most recent video url found: {most_recent_video_url}")

    # ==========================
    # DETERMINE VIDEOS TO SCRAPE
    # ==========================
    # Below, we'll try and determine which videos should be scraped
    # for this particular channel, using the `most_recent_video_url`,
    # `video_limit`, and `video_parse_step_size` arguments.

    video_urls_to_parse = youtube_utils.get_video_urls_from_channel(
        channel=Channel(channel_url),
        most_recent_video_url=None,
        video_limit=video_limit,
        video_parse_step_size=video_parse_step_size,
    )

    # If the `most_recent_video_url` argument was provided, we'll
    # remove all of the videos that were published before that video
    if most_recent_video_url is not None:
        video_urls_to_parse = video_urls_to_parse[
            : video_urls_to_parse.index(most_recent_video_url)
        ]

    # Now, we need to check GBQ to see which videos we've already scraped
    video_urls_to_parse_df = pd.DataFrame(video_urls_to_parse, columns=["url"])

    # Create a temporary table in GBQ
    temporary_table_name = gbq_utils.create_temporary_table_in_gbq(
        dataframe=video_urls_to_parse_df,
        project_id=GBQ_PROJECT_ID,
        dataset_name=GBQ_DATASET_ID,
        table_name="temporary_video_urls_to_parse",
        if_exists="replace",
        gbq_client=gbq_client,
        logger=logger,
    )

    # Create the query to identify the videos that we need to parse
    actual_videos_to_parse_query = f"""
    SELECT
    temp_urls.url
    FROM
    `{temporary_table_name}` temp_urls
    LEFT JOIN
    `backend_data.video_metadata` metadata
    ON
    metadata.url = temp_urls.url
    WHERE
    metadata.id IS NULL
    """

    # Execute the query
    actual_videos_to_parse_df = pd.read_gbq(
        actual_videos_to_parse_query, project_id=GBQ_PROJECT_ID
    )

    # Overriding the video_urls_to_parse with the contents of the actual_videos_to_parse_df
    video_urls_to_parse = list(actual_videos_to_parse_df["url"])

    # Use the gbq_utils to delete the temporary table
    (
        temp_table_project_id,
        temp_table_dataset_id,
        temp_table_name,
    ) = temporary_table_name.split(".")
    gbq_utils.delete_table(
        project_id=temp_table_project_id,
        dataset_id=temp_table_dataset_id,
        table_id=temp_table_name,
        gbq_client=gbq_client,
        logger=logger,
    )

    # If there aren't any videos to parse, we'll log that and exit
    if len(actual_videos_to_parse_df) == 0:
        logger.info(
            f"There aren't any videos to parse for channel {channel_url}. Exiting."
        )
        return

    # Log some information about how many videos we actually have to parse
    logger.info(
        f"Found {len(actual_videos_to_parse_df)} videos to parse for channel {channel_url}."
    )

    # =======================
    # SCRAPING VIDEO METADATA
    # =======================
    # Now that I've determined which videos ought to be scraped, I'm going to
    # iterate through each of them and scrape their metadata.

    # We'll iterate through each of the videos in the list and parse their metadata
    video_metadata_dicts_by_video_url = {}
    for video_url in tqdm(video_urls_to_parse, disable=not TQDM_ENABLED):
        # Include a debug statement
        logger.debug(f"Parsing metadata for video {video_url}.")

        # We'll wrap this in a try/except block so that we can catch any errors that occur
        try:
            # Parse the metadata from the video
            video_metadata_dict = youtube_utils.parse_metadata_from_video(video_url)

            # Add the video metadata dictionary to the dictionary of video metadata dictionaries
            video_metadata_dicts_by_video_url[video_url] = video_metadata_dict

            # Sleep for a random amount of time
            misc_utils.sleep_random_time(
                time_to_sleep_between_requests, sleep_time_multiplier
            )

        # If we run into an Exception, then we'll log an error and continue
        except Exception as e:
            logger.error(
                f"While parsing the video metadata for video {video_url}, we ran into the following error: '{e}'\nTraceback is as follows:\n{traceback.format_exc()}"
            )
            continue

    logger.info(
        f"Finished parsing metadata for {len(video_metadata_dicts_by_video_url)} videos."
    )

    # =======================
    # SCRAPING VIDEO METADATA
    # =======================
    # Now that I've determined which videos ought to be scraped, I'm going to
    # iterate through each of them and scrape their metadata.

    # Finally, now that we've parsed all of the metadata, we'll store it in GBQ
    rows_to_add = [val for val in video_metadata_dicts_by_video_url.values()]

    # Add the rows to the table
    if len(rows_to_add) > 0:
        gbq_utils.add_rows_to_table(
            project_id=GBQ_PROJECT_ID,
            dataset_id=GBQ_DATASET_ID,
            table_id="video_metadata",
            rows=rows_to_add,
            gbq_client=gbq_client,
            logger=logger,
        )

    logger.info("Finished adding rows to the `video_metadata` table.")
    logger.info("Finished the DOWNLOAD VIDEO METADATA job.")
