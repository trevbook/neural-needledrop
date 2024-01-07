"""
This file contains various utility functions for interacting with YouTube.
It mostly uses `pytube` (or, rather, the `pytubefix` fork of pytube) to do so.
Source: https://github.com/pytube/pytube/issues/1857
`pytubefix`: https://github.com/JuanBindez/pytubefix/tree/main
"""

# =====
# SETUP
# =====
# The code below will set up the file.

# General import statements
from pytubefix import YouTube
import traceback
import datetime
import subprocess

# Importing custom modules
from utils.logging import get_dummy_logger

# =======================
# GENERAL YOUTUBE METHODS
# =======================
# All of these methods are general purpose YouTube methods.


def get_video_urls_from_channel(
    channel, most_recent_video_url=None, video_limit=None, video_parse_step_size=10
):
    """
    Helper method to identify all of the video URLs from a channel.
    If `most_recent_video_url` is None, then we're going to download information for all of the videos we can,
    all the way up to the `video_limit`. If *that* is None, then we're going to download information for all of the videos.
    The `video_parse_step_size` indicates how many videos we're going to parse at a time.
    """

    # Initialize the video URLs
    video_urls = []

    # Initialize the video count
    video_count = 0

    # Iterate through the channel's videos until we find the `most_recent_video_url`
    while most_recent_video_url not in video_urls:
        # Fetch the video URLs
        new_video_urls = channel.video_urls[
            video_count : video_count + video_parse_step_size
        ]

        # Break out if no new video URLs were found
        if len(new_video_urls) == 0:
            break

        video_urls.extend(new_video_urls)

        # Update the video count
        video_count += video_parse_step_size

        # If we've reached the video limit, then break
        if video_limit is not None and video_count >= video_limit:
            break

    # Return the video URLs
    return video_urls


def download_audio_from_video(video_url, data_folder_path):
    """
    This method will download the audio for a given video URL.
    """

    # We'll wrap the entire method in a try/except block so that we can catch any errors that occur
    try:
        # Create a video object
        video = YouTube(video_url)

        # Find the highest-bitrate mp4 audio stream
        highest_bitrate_mp4_audio_stream = None
        highest_bitrate_found = 0
        for stream in video.streams.filter(only_audio=True):
            if stream.mime_type == "audio/mp4":
                if stream.abr is None:
                    if highest_bitrate_found == 0:
                        highest_bitrate_mp4_audio_stream = stream
                        highest_bitrate_found = 128
                    continue
                stream_bitrate = int(stream.abr.split("kbps")[0])
                if stream_bitrate > highest_bitrate_found:
                    highest_bitrate_mp4_audio_stream = stream
                    highest_bitrate_found = stream_bitrate

        # Download the audio
        highest_bitrate_mp4_audio_stream.download(
            output_path=data_folder_path,
            filename=f"{video.video_id}.m4a",
            skip_existing=False,
        )

    # If we run into an exception, then we'll throw a custom Exception with the traceback
    except Exception as e:
        raise Exception(
            f"Error downloading audio for video {video_url}: '{e}'\nTraceback is as follows:\n{traceback.format_exc()}"
        )


def convert_m4a_to_mp3(input_file_path, output_file_path, logger=None):
    logger = logger or get_dummy_logger()

    # Generate the ffmpeg command we'll use
    command = f"""ffmpeg -i {input_file_path} {output_file_path}"""
    logger.debug(f"Converting from .m4a to .mp3 using the following command: {command}")

    # Run the command
    rsp = subprocess.run(command)


def parse_metadata_from_video(video_url):
    """
    This method will parse a dictionary containing metadata from a video, given its URL.
    """

    # Create a video object
    video = YouTube(video_url)

    # Keep a dictionary to keep track of the metadata we're interested in
    video_metadata_dict = {}

    # We'll wrap this in a try/except block so that we can catch any errors that occur
    try:
        # Parse the `videoDetails` from the video; this contains a lot of the metadata we're interested in
        vid_info_dict = video.vid_info
        video_info_dict = vid_info_dict.get("videoDetails")

    # If we run into an Exception this early on, we'll raise an Exception
    except Exception as e:
        raise Exception(
            f"Error parsing video metadata for video {video_url}: '{e}'\nTraceback is as follows:\n{traceback.format_exc()}"
        )

    # Extract different pieces of the video metadata
    video_metadata_dict["id"] = video_info_dict.get("videoId")
    video_metadata_dict["title"] = video_info_dict.get("title")
    video_metadata_dict["length"] = video_info_dict.get("lengthSeconds")
    video_metadata_dict["channel_id"] = video_info_dict.get("channelId")
    video_metadata_dict["channel_name"] = video_info_dict.get("author")
    video_metadata_dict["short_description"] = video_info_dict.get("shortDescription")
    video_metadata_dict["view_ct"] = video_info_dict.get("viewCount")
    video_metadata_dict["url"] = video_info_dict.get("video_url")
    video_metadata_dict["small_thumbnail_url"] = (
        video_info_dict.get("thumbnail").get("thumbnails")[0].get("url")
    )
    video_metadata_dict["large_thumbnail_url"] = (
        video_info_dict.get("thumbnail").get("thumbnails")[-1].get("url")
    )

    # Try and extract the the publish_date
    try:
        publish_date = video.publish_date
        video_metadata_dict["publish_date"] = publish_date
    except:
        video_metadata_dict["publish_date"] = None

    # Try and extract the full description
    try:
        full_description = video.description
        video_metadata_dict["description"] = full_description
    except:
        video_metadata_dict["description"] = None

    # Use datetime to get the scrape_date (the current datetime)
    video_metadata_dict["scrape_date"] = datetime.datetime.now()

    # Add the url to the video_metadata_dict
    video_metadata_dict["url"] = video_url

    # Finally, return the video metadata dictionary
    return video_metadata_dict
