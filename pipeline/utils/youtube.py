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
import pandas as pd
from pytubefix import YouTube, Channel
from google.cloud import bigquery
import traceback
import time
import random
from tqdm import tqdm
import pandas_gbq
import datetime
import uuid
from datetime import timedelta
from pathlib import Path

# Importing custom utility functions
import utils.gbq as gbq_utils

# Indicate whether or not we want tqdm progress bars
tqdm_enabled = True

# Set some constants for the project
GBQ_PROJECT_ID = "neural-needledrop"
GBQ_DATASET_ID = "backend_data"

# Set the pandas_gbq context to the project ID
# pandas_gbq.context.project = GBQ_PROJECT_ID

# =======================
# GENERAL YOUTUBE METHODS
# =======================
# All of these methods are general purpose YouTube methods.


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
