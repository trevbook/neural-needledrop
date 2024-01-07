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
import utils.youtube as youtube_utils
import utils.gcs as gcs_utils
import utils.logging as log_utils
import utils.enriching_data as enrichment_utils
import utils.openai as openai_utils

# Indicate whether or not we want tqdm progress bars
tqdm_enabled = True

# Set some constants for the project
GBQ_PROJECT_ID = "neural-needledrop"
GBQ_DATASET_ID = "backend_data"