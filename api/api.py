"""
This file contains the API for the Neural Needledrop app. The API is built using FastAPI. 
"""

# =====
# SETUP
# =====
# Below, I'm going to set up the file.

# Import statements
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from cachetools import cached, TTLCache
from typing import Optional
import traceback

# Create a cache object with a TTL (time-to-live) of 30 minutes (1800 seconds)
# and a maximum of 1000 items
cache = TTLCache(maxsize=10, ttl=1800)

# Importing custom modules
import utils.search as search_utils
from utils.logging import get_logger
from utils.settings import LOG_TO_CONSOLE
import utils.postgres_queries as pg_queries

# Set up a logger for this notebook
logger = get_logger("api", log_to_console=LOG_TO_CONSOLE)

# Setting up the FastAPI app
app = FastAPI()

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:5173",
        "http://localhost",
        "http://host.docker.internal",
        "http://34.173.97.175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========
# ENDPOINTS
# =========
# Next, we're going to define each of the app's endpoints


# Define a Pydantic model for the search request body
class SearchRequest(BaseModel):
    query: str
    neural_search_strength: float
    keyword_search_strength: float
    review_score_filter: Optional[list] = None
    video_type_filter: Optional[list] = None
    release_date_filter: Optional[list] = None


# The cache key will be a combination of the search request's query and the user's filters
def cache_key(*args, **kwargs):
    return json.dumps(args, sort_keys=True, default=str) + json.dumps(
        kwargs, sort_keys=True, default=str
    )


@app.post("/search")
@cached(cache, key=cache_key)
def search(search_request: SearchRequest):
    """
    This is the search endpoint.
    """

    # Print the search request
    print(f"\nRECIEVED REQUEST:\n{search_request}\n")

    try:

        # If the user's neural search strength is 1, then we'll run a neural search
        if search_request.neural_search_strength == 1:
            return search_utils.neural_search(
                query=search_request.query,
                review_score_filter=search_request.review_score_filter,
                video_type_filter=search_request.video_type_filter,
                release_date_filter=search_request.release_date_filter,
            )

        # If the user's keyword search strength is 1, then we'll run a keyword search
        elif search_request.keyword_search_strength == 1:
            return search_utils.keyword_search(
                query=search_request.query,
                review_score_filter=search_request.review_score_filter,
                video_type_filter=search_request.video_type_filter,
                release_date_filter=search_request.release_date_filter,
            )

        # Otherwise, the user's search strength is somewhere in between
        else:

            return search_utils.hybrid_search(
                query=search_request.query,
                max_video_per_search_method=5,
                max_results=10,
                keyword_weight=search_request.keyword_search_strength,
                neural_weight=search_request.neural_search_strength,
                review_score_filter=search_request.review_score_filter,
                video_type_filter=search_request.video_type_filter,
                release_date_filter=search_request.release_date_filter,
            )

    # In the instance of an error, we'll log the error and return it
    except Exception as e:

        # Log the error
        logger.error(f"Error with search request: {e}")

        # Log the traceback
        logger.error(traceback.format_exc())

        # Return the error
        return {"error": str(e)}


@app.get("/table_metadata")
def table_metadata():
    """
    This endpoint returns metadata about the table.
    """

    # Use the retrieve_table_size_metadata function to get the table metadata
    video_metadata_df = search_utils.retrieve_table_size_metadata()

    # Return a dictionary with the metadata
    return video_metadata_df.to_dict(orient="records")
