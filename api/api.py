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

# Create a cache object with a TTL (time-to-live) of 30 minutes (1800 seconds)
# and a maximum of 1000 items
cache = TTLCache(maxsize=10, ttl=1800)

# Importing custom modules
import utils.search as search_utils

# Setting up the FastAPI app
app = FastAPI()

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
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
    search_type: str = "neural"


# The cache key will be the 'query' from the function arguments
def cache_key(*args, **kwargs):
    return kwargs["search_request"].query


@app.post("/search")
@cached(cache, key=cache_key)
def search(search_request: SearchRequest):
    """
    This is the search endpoint.
    """

    # Print the search request
    print(f"\nRECIEVED REQUEST:\n{search_request}\n")

    # If the user wants neural search, then we'll run this.
    if search_request.search_type == "neural":
        return search_utils.neural_search(
            query=search_request.query,
        )

    # If the user wants keyword search, then we'll run this.
    if search_request.search_type == "keyword":
        return search_utils.keyword_search(
            query=search_request.query,
        )

    # If the user wants hybrid search, then we'll run this.
    if search_request.search_type == "hybrid":

        # Determine how many individual words are in the query
        num_words = len(search_request.query.split())

        # If there are less than 3 words, we'll weigh the keyword search more heavily
        if num_words < 3:
            keyword_weight = 1
        else:
            keyword_weight = 0.8

        return search_utils.hybrid_search(
            query=search_request.query,
            max_video_per_search_method=5,
            max_results=10,
            keyword_weight=keyword_weight,
            neural_weight=1,
        )
