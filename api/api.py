"""
This file contains the API for the Neural Needledrop app. The API is built using FastAPI. 
"""

# =====
# SETUP
# =====
# Below, I'm going to set up the file.

# Import statements
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

# Importing custom modules
import utils.search as search_utils

# Setting up the FastAPI app
app = FastAPI()

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========
# ENDPOINTS
# =========
# Next, we're going to define each of the app's endpoints


@app.post("/search")
def search(query: str, search_type: str = "neural"):
    """
    This is the search endpoint.
    """

    # If the user wants neural search, then we'll run this.
    if search_type == "neural":
        return search_utils.neural_search(
            query=query,
        )
