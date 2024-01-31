"""
This file contains various utility functions related to OpenAI.
"""

# =====
# SETUP
# =====
# The code below will set up the rest of the file.

# Import statements
import openai
import math
import traceback
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_fixed, stop_after_attempt
import numpy as np

# Declaring some constants that'll help with the rate limiting
DEFAULT_REQUESTS_PER_MINUTE = 3000
DEFAULT_TOKENS_PER_MINUTE = 1000000
CHAR_PER_TOKEN_ESTIMATE = 3.5

# ======================
# GENERAL-PUROSE METHODS
# ======================
# Below, I've got a number of methods that are useful for general purposes.


# Helper function to calculate delay between requests
def calculate_delay(requests_per_minute, n_workers, delay_multiplier=2):
    return ((60 / requests_per_minute) * n_workers) * delay_multiplier


def approximate_token_amt(text):
    """
    This helper method will try and estimate the number of tokens in a particular piece of text.
    """
    return math.ceil(len(text) / CHAR_PER_TOKEN_ESTIMATE)


def embed_text(text, model="text-embedding-3-small", max_tokens_before_truncation=7500):
    """
    This method will embed a single piece of text, and return the embedding.
    """

    # We're going to wrap things in a try/except block so that we can catch any errors that occur
    try:
        # Try and approximate the number of tokens in the text
        token_amt = approximate_token_amt(text)

        # If the number of tokens is greater than the max, then we'll truncate the text
        if token_amt > max_tokens_before_truncation:
            # Truncate the text
            text = (
                text[: max_tokens_before_truncation * CHAR_PER_TOKEN_ESTIMATE] + "..."
            )

        # Use the OpenAI library to generate an embedding
        embedding_object = openai.embeddings.create(
            input=text, model="text-embedding-3-small"
        )

        # Extract the raw embedding
        raw_embedding = embedding_object.data[0].embedding

        # Return the raw embedding
        return raw_embedding

    # If we run into an Exception, then we'll raise our own custom Exception
    except Exception as e:
        max_text_length = 50
        truncated_text = (
            text[:max_text_length] + "..." if len(text) > max_text_length else text
        )
        raise Exception(
            f"Error embedding text '{truncated_text}': '{e}'\nTraceback is as follows:\n{traceback.format_exc()}"
        )


# The embed_text_list method
def embed_text_list(
    text_list,
    requests_per_minute=DEFAULT_REQUESTS_PER_MINUTE,
    model="text-embedding-3-small",
    max_workers=8,
    show_progress=True,
):
    """
    This method will embed a list of texts, and return a list of embeddings.
    It uses ThreadPoolExecutor and tenacity for parallel execution and retrying.
    """

    # Calculate the delay needed to respect the requests_per_minute limit
    delay = calculate_delay(requests_per_minute, n_workers=max_workers)

    # The wrapper function to embed text with retries
    @retry(wait=wait_fixed(delay * 2), stop=stop_after_attempt(3))
    def embed_with_retry(text):
        # Estimate the number of tokens in the text
        token_amt = approximate_token_amt(text)

        # Determine how long we should sleep, given the number of tokens and workers
        sleep_time_multiplier = 2
        sleep_time = (
            (token_amt / DEFAULT_TOKENS_PER_MINUTE)
            * 60
            * sleep_time_multiplier
            * max_workers
        ) + delay

        # Try and embed the text
        try:
            embedding = embed_text(text, model)
        except Exception:
            embedding = None

        # Now, sleep for the appropriate amount of time
        time.sleep(sleep_time)

        # Return the embedding
        return embedding

    # Use ThreadPoolExecutor to run embed_text in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Schedule the embed_text calls
        future_to_text = {
            executor.submit(embed_with_retry, text): text for text in text_list
        }

        # Initialize tqdm iterator to show progress
        futures_iterator = tqdm(
            as_completed(future_to_text),
            total=len(text_list),
            desc="Embedding Texts",
            disable=not show_progress,
        )

        # Collect the results maintaining the order
        results = []
        for future in futures_iterator:
            results.append(future.result())

    return results


def save_as_npy(embedding, file_name):
    """
    Save a list of floats as a .npy file.

    Args:
    embedding (list of float): The embedding (which is a list of floats) to save.
    file_name (str): The name of the file to create, including the .npy extension.
    """
    # Convert the list to a numpy array
    array = np.array(embedding, dtype=float)

    # Save the array as a .npy file
    np.save(file_name, array)
