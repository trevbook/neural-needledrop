"""
This file contains various miscellaneous utility functions. 
"""

# =====
# SETUP
# =====
# The code below will set up the rest of the file.

# Import statements
import time
import random

# ============================
# MISCELLANEOUS HELPER METHODS
# ============================
# Below, I've written a number of miscellaneous helper methods.


def sleep_random_time(
    time_to_sleep,
    sleep_time_multiplier=2.5,
):
    """
    This method will sleep for a random amount of time,
    with the `time_to_sleep` as the lower bound, and the
    `time_to_sleep * sleep_time_multiplier` as the upper bound.
    """

    # Sleep for a random amount of time
    upper_bound = time_to_sleep * sleep_time_multiplier
    sleep_time = random.uniform(time_to_sleep, upper_bound)
    time.sleep(sleep_time)
