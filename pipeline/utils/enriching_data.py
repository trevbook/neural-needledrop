"""
This file contains a bunch of functions that're related to "enriching" the 
video metadata for Fantano's reviews. This includes things like classifying 
the type of video (e.g. album review, track review, etc.), extracting the
album name, etc.
"""

# =====
# SETUP
# =====
# This code will set up the rest of the file

# Import statements
import json
from pathlib import Path
from tqdm import tqdm
import re
from time import sleep


# # Import statements
# from Levenshtein import ratio
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

# # Setting up the API client
# spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())


# This method will attempt to determine whether a video is a review or not
def detect_review(title):
    # Determine if "album review" is a substring of the video title, and return a bool indicating that
    if title.lower().find("album review") == -1:
        return False
    else:
        return True


# This method will attempt to determine what type of video a particular video is
def classify_video_type(title):
    # Transform the video title to a lowercase string
    lowercase_video_title = title.lower()

    # Try and determine what type of video this one is by parsing the title
    if (
        lowercase_video_title.find("album review") != -1
        and lowercase_video_title.count("-") > 0
    ):
        return "album_review"
    elif lowercase_video_title.find("ep review") != -1:
        return "ep_review"
    elif lowercase_video_title.find("track review") != -1:
        return "track_review"
    elif lowercase_video_title.find("mixtape review") != -1:
        return "mixtape_review"
    elif (
        lowercase_video_title.find("yunoreview") != -1
        or lowercase_video_title.find("y u no review") != -1
    ):
        return "yunoreview"
    elif (
        lowercase_video_title.find("weekly track roundup") != -1
        or lowercase_video_title.find("best & worst tracks") != -1
    ):
        return "weekly_track_roundup"
    elif lowercase_video_title.find("tnd podcast #") != -1:
        return "tnd_podcast"
    elif lowercase_video_title.find("vinyl update") != -1:
        return "vinyl_update"
    else:
        return "misc"


# This method will try to extract the review score from a video's description using a regex.
# If successful, it'll return an int. If unsuccessful, it'll return None.
def extract_review_score(description):
    # Try to parse the review score from the video's description
    try:
        search = re.findall(r"[^0-9][0-9]{1,2}/10", description, re.IGNORECASE)
        return int(search[-1].split("/")[0])

    # Return None if we ran into an error
    except Exception as e:
        print(
            f"Ran into an error while extracting the review score from the description: '{e}'"
        )
        return None


# This method will try and extract the album title and artist name from a review's title
def extract_album_info(title):
    try:
        video_title = title.lower()
        single_dash = video_title.count("-") == 1
        if single_dash:
            artist, album_title = [
                x.strip()
                for x in video_title.split("album review")[0].strip().split("-")
            ]
        else:
            video_title = video_title.split("album review")[0]
            artist, album_title = [
                x.strip() for x in re.split(r"\s*-\s*", video_title, maxsplit=1)
            ]
        return {"artist": artist, "album_title": album_title}
    except:
        return {"artist": None, "album_title": None}


# def search_spotify_album_id(album_title, artist):
#     # Search Spotify for a particular album
#     try:
#         search_str = f"{album_title} {artist}".lower()
#         search_res = spotify.search(search_str, limit=1, type="album")
#         sleep(1)

#         # Extract some information from this Spotify search result
#         album_id = search_res["albums"]["items"][0]["id"]
#         spotify_res_artist = search_res["albums"]["items"][0]["artists"][0]["name"]
#         spotify_res_album_title = search_res["albums"]["items"][0]["name"]
#         spotify_res_search_str = (
#             f"{spotify_res_album_title} {spotify_res_artist}".lower()
#         )

#         # Determine how similar the result was to the search string
#         lev_sim = ratio(spotify_res_search_str.lower(), search_str)

#         # If the result is above a particular similarity, we're going to return that information
#         if lev_sim >= 0.8:
#             return album_id
#         else:
#             return None

#     # If we run into an Exception, return None
#     except:
#         return None
