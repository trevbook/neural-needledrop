/**
 * This file contains various functions that will ping the API.
 */

/**
 * =====
 * SETUP
 * =====
 * Below, I've got some import statements.
 */

// Import statements
import axios from "axios";

// Set the base URL for the API
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
console.log("API_BASE_URL:", API_BASE_URL);

export const performSearch = async (
  searchQuery,
  neuralStrength = 0.5,
  keywordStrength = 0.5,
  review_score_filter = null,
  video_type_filter = null,
  release_date_filter = null
) => {
  try {
    // Log all of the parameters to the console
    console.log("RUNNING SEARCH WITH THE FOLLOWING PARAMETERS:");
    console.log("Search query:", searchQuery);
    console.log("Neural strength:", neuralStrength);
    console.log("Keyword strength:", keywordStrength);
    console.log("Review score filter:", review_score_filter);
    console.log("Video type filter:", video_type_filter);
    console.log("Release date filter:", release_date_filter);
    console.log();

    const response = await axios.post(`${API_BASE_URL}/search`, {
      query: searchQuery,
      neural_search_strength: neuralStrength,
      keyword_search_strength: keywordStrength,
      review_score_filter: review_score_filter,
      video_type_filter: video_type_filter,
      release_date_filter: release_date_filter,
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};
