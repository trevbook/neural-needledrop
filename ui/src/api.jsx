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

export const performSearch = async (searchQuery) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/search`, {
      query: searchQuery,
      search_type: "neural",
    });
    console.log("Search response:", response.data);
    return response.data;
  } catch (error) {
    console.error("Error during search:", error.response || error);
    throw error;
  }
};
