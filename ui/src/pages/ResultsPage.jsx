/**
 * This file is the Results Page for the React application.
 */

/**
 * =====
 * SETUP
 * =====
 * Below, I've got some import statements.
 */

// Import statements
import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { Loader } from "@mantine/core";
import { performSearch } from "../api";
import SearchResult from "../components/SearchResult";
import { useSearchSettingsStore } from "../stores";

/**
 * ========
 * MAIN PAGE
 * ========
 * The code below will define the main function that will be called when this page is rendered
 */

function ResultsPage() {
  // Create a state that'll hold the search results
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState(null);
  const [queryInProgress, setQueryInProgress] = useState(false);

  // Grab the query parameters from the URL
  const location = useLocation();
  const urlSearchParams = new URLSearchParams(location.search);
  const query_params = Object.fromEntries(urlSearchParams);

  // Grab various parameters from the search settings store
  const neuralStrength = useSearchSettingsStore(
    (state) => state.neuralStrength
  );
  const keywordStrength = useSearchSettingsStore(
    (state) => state.keywordStrength
  );
  const reviewScoreRange = useSearchSettingsStore(
    (state) => state.reviewScoreRange
  );
  const videoTypeFilter = useSearchSettingsStore(
    (state) => state.videoTypeFilter
  );
  const releaseDateFilter = useSearchSettingsStore(
    (state) => state.releaseDateFilter
  );

  // This effect will fetch the search results when the query_params contain a query
  useEffect(() => {
    const fetchData = async () => {
      // If the query_params are null / undefined, or there's currently a query in progress, stop
      if (
        !query_params.query ||
        query_params.query.trim() === "" ||
        queryInProgress
      ) {
        setLoading(false);
        return;
      }

      setQueryInProgress(true);

      // Try and run the search
      try {
        const data = await performSearch(
          query_params.query,
          neuralStrength,
          keywordStrength,
          reviewScoreRange,
          videoTypeFilter,
          releaseDateFilter
        );

        // Check if data is a string and try to parse it
        let parsedData;
        if (typeof data === "string") {
          parsedData = JSON.parse(data);
        } else {
          parsedData = data;
        }

        setResults(parsedData);
        setLoading(false);
      } catch (error) {
        // If we run into an error, log it and stop loading
        console.error("Error during search:", error);
        setResults("ERROR");
        setLoading(false);
      } finally {
        setQueryInProgress(false);
      }
    };

    fetchData();
  }, [query_params.query]);

  return (
    <div>
      <h1>Results Page</h1>
      {loading ? (
        <div>
          <Loader />
          <p>Your results are coming soon...</p>
        </div>
      ) : Array.isArray(results) ? (
        results.map((result, index) => (
          <SearchResult key={index} result={result} />
        ))
      ) : (
        <div>There was an error with your search</div>
      )}
    </div>
  );
}

export default ResultsPage;
