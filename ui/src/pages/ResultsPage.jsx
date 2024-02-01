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
import React from "react";
import { useLocation } from "react-router-dom";

/**
 * ========
 * MAIN PAGE
 * ========
 * The code below will define the main function that will be called when this page is rendered
 */

function ResultsPage() {

  // Grab the query parameters from the URL
  const location = useLocation();
  const urlSearchParams = new URLSearchParams(location.search);
  const query_params = Object.fromEntries(urlSearchParams);

  return (
    <div>
      <h1>Results Page</h1>
      <p>{JSON.stringify(query_params)}</p>
    </div>
  );
}

export default ResultsPage;
