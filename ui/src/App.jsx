/**
 * This file is the main entry point for the React application.
 */

/**
 * =====
 * SETUP
 * =====
 * Below, I've got some import statements.
 */

import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// Imports for Mantine
import "@mantine/core/styles.css";
import { MantineProvider } from "@mantine/core";

// Importing different pages
import SearchPage from "./pages/SearchPage";
import ResultsPage from "./pages/ResultsPage";

/**
 * =====
 * APP()
 * =====
 * The code below will define the main function that will be called when this page is rendered
 */

function App() {
  return (
    <MantineProvider>
      <Router>
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </Router>
    </MantineProvider>
  );
}

export default App;
