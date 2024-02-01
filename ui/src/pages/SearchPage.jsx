/**
 * This file is the Search Page for the React application.
 */

/**
 * =====
 * SETUP
 * =====
 * Below, I've got some import statements.
 */

import React, { useState, useEffect } from "react";
import { Center, TextInput, Button } from "@mantine/core";
import appLogo from "../assets/app-logo.png"; // Make sure the path is correct
import { performSearch } from "../api";

/**
 * ========
 * MAIN PAGE
 * ========
 * The code below will define the main function that will be called when this page is rendered
 */

function SearchPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [submitSearch, setSubmitSearch] = useState(false);

  useEffect(() => {
    if (submitSearch && searchTerm) {
        console.log(searchTerm)
      performSearch(searchTerm).then((results) => {
        console.log(results);
        setSubmitSearch(false); // Reset the submit state after search
      });
    }
  }, [submitSearch, searchTerm]);

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (searchTerm) {
      setSubmitSearch(true);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "70vh",
      }}
    >
      <Center style={{ flexDirection: "column" }}>
        <img
          src={appLogo}
          alt="Neural Needledrop Logo"
          style={{ height: `${15}vh` }}
        />
        <h1>Neural Needledrop</h1>
        <form onSubmit={handleSubmit}>
          <TextInput
            value={searchTerm}
            onChange={handleSearchChange}
            placeholder="Enter search term"
            required
          />
          <Button type="submit" disabled={!searchTerm}>
            Search
          </Button>
        </form>
      </Center>
    </div>
  );
}

export default SearchPage;
