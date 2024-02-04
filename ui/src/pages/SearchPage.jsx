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
import { Search } from "tabler-icons-react";
import { useNavigate  } from "react-router-dom";

/**
 * ========
 * MAIN PAGE
 * ========
 * The code below will define the main function that will be called when this page is rendered
 */

function SearchPage() {
  // This state will hold the search query
  const [searchQuery, setSearchQuery] = useState("");

  const navigate = useNavigate ();

  const handleSearch = (event) => {
    event.preventDefault();
    navigate(`/results?query=${searchQuery}`);
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
      <Center style={{ flexDirection: "column", width: "80%" }}>
        <img
          src={appLogo}
          alt="Neural Needledrop Logo"
          style={{ height: `${15}vh` }}
        />
        <div style={{ marginBottom: "5px" }}>
          <h1>Neural Needledrop</h1>
        </div>
        <form className="formStyle" onSubmit={handleSearch}>
          <TextInput
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Enter search query"
            required
            className="textInputStyle"
          />
          <Button className="buttonStyle" type="submit" disabled={!searchQuery}>
            <Search size={20} />
          </Button>
        </form>
      </Center>
    </div>
  );
}

export default SearchPage;
