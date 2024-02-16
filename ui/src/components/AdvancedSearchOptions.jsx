/**
 * This file contains the AdvancedSearchOptions component. It will control a couple of
 * different input fields that will allow the user to specify more advanced search options.
 * I've split it into its own component to keep the SearchPage component clean and easy to
 * understand.
 */

/**
 * =====
 * SETUP
 * =====
 * Below, I've got some import statements.
 */

// Import statements
import React from "react";
import { Grid, Paper, Text } from "@mantine/core";

/**
 * =======================
 * AdvancedSearchOptions()
 * =======================
 * The code below will define the main function that will be called when this component is rendered
 */

function AdvancedSearchOptions({ result }) {
  return (
    <Grid>
      <Grid.Col span={5}>Release Date (description)</Grid.Col>
      <Grid.Col span={7}>Release Date (controls)</Grid.Col>
        <Grid.Col span={5}>Video Type (description)</Grid.Col>
        <Grid.Col span={7}>Video Type (controls)</Grid.Col>
        <Grid.Col span={5}>Review Score (description)</Grid.Col>
        <Grid.Col span={7}>Review Score (controls)</Grid.Col>
        <Grid.Col span={5}>Search Type (description)</Grid.Col>
        <Grid.Col span={7}>Search Type (controls)</Grid.Col>
    </Grid>
  );
}

// Export the AdvancedSearchOptions component

export default AdvancedSearchOptions;
