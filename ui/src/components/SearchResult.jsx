/**
 * This file contains the SearchResult component, which will correspond with a single
 * search result. It will show off different pieces of metadata about the video, as well
 * as the relevant snippets of text that match the search query.
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
 * ==============
 * SearchResult()
 * ==============
 * The code below will define the main function that will be called when this component is rendered
 */

function SearchResult({ result }) {
  // Validate the result object
  if (!result || typeof result !== "object") {
    return <div>Error: Invalid result data</div>;
  }

  // Extract the relevant data
  const {
    url,
    title,
    length,
    channel_name,
    short_description,
    view_ct,
    large_thumbnail_url,
    review_score,
    publish_date,
    top_segment_chunks,
  } = result;

  // Format the data as needed
  const formattedPublishDate = new Date(publish_date).toLocaleDateString();

  //   return <div style={{ marginBottom: "40px" }}>{JSON.stringify(result)}</div>;

  return (
    <Paper style={{ marginBottom: "40px", padding: "xs" }}>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        <div className="thumbnail-container" style={{ marginRight: "20px",}}>
          <img src={large_thumbnail_url} alt={title} />
        </div>
        <div>
          <Text>
            <a href={url}>{title}</a>
          </Text>
          <Text>Channel: {channel_name}</Text>
          <Text>Length: {length}</Text>
          <Text>Views: {view_ct}</Text>
          <Text>Review Score: {review_score}</Text>
          <Text>Publish Date: {formattedPublishDate}</Text>
          <Text>Top segments:</Text>
          {top_segment_chunks ? (
            <ul>
              {top_segment_chunks.map((segment, index) => (
                <li key={index}>{segment}</li>
              ))}
            </ul>
          ) : (
            <Text>Error: Could not retrieve top segment chunks</Text>
          )}
        </div>
      </div>
    </Paper>
  );
}

// Export the SearchResult component

export default SearchResult;
