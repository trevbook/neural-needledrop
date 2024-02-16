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
import {
  Grid,
  MultiSelect,
  Paper,
  RangeSlider,
  Slider,
  Text,
} from "@mantine/core";
import { MonthPickerInput } from "@mantine/dates";

import { useSearchSettingsStore } from "../stores";

/**
 * =======================
 * AdvancedSearchOptions()
 * =======================
 * The code below will define the main function that will be called when this component is rendered
 */

function AdvancedSearchOptions({ result }) {
  // Grab some setters from the search settings store
  const setReviewScoreRange = useSearchSettingsStore(
    (state) => state.setReviewScoreRange
  );
  const setNeuralStrength = useSearchSettingsStore(
    (state) => state.setNeuralStrength
  );
  const setKeywordStrength = useSearchSettingsStore(
    (state) => state.setKeywordStrength
  );
  const setVideoTypeFilter = useSearchSettingsStore(
    (state) => state.setVideoTypeFilter
  );
  const setReleaseDateFilter = useSearchSettingsStore(
    (state) => state.setReleaseDateFilter
  );

  // Get the reviewScoreRange from the search settings store
  const reviewScoreRange = useSearchSettingsStore(
    (state) => state.reviewScoreRange
  );
  const releaseDateFilter = useSearchSettingsStore(
    (state) => state.releaseDateFilter
  );

  // Declare a function that will be called when the search type slider changes
  const onSearchTypeSliderChange = (value) => {
    const neuralStrength = 1 - value / 100;
    const keywordStrength = value / 100;

    setNeuralStrength(neuralStrength);
    setKeywordStrength(keywordStrength);
  };

  // Define the valid video types
  const videoTypes = [
    { value: "album_review", label: "Album Review" },
    { value: "ep_review", label: "EP Review" },
    { value: "mixtape_review", label: "Mixtape Review" },
    { value: "track_review", label: "Track Review" },
    { value: "weekly_track_roundup", label: "Weekly Track Roundup" },
    { value: "yunoreview", label: "Y U No Review" },
    { value: "vinyl_update", label: "Vinyl Update" },
    { value: "tnd_podcast", label: "TND Podcast" },
    { value: "misc", label: "Misc" },
  ];

  return (
    <div>
      <Grid className="search-option-row">
        <Grid.Col span={5}>
          <Text className="control-row-title">
            Release Date
          </Text>
          <Text className="control-row-description">
            Select the range of release dates you are interested in.
          </Text>
        </Grid.Col>
        <Grid.Col span={7}>
          <div style={{ width: "80%", margin: "auto" }}>
            <MonthPickerInput
              type="range"
              placeholder="Pick dates range"
              onChange={setReleaseDateFilter}
            />
          </div>
        </Grid.Col>
      </Grid>
      <Grid className="search-option-row">
        <Grid.Col span={5}>
          <Text className="control-row-title">
            Video Type
          </Text>
          <Text className="control-row-description">
            Select the type of videos you are interested in.
          </Text>
        </Grid.Col>
        <Grid.Col span={7}>
          <div style={{ width: "80%", margin: "auto" }}>
            <MultiSelect
              data={videoTypes}
              placeholder="Select video types"
              onChange={(values) => {
                setVideoTypeFilter(values);
              }}
            />
          </div>
        </Grid.Col>
      </Grid>
      <Grid className="search-option-row">
        <Grid.Col span={5}>
          <Text className="control-row-title">
            Review Score
          </Text>
          <Text className="control-row-description">
            Select the range of review scores you are interested in.
          </Text>
        </Grid.Col>
        <Grid.Col span={7}>
          <div style={{ width: "80%", margin: "auto" }}>
            <RangeSlider
              min={0}
              max={10}
              step={1}
              minRange={0}
              marks={[
                { value: 0, label: "0" },
                { value: 5, label: "5" },
                { value: 10, label: "10" },
              ]}
              defaultValue={reviewScoreRange}
              values={reviewScoreRange}
              onChange={(values) => {
                setReviewScoreRange(values);
              }}
            />
          </div>
        </Grid.Col>
      </Grid>
      <Grid className="search-option-row">
        <Grid.Col span={5}>
          <Text className="control-row-title">
            Search Type
          </Text>
          <Text className="control-row-description">
            Select the type of search you would like to perform.
          </Text>
        </Grid.Col>
        <Grid.Col span={7}>
          <div style={{ width: "80%", margin: "auto" }}>
            <Slider
              className="search-type-control"
              color="#2e353b"
              marks={[
                { value: 0, label: "Neural" },
                { value: 50, label: "Hybrid" },
                { value: 100, label: "Keyword" },
              ]}
              showLabelOnHover={false}
              defaultValue={50}
              thumbSize={20}
              onChange={onSearchTypeSliderChange}
            ></Slider>
          </div>
        </Grid.Col>
      </Grid>
    </div>
  );
}

// Export the AdvancedSearchOptions component

export default AdvancedSearchOptions;
