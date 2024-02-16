import { create } from "zustand";

const useSearchSettingsStore = create((set) => ({
  // Set up the release_date_filter control
  releaseDateFilter: null,
  setReleaseDateFilter: (filter) => {
    set({ releaseDateFilter: filter });
  },

  // This will control the video type filter
  videoTypeFilter: null,
  setVideoTypeFilter: (filter) => {
    set({ videoTypeFilter: filter });
  },

  // This will control the range of review scores that are returned
  reviewScoreRange: [0, 10],
  setReviewScoreRange: (range) => {
    set({ reviewScoreRange: range });
  },

  // These settings will control the type of search that's performed
  neuralStrength: 0.5,
  keywordStrength: 0.5,
  setNeuralStrength: (strength) => set({ neuralStrength: strength }),
  setKeywordStrength: (strength) => set({ keywordStrength: strength }),
}));

export { useSearchSettingsStore };
