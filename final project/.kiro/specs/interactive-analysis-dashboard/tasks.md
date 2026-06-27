# Implementation Plan: Interactive Analysis Dashboard

## Overview

This plan implements the Interactive Analysis Dashboard feature by extending the existing Flask application (`app.py`) with new backend routes for serving graphs and metrics, then adding the frontend Analysis Panel UI (inline HTML/CSS/JS) with tab navigation, disease selector, graph viewer, metrics table, and prediction pipeline visualizer. All implementation follows the single-file Flask application pattern already established.

## Tasks

- [x] 1. Implement backend Flask routes for graph serving and metrics API
  - [x] 1.1 Add the `/graphs/<filename>` route to app.py
    - Implement path traversal validation (reject `..`, `/`, `\` → HTTP 400)
    - Only serve `.png` or `.csv` files from `graphs/current/`
    - Return 404 for non-existent files
    - Set `Content-Type: image/png` for PNG files
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 1.2 Add the `/api/metrics` route to app.py
    - Parse `graphs/current/metrics_summary.csv` as primary data source
    - Fall back to `models/{disease}_metrics.pkl` when CSV row is missing
    - Support optional `?disease=` query parameter to filter results
    - Return JSON with `{ok, metrics}` structure per design schema
    - Round numeric values (accuracy, precision, recall, F1) to appropriate precision
    - Return `{"ok": false, "message": "..."}` when data is unavailable
    - _Requirements: 2.3, 7.1, 7.2, 7.3, 7.4_

  - [ ]* 1.3 Write property tests for graph route (Properties 8, 9)
    - **Property 8: Non-existent graph filename returns 404**
    - **Property 9: Path traversal in filename returns 400**
    - Use Hypothesis with Flask test client
    - **Validates: Requirements 6.2, 6.4**

  - [ ]* 1.4 Write property tests for metrics formatting (Properties 3, 10, 11)
    - **Property 3: Metrics table values are rounded to 2 decimal places**
    - **Property 10: Accuracy displayed as percentage with one decimal place**
    - **Property 11: Metrics data source priority (CSV over pickle)**
    - Use Hypothesis with Flask test client
    - **Validates: Requirements 2.3, 7.1, 7.2**

- [x] 2. Implement frontend navigation and Analysis Panel structure
  - [x] 2.1 Add tab navigation UI to the WEB_PAGE HTML
    - Add "Predict" and "Analysis" tab buttons to the dashboard header
    - Create container divs for prediction form view and analysis panel view
    - Implement JavaScript toggle logic: exactly one view visible at a time via `display:block/none`
    - Set prediction form as the default active view
    - Ensure tab switching preserves all input field values (DOM not destroyed)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 2.2 Add Disease Selector component to the Analysis Panel
    - Render 5 disease buttons (heart, kidney, liver, thyroid, diabetes) plus "Overview" button
    - Set "Overview" as the default selected option
    - Add visual indicator (active class) for currently selected disease
    - Wire click handlers to update the Graph Viewer and metadata display
    - _Requirements: 5.1, 5.3, 5.4_

  - [ ]* 2.3 Write property test for view switching (Property 5)
    - **Property 5: Exactly one view is visible at any time**
    - Verify that for any sequence of tab clicks, exactly one view has `display:block`
    - **Validates: Requirements 4.1**

- [x] 3. Implement Graph Viewer and accuracy summary display
  - [x] 3.1 Implement the Graph Viewer component
    - Create a 2x2 grid layout container for graph images
    - Generate `<img>` elements with `src="/graphs/{disease}_{graph_type}.png"` URLs
    - Add `onerror` handlers to replace broken images with "Graph not available" placeholder
    - Display 1-3 sentence description text beneath each graph image
    - When all 4 graphs are missing, display "Graphs not yet available" message
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 3.2 Implement accuracy summary overview (default Analysis Panel view)
    - Display `accuracy_summary.png` via the graph route when Overview is selected
    - Show fallback message if accuracy summary image is unavailable
    - Fetch metrics from `/api/metrics` and render summary table
    - Display disease name, algorithm, accuracy, precision, recall, F1 in the table
    - Format numeric values to 2 decimal places
    - Show fallback message if metrics data is unavailable
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 3.3 Write property test for graph URL generation (Property 1)
    - **Property 1: Graph URL generation follows naming convention**
    - For any valid disease and graph type, verify URL matches `/graphs/{disease}_{graph_type}.png`
    - **Validates: Requirements 1.1, 1.2**

  - [ ]* 3.4 Write property test for partial graph availability (Property 2)
    - **Property 2: Partial graph availability renders correctly**
    - For any subset of 4 graph types, verify total rendered items equals 4
    - **Validates: Requirements 1.4**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Model Metadata display and disease-specific analysis
  - [x] 5.1 Add model metadata card to the Analysis Panel
    - Display algorithm name, original feature count, PCA components count, and accuracy percentage
    - Format accuracy as percentage with 1 decimal place (e.g., "96.9%")
    - Fetch metadata from `/api/metrics?disease={selected}` endpoint
    - Show "Model not trained yet" message when no data available
    - Ensure metadata loads within 2 seconds of disease selection
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 5.2 Wire disease selection to update Graph Viewer and metadata
    - On disease selection, update all graph `src` attributes to the new disease
    - Fetch and display updated metrics for the selected disease
    - Replace previous disease content with newly selected disease data
    - Ensure content updates complete within 2 seconds
    - _Requirements: 5.2, 5.5, 1.1_

  - [ ]* 5.3 Write property test for disease selection isolation (Property 7)
    - **Property 7: Disease selection isolation**
    - Verify all graph `src` attributes contain the selected disease key and no other disease key
    - **Validates: Requirements 5.2**

- [x] 6. Implement Prediction Pipeline Process Visualizer
  - [x] 6.1 Add Process Visualizer component to the prediction form view
    - Create a container that renders the 5 pipeline steps (data input, imputation, scaling, PCA, inference)
    - Show each step sequentially with 500ms CSS fade-in transition between steps
    - Display icon + title + 1-2 sentence plain-language description for each step
    - Trigger the visualizer only after a successful prediction response
    - Hide the visualizer and show error message on prediction failure
    - _Requirements: 3.1, 3.2, 3.3, 3.6_

  - [x] 6.2 Add dynamic metadata interpolation to pipeline steps
    - Interpolate PCA step with original features count and retained PCA components count
    - Interpolate inference step with model type name and final prediction result
    - Source metadata values from the prediction response or `/api/metrics` endpoint
    - _Requirements: 3.4, 3.5_

  - [ ]* 6.3 Write property test for pipeline step interpolation (Property 4)
    - **Property 4: Pipeline step descriptions interpolate dynamic metadata**
    - Verify PCA step contains original_features and pca_components values
    - Verify inference step contains model_name and result values
    - **Validates: Requirements 3.4, 3.5**

- [x] 7. Integration, styling, and final wiring
  - [x] 7.1 Add CSS styles for the Analysis Panel layout
    - Style tab navigation buttons with active/inactive states
    - Style Disease Selector buttons with hover and selected states
    - Style 2x2 graph grid for responsive layout
    - Style metrics table, metadata card, and Process Visualizer
    - Style fallback/error messages
    - Ensure all styling is inline within the `WEB_PAGE` `<style>` block
    - _Requirements: 1.1, 5.3_

  - [x] 7.2 Wire prediction response to include metadata for Process Visualizer
    - Modify `/predict` route response to include `model_name`, `original_features`, and `pca_components`
    - Ensure the prediction form JavaScript uses this metadata to populate the Process Visualizer
    - _Requirements: 3.4, 3.5_

  - [ ]* 7.3 Write unit tests for end-to-end integration
    - Test login → navigate to analysis → select disease → verify graph URLs correct
    - Test prediction flow → verify Process Visualizer data populated
    - Test fallback behaviors for missing files
    - _Requirements: 1.1, 3.1, 4.1, 6.1_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All implementation stays within the single-file Flask application pattern (`app.py`)
- Frontend code uses vanilla JavaScript and CSS (no external libraries)
- The `/api/metrics` endpoint provides data for both the metrics table and the model metadata card

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "1.4", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1", "3.2"] },
    { "id": 3, "tasks": ["3.3", "3.4", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3"] },
    { "id": 5, "tasks": ["6.1"] },
    { "id": 6, "tasks": ["6.2", "6.3"] },
    { "id": 7, "tasks": ["7.1", "7.2"] },
    { "id": 8, "tasks": ["7.3"] }
  ]
}
```
