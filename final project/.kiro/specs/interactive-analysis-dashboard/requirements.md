# Requirements Document

## Introduction

The Interactive Analysis Dashboard extends the existing Multiple Disease Prediction System web application with a visual analysis layer. After patients receive their prediction results, the dashboard provides transparency into the machine learning pipeline by showing model evaluation graphs (confusion matrices, ROC curves, precision-recall curves, PCA variance plots) and an animated step-by-step breakdown of the prediction process. This makes the prediction experience more interactive, educational, and trustworthy for end users.

## Glossary

- **Dashboard**: The main web interface served by the Flask application at localhost:5000 after user login
- **Analysis_Panel**: A UI section within the Dashboard that displays model evaluation visualizations and prediction process explanations
- **Prediction_Pipeline**: The sequence of steps (imputation, scaling, PCA transformation, model inference) applied to patient input data to produce a disease prediction
- **Evaluation_Graph**: A pre-generated PNG image stored in graphs/current/ that visualizes model performance (confusion matrix, ROC curve, precision-recall curve, PCA variance plot, accuracy summary)
- **Process_Visualizer**: A UI component that shows an animated step-by-step breakdown of what happens during prediction
- **Graph_Viewer**: A UI component that renders Evaluation_Graph images with descriptions for a selected disease
- **Disease_Selector**: A UI control that allows the user to choose which disease model's analysis to view

## Requirements

### Requirement 1: Display Model Evaluation Graphs

**User Story:** As a user, I want to view model evaluation graphs for each disease, so that I can understand how well the prediction models perform.

#### Acceptance Criteria

1. WHEN the user selects a disease from the Disease_Selector, THE Graph_Viewer SHALL display the confusion matrix, ROC curve, precision-recall curve, and PCA variance plot for that disease within 2 seconds of the selection event.
2. THE Graph_Viewer SHALL load Evaluation_Graph images from the graphs/current/ directory via a Flask static file route, using the naming convention {disease}_{graph_type}.png where disease is one of heart, kidney, liver, thyroid, or diabetes, and graph_type is one of confusion_matrix, roc_curve, precision_recall_curve, or pca_variance.
3. IF none of the four expected Evaluation_Graph files exist for the selected disease, THEN THE Graph_Viewer SHALL display a message indicating that graphs are not yet available for that disease.
4. IF some but not all of the four expected Evaluation_Graph files exist for the selected disease, THEN THE Graph_Viewer SHALL display the available graphs and show a message in place of each missing graph indicating that the specific graph is not available.
5. THE Graph_Viewer SHALL display a text description of 1 to 3 sentences beneath each Evaluation_Graph explaining what the graph represents and how to interpret it.

### Requirement 2: Show Accuracy Summary Overview

**User Story:** As a user, I want to see a summary comparison of all model accuracies, so that I can quickly assess which diseases have the most reliable predictions.

#### Acceptance Criteria

1. WHEN the user opens the Analysis_Panel, THE Dashboard SHALL display the accuracy_summary.png graph showing all five disease model accuracies compared against the 90% target threshold.
2. IF the accuracy_summary.png file is unavailable, THEN THE Dashboard SHALL display a message indicating that the accuracy summary graph is not available.
3. WHEN the metrics_summary.csv file is available, THE Dashboard SHALL display a summary table showing each disease model's name, algorithm, accuracy, precision, recall, and F1 score with numeric values rounded to 2 decimal places.
4. IF the metrics_summary.csv file is unavailable, THEN THE Dashboard SHALL display a message indicating that metrics data is not available.

### Requirement 3: Visualize the Prediction Pipeline Steps

**User Story:** As a user, I want to see what happens to my input data during the prediction process, so that I understand how the system arrives at its result.

#### Acceptance Criteria

1. WHEN a prediction is completed, THE Process_Visualizer SHALL display a step-by-step breakdown showing: data input, missing value imputation, feature scaling, PCA dimensionality reduction, and model inference.
2. THE Process_Visualizer SHALL show each Prediction_Pipeline step sequentially with a 500ms animated transition between steps.
3. FOR each Prediction_Pipeline step, THE Process_Visualizer SHALL display a plain-language explanation of 1 to 2 sentences describing what that step does to the data.
4. WHEN the PCA step is shown, THE Process_Visualizer SHALL indicate the number of original features and the number of PCA components retained.
5. WHEN the model inference step is shown, THE Process_Visualizer SHALL display the model type used and the final prediction result.
6. IF the prediction request fails or returns an error, THEN THE Process_Visualizer SHALL not be displayed and the error message SHALL be shown in the result area.

### Requirement 4: Analysis Panel Navigation

**User Story:** As a user, I want to easily switch between the prediction form and the analysis dashboard, so that I can use both features without losing context.

#### Acceptance Criteria

1. WHEN the user is on the Dashboard, THE Dashboard SHALL display a navigation control to switch between the prediction form view and the Analysis_Panel view, with exactly one view visible at a time.
2. WHEN the Dashboard is first displayed after login, THE Dashboard SHALL show the prediction form view as the default active view.
3. WHEN the user switches from the prediction form view to the Analysis_Panel view and back, THE Dashboard SHALL preserve all previously entered patient data input field values in the prediction form so that no re-entry is required.
4. IF an unauthenticated user attempts to access the Analysis_Panel, THEN THE Dashboard SHALL keep the login page displayed and not render the Analysis_Panel.
5. THE Analysis_Panel SHALL be accessible only after login authentication succeeds.

### Requirement 5: Disease-Specific Analysis Selection

**User Story:** As a user, I want to select a specific disease to view its detailed analysis, so that I can focus on the disease prediction that interests me most.

#### Acceptance Criteria

1. THE Disease_Selector SHALL present all five diseases (heart, kidney, liver, thyroid, diabetes) as individually selectable options, each labeled with its display name.
2. WHEN a disease is selected from the Disease_Selector, THE Analysis_Panel SHALL update to show only the four Evaluation_Graph images (confusion matrix, ROC curve, precision-recall curve, PCA variance plot) and the numeric metrics (accuracy, precision, recall, F1 score) for that disease.
3. THE Disease_Selector SHALL visually indicate which disease is currently selected.
4. IF no specific disease is selected, THEN THE Disease_Selector SHALL display the overall accuracy summary graph and the metrics summary table as the default view.
5. WHEN a different disease is selected while the Analysis_Panel is already showing another disease's analysis, THE Analysis_Panel SHALL replace the previous disease content with the newly selected disease's Evaluation_Graph images and metrics within 2 seconds.

### Requirement 6: Serve Graph Images via Flask

**User Story:** As a developer, I want the Flask application to serve graph image files, so that the frontend can display evaluation plots without requiring an external file server.

#### Acceptance Criteria

1. THE Dashboard SHALL expose a Flask route at /graphs/<filename> that serves PNG files from the graphs/current/ directory.
2. WHEN a request is made for a graph file that does not exist, THE Dashboard SHALL return an HTTP 404 response.
3. THE Dashboard SHALL serve graph images with the correct image/png content type header.
4. THE Dashboard SHALL reject requests containing path traversal characters (../) and return an HTTP 400 response.

### Requirement 7: Display Model Metadata

**User Story:** As a user, I want to see which algorithm was used and how many features were processed for each disease, so that I have full transparency into the prediction system.

#### Acceptance Criteria

1. WHEN a disease is selected, THE Analysis_Panel SHALL display the model algorithm name, number of original features, number of PCA components, and overall accuracy (displayed as a percentage rounded to one decimal place) for that disease.
2. THE Analysis_Panel SHALL retrieve model metadata first from the metrics_summary.csv file; IF the CSV file is missing or does not contain a row for the selected disease, THEN THE Analysis_Panel SHALL fall back to reading from the stored metrics pickle file for that disease.
3. IF neither the metrics_summary.csv row nor the metrics pickle file is available for a disease, THEN THE Analysis_Panel SHALL display a message indicating that the model has not been trained yet.
4. WHEN model metadata is successfully loaded, THE Analysis_Panel SHALL display all four metadata fields (algorithm name, original features count, PCA components count, accuracy) together within 2 seconds of disease selection.
