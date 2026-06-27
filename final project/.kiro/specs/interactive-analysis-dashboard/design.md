# Design Document: Interactive Analysis Dashboard

## Overview

The Interactive Analysis Dashboard adds a visual analysis layer to the existing Multiple Disease Prediction System Flask application. It provides users with model evaluation graphs, accuracy summaries, a step-by-step prediction pipeline visualizer, and model metadata — all accessible through a tab-based navigation system within the existing single-page dashboard.

### Design Goals

- Extend the existing inline HTML (`WEB_PAGE` variable) with an Analysis Panel accessible via tab navigation
- Serve graph images through a new Flask route (`/graphs/<filename>`)
- Load metrics data from existing CSV and pickle files via a new API endpoint
- Animate the prediction pipeline steps client-side using vanilla JavaScript
- Maintain the single-file Flask application pattern

### Key Design Decisions

1. **Inline HTML extension**: The Analysis Panel UI is added directly to the `WEB_PAGE` string, keeping the single-file pattern intact.
2. **Client-side rendering**: Graph images are loaded as `<img>` tags pointing to the Flask graph route; metrics table is rendered from JSON fetched via a new `/api/metrics` endpoint.
3. **No external libraries**: All animations and UI interactions use vanilla CSS transitions and JavaScript `setTimeout`.
4. **Progressive loading**: Graphs load asynchronously; missing files show fallback messages rather than breaking the UI.

## Architecture

```mermaid
graph TD
    subgraph Flask Backend
        A[app.py] --> B[/ route - WEB_PAGE]
        A --> C[/graphs/filename route]
        A --> D[/api/metrics route]
        A --> E[/predict route]
        A --> F[/start-training route]
    end

    subgraph Static Assets
        G[graphs/current/*.png]
        H[graphs/current/metrics_summary.csv]
        I[models/*_metrics.pkl]
    end

    subgraph Client Browser
        J[Login Page]
        J -->|auth| K[Dashboard]
        K --> L[Prediction Form View]
        K --> M[Analysis Panel View]
        M --> N[Disease Selector]
        M --> O[Graph Viewer]
        M --> P[Metrics Table]
        M --> Q[Model Metadata Card]
        L --> R[Process Visualizer]
    end

    C --> G
    D --> H
    D --> I
    O -->|img src| C
```

### Request Flow

1. User logs in → Dashboard renders with Prediction Form as default view
2. User clicks "Analysis" tab → Analysis Panel shows with accuracy summary (default)
3. User selects a disease → Client fetches graphs via `/graphs/{disease}_{type}.png` and metrics via `/api/metrics`
4. User runs a prediction → Process Visualizer animates the pipeline steps

## Components and Interfaces

### Backend Components

#### 1. Graph Serving Route (`/graphs/<filename>`)

```python
@app.route("/graphs/<filename>")
def serve_graph(filename):
    """Serve PNG files from graphs/current/ directory.
    
    Args:
        filename: The PNG filename to serve (e.g., "heart_roc_curve.png")
    
    Returns:
        - 200 with image/png content type on success
        - 400 if path traversal detected
        - 404 if file not found
    """
```

**Validation rules:**
- Reject any filename containing `..` or `/` or `\` (return 400)
- Only serve files ending in `.png` or `.csv` (return 404 otherwise)
- Check file existence in `graphs/current/` (return 404 if missing)

#### 2. Metrics API Route (`/api/metrics`)

```python
@app.route("/api/metrics")
def get_metrics():
    """Return model metrics data as JSON.
    
    Query params:
        disease (optional): Filter to a specific disease
    
    Returns:
        JSON with structure:
        {
            "ok": true,
            "metrics": [
                {
                    "disease": "heart",
                    "display_name": "Heart Disease",
                    "model": "Random Forest",
                    "original_features": 13,
                    "pca_components": 11,
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0
                }, ...
            ]
        }
    """
```

**Data source priority:**
1. Parse `graphs/current/metrics_summary.csv` first
2. If CSV is missing or disease row absent, fall back to `models/{disease}_metrics.pkl`
3. If neither available, return `{"ok": false, "message": "..."}`

### Frontend Components

#### 3. Navigation Controls

Two tab buttons in the dashboard header:
- "Predict" tab (default active) → shows prediction form
- "Analysis" tab → shows Analysis Panel

Implementation: Toggle `display:none/block` on container divs. Input values persist since the DOM is not destroyed.

#### 4. Disease Selector

A row of 5 buttons (one per disease) plus an "Overview" button (default selected):
- Overview: Shows `accuracy_summary.png` and full metrics table
- Disease button: Shows 4 graphs + metadata card for that disease

#### 5. Graph Viewer

Container that renders up to 4 `<img>` elements in a 2x2 grid layout:
- Each image points to `/graphs/{disease}_{graph_type}.png`
- On load error (404), show a placeholder message
- Below each image, a static description paragraph

#### 6. Process Visualizer

Triggered after a successful prediction response:
- 5 steps shown sequentially with 500ms CSS fade-in transitions
- Each step: icon + title + 1-2 sentence description
- PCA step shows feature count reduction
- Final step shows model type and prediction result

### Interface Contracts

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/graphs/<filename>` | GET | filename in URL path | PNG binary or HTTP error |
| `/api/metrics` | GET | `?disease=heart` (optional) | JSON metrics object |
| `/predict` | POST | `{disease, values}` | `{ok, message}` (existing) |

## Data Models

### Metrics Response Schema

```json
{
    "ok": true,
    "metrics": [
        {
            "disease": "string (heart|kidney|liver|thyroid|diabetes)",
            "display_name": "string (e.g., Heart Disease)",
            "model": "string (algorithm name)",
            "original_features": "integer",
            "pca_components": "integer | null",
            "accuracy": "float (0-1)",
            "precision": "float (0-1)",
            "recall": "float (0-1)",
            "f1": "float (0-1)"
        }
    ]
}
```

### Graph Filename Convention

Pattern: `{disease}_{graph_type}.png`

| disease | graph_type options |
|---------|-------------------|
| heart | confusion_matrix, roc_curve, precision_recall_curve, pca_variance |
| kidney | confusion_matrix, roc_curve, precision_recall_curve, pca_variance |
| liver | confusion_matrix, roc_curve, precision_recall_curve, pca_variance |
| thyroid | confusion_matrix, roc_curve, precision_recall_curve, pca_variance |
| diabetes | confusion_matrix, roc_curve, precision_recall_curve, pca_variance |

Special file: `accuracy_summary.png` (overview graph)

### Pipeline Steps Data Structure (Client-side)

```javascript
const PIPELINE_STEPS = [
    {
        id: "input",
        title: "Data Input",
        description: "Your patient measurements are collected and formatted into a feature vector.",
        icon: "📥"
    },
    {
        id: "impute",
        title: "Missing Value Imputation",
        description: "Any missing values are filled using the median strategy learned from training data.",
        icon: "🔧"
    },
    {
        id: "scale",
        title: "Feature Scaling",
        description: "Values are normalized using RobustScaler to handle outliers and bring features to a comparable range.",
        icon: "⚖️"
    },
    {
        id: "pca",
        title: "PCA Dimensionality Reduction",
        description: "Features are projected onto principal components, reducing from {original} to {components} dimensions while retaining 95% of variance.",
        icon: "📐"
    },
    {
        id: "predict",
        title: "Model Inference",
        description: "The {model_name} model processes the transformed features and outputs a prediction: {result}.",
        icon: "🤖"
    }
];
```

### Disease Configuration (Client-side)

```javascript
const DISEASES = {
    heart:    { name: "Heart Disease",   key: "heart" },
    kidney:   { name: "Kidney Disease",  key: "kidney" },
    liver:    { name: "Liver Disease",   key: "liver" },
    thyroid:  { name: "Thyroid Disease", key: "thyroid" },
    diabetes: { name: "Diabetes",        key: "diabetes" }
};

const GRAPH_TYPES = [
    { key: "confusion_matrix",        label: "Confusion Matrix",       description: "Shows the count of correct and incorrect predictions broken down by actual vs predicted class." },
    { key: "roc_curve",               label: "ROC Curve",              description: "Plots the trade-off between true positive rate and false positive rate at various thresholds. A curve closer to the top-left corner indicates better performance." },
    { key: "precision_recall_curve",  label: "Precision-Recall Curve", description: "Shows precision vs recall at different decision thresholds. Useful for imbalanced datasets." },
    { key: "pca_variance",           label: "PCA Variance",           description: "Displays cumulative explained variance as PCA components are added, showing how much information is retained." }
];
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Graph URL generation follows naming convention

*For any* disease in {heart, kidney, liver, thyroid, diabetes} and *for any* graph type in {confusion_matrix, roc_curve, precision_recall_curve, pca_variance}, the generated image URL path SHALL be exactly `/graphs/{disease}_{graph_type}.png`.

**Validates: Requirements 1.1, 1.2**

### Property 2: Partial graph availability renders correctly

*For any* subset of the 4 graph types being available for a given disease, the Graph Viewer SHALL render `<img>` elements for each available graph and a fallback message element for each unavailable graph, such that the total count of rendered items (images + messages) equals exactly 4.

**Validates: Requirements 1.4**

### Property 3: Metrics table values are rounded to 2 decimal places

*For any* numeric metric value (accuracy, precision, recall, F1) between 0 and 1, the formatted display string SHALL contain exactly 2 digits after the decimal point.

**Validates: Requirements 2.3**

### Property 4: Pipeline step descriptions interpolate dynamic metadata

*For any* valid prediction metadata containing (original_features: positive integer, pca_components: positive integer, model_name: non-empty string, result: non-empty string), the PCA step description SHALL contain both the original_features and pca_components values, and the inference step description SHALL contain both the model_name and the result.

**Validates: Requirements 3.4, 3.5**

### Property 5: Exactly one view is visible at any time

*For any* sequence of navigation tab clicks (predict or analysis), at every point in time exactly one of the two views (prediction form, analysis panel) SHALL have `display: block` and the other SHALL have `display: none`.

**Validates: Requirements 4.1**

### Property 6: View switching preserves input field values

*For any* set of values entered into the prediction form input fields, after switching to the Analysis Panel and switching back to the prediction form, all input field values SHALL equal the originally entered values.

**Validates: Requirements 4.3**

### Property 7: Disease selection isolation

*For any* disease selected from the Disease Selector, all graph image `src` attributes displayed in the Analysis Panel SHALL contain the selected disease's key string, and SHALL NOT contain any other disease's key string.

**Validates: Requirements 5.2**

### Property 8: Non-existent graph filename returns 404

*For any* filename string that does not match an existing file in `graphs/current/` and does not contain path traversal characters, the `/graphs/<filename>` route SHALL return HTTP status 404.

**Validates: Requirements 6.2**

### Property 9: Path traversal in filename returns 400

*For any* filename string containing the substring `..` or the character `/` or the character `\`, the `/graphs/<filename>` route SHALL return HTTP status 400 regardless of whether a matching file exists.

**Validates: Requirements 6.4**

### Property 10: Accuracy displayed as percentage with one decimal place

*For any* accuracy float value between 0 and 1 (inclusive), the formatted percentage string SHALL equal the value multiplied by 100 and rounded to exactly one decimal place, followed by the `%` character (e.g., 0.969 → "96.9%").

**Validates: Requirements 7.1**

### Property 11: Metrics data source priority

*For any* disease, when both metrics_summary.csv contains a row for that disease AND the metrics pickle file exists, the returned metadata SHALL match the CSV row values. When only the pickle file is available, the returned metadata SHALL match the pickle values.

**Validates: Requirements 7.2**

## Error Handling

### Backend Errors

| Scenario | Behavior |
|----------|----------|
| Graph file not found | Return HTTP 404 with empty body |
| Path traversal attempt | Return HTTP 400 with error message |
| CSV file missing/malformed | `/api/metrics` falls back to pickle files |
| Pickle file missing | `/api/metrics` returns `{"ok": false, "message": "Model not trained"}` for that disease |
| Both CSV and pickle missing | Return indicator that model hasn't been trained yet |
| Invalid disease parameter | Return `{"ok": false, "message": "Invalid disease"}` |

### Frontend Errors

| Scenario | Behavior |
|----------|----------|
| Graph image fails to load (onerror) | Replace `<img>` with placeholder message |
| `/api/metrics` fetch fails | Show "Unable to load metrics" message |
| Prediction request fails | Hide Process Visualizer, show error in result area |
| Network timeout | Show generic connection error message |

### Error Response Format

```json
{
    "ok": false,
    "message": "Human-readable error description"
}
```

## Testing Strategy

### Property-Based Tests (fast-check with Flask test client)

The backend route logic (path traversal validation, 404 handling, metrics formatting, fallback priority) is well-suited for property-based testing. Use **Hypothesis** (Python PBT library) with Flask's test client.

**Configuration:**
- Minimum 100 iterations per property test
- Each test tagged with: `Feature: interactive-analysis-dashboard, Property {N}: {title}`

**Properties to implement:**
- Property 8: Generate random filenames without traversal chars, verify 404 for non-existent
- Property 9: Generate filenames with `..`, `/`, or `\` injected at random positions, verify 400
- Property 10: Generate random floats in [0, 1], verify percentage formatting
- Property 11: Generate random metrics data, test with/without CSV, verify priority logic
- Property 3: Generate random floats, verify 2-decimal formatting

### Unit Tests (pytest)

- Verify graph route returns 200 for existing known files
- Verify correct `image/png` Content-Type header
- Verify `/api/metrics` returns all 5 diseases when CSV exists
- Verify `/api/metrics` fallback to pickle when CSV row is missing
- Verify "not trained" response when both sources are absent
- Verify navigation default state (prediction form visible)
- Verify Disease Selector renders all 5 options
- Verify pipeline visualizer shows 5 steps with descriptions

### Integration Tests

- End-to-end flow: login → navigate to analysis → select disease → verify graphs load
- Prediction flow: predict → verify Process Visualizer animates
- Verify graph images are accessible after training completes

### Frontend Testing (manual + Selenium if needed)

- Tab switching preserves form values (Property 5, 6)
- Disease selection updates content (Property 7)
- Pipeline animation timing (500ms per step)
- Responsive layout of graph grid
- Fallback messages for missing content
