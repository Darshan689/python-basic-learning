import argparse
import contextlib
import json
import os
import sys
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_ROOT = "data"
MODELS_DIR = "models"
GRAPHS_DIR = os.path.join("graphs", "current")
TEMPLATES_DIR = "templates"
TARGET_ACCURACY = 0.90
RANDOM_STATE = 42
PCA_VARIANCE = 0.95  # Retain 95% of variance

DATASET_FILES = {
    "heart": "heart.csv",
    "kidney": "kidney.csv",
    "liver": "liver.csv",
    "thyroid": "thyroid.csv",
    "diabetes": "diabetes.csv",
}

# Display names used consistently in both console output and web dashboard
DISEASE_DISPLAY_NAMES = {
    "heart": "Heart Disease",
    "kidney": "Kidney Disease",
    "liver": "Liver Disease",
    "thyroid": "Thyroid Disease",
    "diabetes": "Diabetes",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train efficient disease prediction models and save evaluation plots."
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=TARGET_ACCURACY,
        help="Accuracy goal used in the final summary.",
    )
    parser.add_argument(
        "--thyroid-sample",
        type=int,
        default=50000,
        help="Maximum thyroid rows to use for fast training. Use 0 for the full file.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Open the interactive prediction prompt after training.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Open the merged login page and web prediction dashboard.",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        default=True,
        help="Enable PCA feature extraction (default: True).",
    )
    return parser.parse_args()


def ensure_folders():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    for filename in os.listdir(GRAPHS_DIR):
        if filename.endswith((".png", ".csv")):
            try:
                os.remove(os.path.join(GRAPHS_DIR, filename))
            except PermissionError:
                pass


def clean_target(disease, df):
    df = df.dropna(axis=1, how="all").copy()
    target = df.columns[-1]

    if disease == "thyroid":
        cleaned = df[target].astype(str).str.lower().str.strip()
        cleaned = cleaned.str.replace(r"[^a-zA-Z0-9]", "", regex=True)
        thyroid_map = {
            "benign": 0,
            "negative": 0,
            "normal": 0,
            "no": 0,
            "false": 0,
            "0": 0,
            "malignant": 1,
            "positive": 1,
            "yes": 1,
            "true": 1,
            "1": 1,
            "sick": 1,
            "hyperthyroid": 1,
            "hypothyroid": 1,
        }
        df[target] = cleaned.map(thyroid_map)
    elif disease == "kidney":
        cleaned = df[target].astype(str).str.lower().str.strip()
        cleaned = cleaned.str.replace(r"[^a-zA-Z0-9]", "", regex=True)
        df[target] = cleaned.map({"ckd": 1, "notckd": 0, "0": 0, "1": 1})
    elif disease == "liver":
        df[target] = df[target].replace({1: 1, 2: 0})

    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    return df, target


def load_dataset(disease, path, thyroid_sample):
    df = pd.read_csv(path)
    df, target = clean_target(disease, df)

    if disease == "thyroid" and thyroid_sample and len(df) > thyroid_sample:
        df = df.sample(n=thyroid_sample, random_state=RANDOM_STATE)

    x = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    y = df[target]
    return x, y


def candidate_models(disease):
    common_logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    random_forest = RandomForestClassifier(
        n_estimators=180,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    extra_trees = ExtraTreesClassifier(
        n_estimators=220,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    recipes = {
        "thyroid": {"Logistic Regression": common_logistic},
        "kidney": {
            "Logistic Regression": common_logistic,
            "Random Forest": random_forest,
        },
        "diabetes": {
            "Logistic Regression": common_logistic,
            "Random Forest": random_forest,
        },
        "heart": {
            "Logistic Regression": common_logistic,
            "Random Forest": random_forest,
            "Extra Trees": extra_trees,
        },
        "liver": {
            "Logistic Regression": common_logistic,
            "Random Forest": random_forest,
            "Extra Trees": extra_trees,
        },
    }
    return recipes[disease]


def plot_confusion_matrix(disease, model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
    )
    plt.title(f"{disease.title()} - {model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    output = os.path.join(GRAPHS_DIR, f"{disease}_confusion_matrix.png")
    plt.savefig(output, dpi=140)
    plt.close()
    return output


def plot_probability_curves(disease, y_true, scores):
    roc_output = os.path.join(GRAPHS_DIR, f"{disease}_roc_curve.png")
    pr_output = os.path.join(GRAPHS_DIR, f"{disease}_precision_recall_curve.png")

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{disease.title()} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_output, dpi=140)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.title(f"{disease.title()} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(pr_output, dpi=140)
    plt.close()
    return roc_output, pr_output


def plot_pca_variance(disease, pca):
    """Plot the cumulative explained variance by PCA components."""
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
    plt.axhline(y=PCA_VARIANCE, color='r', linestyle='--', label=f'Target: {PCA_VARIANCE:.0%}')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"{disease.title()} - PCA Explained Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output = os.path.join(GRAPHS_DIR, f"{disease}_pca_variance.png")
    plt.savefig(output, dpi=140)
    plt.close()
    return output


def plot_accuracy_summary(results, target_accuracy):
    names = [result["disease"] for result in results]
    accuracies = [result["accuracy"] for result in results]
    colors = ["#2ca25f" if score >= target_accuracy else "#de6e4b" for score in accuracies]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=accuracies, palette=colors)
    plt.axhline(target_accuracy, color="black", linestyle="--", label=f"Target {target_accuracy:.2f}")
    plt.ylim(0, 1)
    plt.title("Best Model Accuracy by Disease")
    plt.xlabel("Disease")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    output = os.path.join(GRAPHS_DIR, "accuracy_summary.png")
    plt.savefig(output, dpi=140)
    plt.close()
    return output


def save_artifacts(disease, model, scaler, imputer, pca, features, metrics):
    joblib.dump(model, os.path.join(MODELS_DIR, f"{disease}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{disease}_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODELS_DIR, f"{disease}_imputer.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, f"{disease}_pca.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, f"{disease}_features.pkl"))
    joblib.dump(metrics, os.path.join(MODELS_DIR, f"{disease}_metrics.pkl"))


def train_one_dataset(disease, path, args):
    start = time.perf_counter()
    x, y = load_dataset(disease, path, args.thyroid_sample)

    if y.nunique() < 2:
        raise ValueError(f"{disease} has fewer than two target classes after cleaning.")

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    feature_names = x.columns.tolist()
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train = imputer.fit_transform(x_train_raw)
    x_test = imputer.transform(x_test_raw)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Apply PCA for feature extraction
    pca = None
    original_n_features = x_train.shape[1]
    if args.pca and original_n_features > 2:
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        n_components = pca.n_components_
        print(
            f"  PCA: Reduced from {original_n_features} to {n_components} components "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})"
        )

    best = None
    for model_name, model in candidate_models(disease).items():
        fitted_model = clone(model)
        fitted_model.fit(x_train, y_train)
        prediction = fitted_model.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)
        f1 = f1_score(y_test, prediction, zero_division=0)

        if best is None or (accuracy, f1) > (best["accuracy"], best["f1"]):
            best = {
                "model_name": model_name,
                "model": fitted_model,
                "prediction": prediction,
                "accuracy": accuracy,
                "f1": f1,
            }

    prediction = best["prediction"]
    metrics = {
        "disease": disease,
        "model": best["model_name"],
        "rows": len(y),
        "original_features": original_n_features,
        "pca_components": pca.n_components_ if pca else None,
        "accuracy": best["accuracy"],
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": best["f1"],
        "seconds": time.perf_counter() - start,
        "classification_report": classification_report(
            y_test,
            prediction,
            target_names=["No Disease", "Disease"],
            zero_division=0,
        ),
    }

    confusion_plot = plot_confusion_matrix(disease, best["model_name"], y_test, prediction)
    probability_plots = []
    if hasattr(best["model"], "predict_proba"):
        scores = best["model"].predict_proba(x_test)[:, 1]
        probability_plots = list(plot_probability_curves(disease, y_test, scores))

    pca_plot = None
    if pca:
        pca_plot = plot_pca_variance(disease, pca)

    save_artifacts(disease, best["model"], scaler, imputer, pca, feature_names, metrics)
    metrics["plots"] = [confusion_plot] + probability_plots + ([pca_plot] if pca_plot else [])
    return metrics


def print_summary(results, target_accuracy):
    print("\nSUMMARY")
    print("-" * 110)
    print(
        f"{'Disease':<10} {'Best model':<20} {'Features':>10} {'PCA Comp':>10} "
        f"{'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Time':>8} Goal"
    )
    print("-" * 110)
    for result in results:
        goal = "PASS" if result["accuracy"] >= target_accuracy else "BELOW"
        pca_comp = result.get("pca_components", "N/A")
        print(
            f"{result['disease']:<10} {result['model']:<20} "
            f"{result.get('original_features', 'N/A'):>10} {str(pca_comp):>10} "
            f"{result['accuracy']:>9.3f} {result['precision']:>10.3f} "
            f"{result['recall']:>8.3f} {result['f1']:>8.3f} "
            f"{result['seconds']:>7.1f}s {goal}"
        )
    print("-" * 110)

    below_goal = [r for r in results if r["accuracy"] < target_accuracy]
    if below_goal:
        names = ", ".join(r["disease"] for r in below_goal)
        print(
            f"Accuracy goal note: {names} did not reach {target_accuracy:.2f} on the held-out test split. "
            "The code reports this instead of inflating accuracy with test leakage."
        )


WEB_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multiple Disease Prediction System</title>

<style>
body{
    margin:0;
    font-family:Arial,sans-serif;
    background:#f2f2f2;
}

#loginPage{
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
    background:url("https://images.unsplash.com/photo-1579154204601-01588f351e67")
    no-repeat center center fixed;
    background-size:cover;
}

.login-box{
    background:rgba(0,0,0,0.7);
    color:white;
    padding:30px;
    border-radius:10px;
    width:300px;
}

input,button{
    width:100%;
    padding:10px;
    margin:6px 0;
}

button{
    cursor:pointer;
}

#dashboard{
    display:none;
    padding:20px;
    background:white;
    min-height:100vh;
}

.card{
    background:#f9f9f9;
    padding:20px;
    border-radius:10px;
    width:500px;
    margin-top:20px;
}

.result{
    margin-top:15px;
    font-size:16px;
    font-weight:bold;
    line-height:1.8;
}

#trainingStatus{
    margin-top:10px;
    font-size:14px;
    color:#555;
}

.tab-nav{
    display:flex;
    gap:0;
    margin-bottom:15px;
}

.tab-btn{
    padding:10px 25px;
    border:none;
    background:#e0e0e0;
    color:#333;
    font-size:15px;
    cursor:pointer;
    border-radius:5px 5px 0 0;
    transition:background 0.2s, color 0.2s;
}

.tab-btn:hover{
    background:#c0c0c0;
}

.tab-btn.active{
    background:#4CAF50;
    color:white;
    font-weight:bold;
}

#analysis-view{
    display:none;
    padding:10px 0;
}

.graph-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:20px;
    margin-top:15px;
}

.graph-item{
    background:#fff;
    border:1px solid #ddd;
    border-radius:8px;
    padding:10px;
    text-align:center;
}

.graph-item img{
    width:100%;
    max-width:400px;
    height:auto;
    border-radius:4px;
}

.graph-item .graph-description{
    font-size:13px;
    color:#555;
    margin-top:8px;
    line-height:1.4;
}

.graph-placeholder{
    display:flex;
    align-items:center;
    justify-content:center;
    min-height:200px;
    background:#f0f0f0;
    border:2px dashed #ccc;
    border-radius:8px;
    color:#888;
    font-size:14px;
    padding:20px;
}

#graph-viewer .graphs-unavailable{
    text-align:center;
    padding:40px;
    color:#888;
    font-size:16px;
}

.disease-selector{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-bottom:20px;
}

.disease-btn{
    padding:10px 18px;
    border:2px solid #4CAF50;
    background:white;
    color:#333;
    font-size:14px;
    font-weight:500;
    cursor:pointer;
    border-radius:20px;
    transition:background 0.2s, color 0.2s, border-color 0.2s;
}

.disease-btn:hover{
    background:#e8f5e9;
}

.disease-btn.active{
    background:#4CAF50;
    color:white;
    border-color:#4CAF50;
    font-weight:bold;
}

.metrics-table{
    width:100%;
    border-collapse:collapse;
    margin-top:15px;
    font-size:14px;
}

.metrics-table th,
.metrics-table td{
    border:1px solid #ddd;
    padding:10px 12px;
    text-align:left;
}

.metrics-table th{
    background:#4CAF50;
    color:white;
    font-weight:bold;
}

.metrics-table tr:nth-child(even){
    background:#f9f9f9;
}

.metrics-table tr:hover{
    background:#e8f5e9;
}

.overview-graph{
    text-align:center;
    margin-bottom:20px;
}

.overview-graph img{
    max-width:100%;
    height:auto;
    border-radius:8px;
    border:1px solid #ddd;
}

.fallback-message{
    text-align:center;
    padding:30px;
    color:#888;
    font-size:15px;
    background:#f5f5f5;
    border:2px dashed #ccc;
    border-radius:8px;
    margin:15px 0;
}

.metadata-card{
    background:#fff;
    border:1px solid #ddd;
    border-radius:10px;
    padding:20px;
    margin-bottom:20px;
    box-shadow:0 2px 6px rgba(0,0,0,0.08);
}

.metadata-card h4{
    margin:0 0 15px 0;
    color:#333;
    font-size:16px;
    border-bottom:2px solid #4CAF50;
    padding-bottom:8px;
}

.metadata-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:12px;
}

.metadata-item{
    background:#f9f9f9;
    border-radius:6px;
    padding:12px;
    text-align:center;
}

.metadata-item .meta-label{
    font-size:12px;
    color:#777;
    text-transform:uppercase;
    letter-spacing:0.5px;
    margin-bottom:4px;
}

.metadata-item .meta-value{
    font-size:18px;
    font-weight:bold;
    color:#333;
}

.metadata-item .meta-value.accuracy-value{
    color:#4CAF50;
}

.metadata-not-trained{
    text-align:center;
    padding:20px;
    color:#888;
    font-size:15px;
    background:#fff3e0;
    border:1px solid #ffe0b2;
    border-radius:8px;
    margin-bottom:20px;
}

#process-visualizer{
    display:none;
    margin-top:20px;
    padding:15px;
    background:#f9f9f9;
    border-radius:10px;
    border:1px solid #e0e0e0;
}

#process-visualizer h4{
    margin:0 0 15px 0;
    color:#333;
    font-size:16px;
    border-bottom:2px solid #4CAF50;
    padding-bottom:8px;
}

.pipeline-step{
    display:flex;
    align-items:flex-start;
    gap:12px;
    padding:12px;
    margin-bottom:10px;
    background:#fff;
    border-radius:8px;
    border:1px solid #eee;
    opacity:0;
    transition:opacity 0.5s ease-in-out;
}

.pipeline-step.visible{
    opacity:1;
}

.pipeline-step .step-icon{
    font-size:24px;
    flex-shrink:0;
}

.pipeline-step .step-content{
    flex:1;
}

.pipeline-step .step-title{
    font-weight:bold;
    font-size:14px;
    color:#333;
    margin-bottom:4px;
}

.pipeline-step .step-description{
    font-size:13px;
    color:#555;
    line-height:1.4;
    margin:0;
}

@media (max-width:768px){
    .graph-grid{
        grid-template-columns:1fr;
    }
    .metadata-grid{
        grid-template-columns:1fr;
    }
    .disease-selector{
        justify-content:center;
    }
    .card{
        width:100%;
        box-sizing:border-box;
    }
}
</style>
</head>

<body>

<!-- LOGIN -->
<div id="loginPage">
    <div class="login-box" id="loginBox">
        <h2>Login</h2>

        <input type="text" id="username" placeholder="Username">
        <input type="password" id="password" placeholder="Password">

        <button onclick="login()">Login</button>
        <p id="loginMsg"></p>
        <p style="margin-top:10px;font-size:13px;">Don't have an account? <a href="#" onclick="showRegister()" style="color:#4CAF50;cursor:pointer;">Register here</a></p>
    </div>

    <div class="login-box" id="registerBox" style="display:none;">
        <h2>Register</h2>

        <input type="text" id="regUsername" placeholder="Username">
        <input type="email" id="regEmail" placeholder="Email">
        <input type="password" id="regPassword" placeholder="Password">
        <input type="password" id="regConfirmPassword" placeholder="Confirm Password">

        <button onclick="register()">Register</button>
        <p id="registerMsg"></p>
        <p style="margin-top:10px;font-size:13px;">Already have an account? <a href="#" onclick="showLogin()" style="color:#4CAF50;cursor:pointer;">Login here</a></p>
    </div>
</div>

<!-- DASHBOARD -->
<div id="dashboard">

<h2>Multiple Disease Prediction System</h2>

<div id="trainingStatus">Checking training status...</div>

<!-- Tab Navigation -->
<div class="tab-nav">
    <button class="tab-btn active" id="tab-predict" onclick="switchTab('predict')">Predict</button>
    <button class="tab-btn" id="tab-analysis" onclick="switchTab('analysis')">Analysis</button>
</div>

<!-- Prediction Form View (default active) -->
<div id="predict-view">

<div class="card">

    <h3>Enter Patient Details</h3>

    <input type="number" id="age" placeholder="Age">
    <input type="number" id="glucose" placeholder="Glucose">
    <input type="number" id="bmi" placeholder="BMI">
    <input type="number" id="bp" placeholder="Blood Pressure">
    <input type="number" id="albumin" placeholder="Albumin">
    <input type="number" id="creatinine" placeholder="Serum Creatinine">
    <input type="number" id="tsh" placeholder="TSH">
    <input type="number" id="t3" placeholder="T3">
    <input type="number" id="bilirubin" placeholder="Total Bilirubin">
    <input type="number" id="cholesterol" placeholder="Cholesterol">

    <button onclick="predictAll()">Predict All Diseases</button>

    <div class="result" id="result"></div>

</div>

<div id="process-visualizer">
    <h4>🔬 Prediction Pipeline</h4>
    <div id="pipeline-steps-container"></div>
</div>

</div>

<!-- Analysis Panel View (hidden by default) -->
<div id="analysis-view">
    <h3>Analysis Panel</h3>
    <p>Select a disease to view model evaluation graphs and metrics.</p>

    <!-- Disease Selector -->
    <div class="disease-selector">
        <button class="disease-btn active" onclick="selectDisease('overview')">Overview</button>
        <button class="disease-btn" onclick="selectDisease('heart')">Heart Disease</button>
        <button class="disease-btn" onclick="selectDisease('kidney')">Kidney Disease</button>
        <button class="disease-btn" onclick="selectDisease('liver')">Liver Disease</button>
        <button class="disease-btn" onclick="selectDisease('thyroid')">Thyroid Disease</button>
        <button class="disease-btn" onclick="selectDisease('diabetes')">Diabetes</button>
    </div>

    <!-- Analysis content area (populated by Graph Viewer and metadata) -->
    <div id="analysis-content">
        <div id="graph-viewer"></div>
    </div>
</div>

<button onclick="logout()">Logout</button>

</div>

<script>

function switchTab(tab){
    var predictView = document.getElementById("predict-view");
    var analysisView = document.getElementById("analysis-view");
    var predictBtn = document.getElementById("tab-predict");
    var analysisBtn = document.getElementById("tab-analysis");

    if(tab === "predict"){
        predictView.style.display = "block";
        analysisView.style.display = "none";
        predictBtn.classList.add("active");
        analysisBtn.classList.remove("active");
    } else {
        predictView.style.display = "none";
        analysisView.style.display = "block";
        predictBtn.classList.remove("active");
        analysisBtn.classList.add("active");
        // Show overview by default when Analysis tab is opened
        if(currentDisease === "overview"){
            showOverview();
        }
    }
}

var currentDisease = "overview";

function selectDisease(key){
    currentDisease = key;

    // Update active class on disease buttons
    var buttons = document.querySelectorAll(".disease-btn");
    buttons.forEach(function(btn){
        btn.classList.remove("active");
    });

    // Find the clicked button and mark it active
    buttons.forEach(function(btn){
        var btnKey = btn.getAttribute("onclick").match(/selectDisease\\('(\\w+)'\\)/);
        if(btnKey && btnKey[1] === key){
            btn.classList.add("active");
        }
    });

    // Update the analysis content area
    var contentArea = document.getElementById("analysis-content");
    contentArea.innerHTML = "";

    if(key === "overview"){
        showOverview();
    } else {
        showMetadataCard(key);
        showDiseaseGraphs(key);
    }
}

var GRAPH_TYPES = [
    { key: "confusion_matrix", label: "Confusion Matrix", description: "Shows the count of correct and incorrect predictions broken down by actual vs predicted class." },
    { key: "roc_curve", label: "ROC Curve", description: "Plots the trade-off between true positive rate and false positive rate at various thresholds. A curve closer to the top-left corner indicates better performance." },
    { key: "precision_recall_curve", label: "Precision-Recall Curve", description: "Shows precision vs recall at different decision thresholds. Useful for imbalanced datasets." },
    { key: "pca_variance", label: "PCA Variance", description: "Displays cumulative explained variance as PCA components are added, showing how much information is retained." }
];

async function showMetadataCard(disease){
    var container = document.getElementById("analysis-content");

    // Create or find the metadata-card div
    var cardDiv = document.getElementById("metadata-card");
    if(!cardDiv){
        cardDiv = document.createElement("div");
        cardDiv.id = "metadata-card";
        container.insertBefore(cardDiv, container.firstChild);
    }
    cardDiv.innerHTML = '<p style="color:#555;">Loading model metadata...</p>';

    try{
        var response = await fetch("/api/metrics?disease=" + encodeURIComponent(disease));
        var data = await response.json();

        if(!data.ok || !data.metrics || data.metrics.length === 0){
            cardDiv.innerHTML = '<div class="metadata-not-trained">Model has not been trained yet</div>';
            return;
        }

        var m = data.metrics[0];
        var accuracyPct = (m.accuracy * 100).toFixed(1) + "%";

        var html = '<div class="metadata-card">';
        html += '<h4>Model Metadata</h4>';
        html += '<div class="metadata-grid">';
        html += '<div class="metadata-item"><div class="meta-label">Algorithm</div><div class="meta-value">' + m.model + '</div></div>';
        html += '<div class="metadata-item"><div class="meta-label">Original Features</div><div class="meta-value">' + m.original_features + '</div></div>';
        html += '<div class="metadata-item"><div class="meta-label">PCA Components</div><div class="meta-value">' + (m.pca_components !== null ? m.pca_components : "N/A") + '</div></div>';
        html += '<div class="metadata-item"><div class="meta-label">Accuracy</div><div class="meta-value accuracy-value">' + accuracyPct + '</div></div>';
        html += '</div></div>';

        cardDiv.innerHTML = html;
    } catch(err){
        cardDiv.innerHTML = '<div class="metadata-not-trained">Model has not been trained yet</div>';
    }
}

function showDiseaseGraphs(disease){
    var container = document.getElementById("analysis-content");
    // Create or find the graph-viewer div
    var viewer = document.getElementById("graph-viewer");
    if(!viewer){
        viewer = document.createElement("div");
        viewer.id = "graph-viewer";
        container.appendChild(viewer);
    }
    viewer.innerHTML = "";

    var grid = document.createElement("div");
    grid.className = "graph-grid";

    var errorCount = 0;
    var loadedCount = 0;
    var totalGraphs = GRAPH_TYPES.length;

    GRAPH_TYPES.forEach(function(graphType){
        var item = document.createElement("div");
        item.className = "graph-item";

        var img = document.createElement("img");
        img.src = "/graphs/" + disease + "_" + graphType.key + ".png";
        img.alt = graphType.label + " for " + disease;

        img.onerror = function(){
            errorCount++;
            var placeholder = document.createElement("div");
            placeholder.className = "graph-placeholder";
            placeholder.textContent = graphType.label + " not available for " + disease;
            item.replaceChild(placeholder, img);
            checkAllGraphs();
        };

        img.onload = function(){
            loadedCount++;
            checkAllGraphs();
        };

        var desc = document.createElement("p");
        desc.className = "graph-description";
        desc.textContent = graphType.description;

        item.appendChild(img);
        item.appendChild(desc);
        grid.appendChild(item);
    });

    viewer.appendChild(grid);

    function checkAllGraphs(){
        if(errorCount + loadedCount === totalGraphs){
            if(errorCount === totalGraphs){
                viewer.innerHTML = "";
                var msg = document.createElement("div");
                msg.className = "graphs-unavailable";
                msg.textContent = "Graphs not yet available for this disease";
                viewer.appendChild(msg);
            }
        }
    }
}

async function showOverview(){
    var contentArea = document.getElementById("analysis-content");
    contentArea.innerHTML = "";

    // Create overview content container
    var overviewDiv = document.createElement("div");
    overviewDiv.id = "overview-content";

    // Add accuracy summary graph
    var graphDiv = document.createElement("div");
    graphDiv.className = "overview-graph";
    var img = document.createElement("img");
    img.src = "/graphs/accuracy_summary.png";
    img.alt = "Accuracy Summary";
    img.onerror = function(){
        graphDiv.innerHTML = '<div class="fallback-message">Accuracy summary graph is not available</div>';
    };
    graphDiv.appendChild(img);
    overviewDiv.appendChild(graphDiv);

    // Add placeholder for metrics table (will be populated after fetch)
    var tableDiv = document.createElement("div");
    tableDiv.id = "metrics-table-container";
    tableDiv.innerHTML = '<p style="color:#555;">Loading metrics...</p>';
    overviewDiv.appendChild(tableDiv);

    contentArea.appendChild(overviewDiv);

    // Fetch metrics data
    try{
        var response = await fetch("/api/metrics");
        var data = await response.json();

        if(!data.ok){
            tableDiv.innerHTML = '<div class="fallback-message">Metrics data is not available</div>';
            return;
        }

        // Build summary table
        var html = '<table class="metrics-table">';
        html += '<thead><tr>';
        html += '<th>Disease</th><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th>';
        html += '</tr></thead><tbody>';

        data.metrics.forEach(function(m){
            html += '<tr>';
            html += '<td>' + m.display_name + '</td>';
            html += '<td>' + m.model + '</td>';
            html += '<td>' + parseFloat(m.accuracy).toFixed(2) + '</td>';
            html += '<td>' + parseFloat(m.precision).toFixed(2) + '</td>';
            html += '<td>' + parseFloat(m.recall).toFixed(2) + '</td>';
            html += '<td>' + parseFloat(m.f1).toFixed(2) + '</td>';
            html += '</tr>';
        });

        html += '</tbody></table>';
        tableDiv.innerHTML = html;
    } catch(err){
        tableDiv.innerHTML = '<div class="fallback-message">Metrics data is not available</div>';
    }
}

function showRegister(){
    document.getElementById("loginBox").style.display = "none";
    document.getElementById("registerBox").style.display = "block";
    document.getElementById("registerMsg").innerHTML = "";
}

function showLogin(){
    document.getElementById("registerBox").style.display = "none";
    document.getElementById("loginBox").style.display = "block";
    document.getElementById("loginMsg").innerHTML = "";
}

async function register(){
    var username = document.getElementById("regUsername").value.trim();
    var email = document.getElementById("regEmail").value.trim();
    var password = document.getElementById("regPassword").value;
    var confirmPassword = document.getElementById("regConfirmPassword").value;

    if(!username || !email || !password){
        document.getElementById("registerMsg").innerHTML = "⚠ Please fill in all fields.";
        return;
    }
    if(password !== confirmPassword){
        document.getElementById("registerMsg").innerHTML = "⚠ Passwords do not match.";
        return;
    }
    if(password.length < 4){
        document.getElementById("registerMsg").innerHTML = "⚠ Password must be at least 4 characters.";
        return;
    }

    try{
        var response = await fetch("/register", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({username: username, email: email, password: password})
        });
        var data = await response.json();
        if(data.ok){
            document.getElementById("registerMsg").innerHTML = '<span style="color:#4CAF50;">✓ Registration successful! Please login.</span>';
            setTimeout(showLogin, 1500);
        } else {
            document.getElementById("registerMsg").innerHTML = "⚠ " + data.message;
        }
    } catch(err){
        document.getElementById("registerMsg").innerHTML = "⚠ Registration failed. Try again.";
    }
}

async function login(){
    let u = document.getElementById("username").value.trim();
    let p = document.getElementById("password").value;

    if(!u || !p){
        document.getElementById("loginMsg").innerHTML = "⚠ Please enter username and password.";
        return;
    }

    try{
        var response = await fetch("/login", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({username: u, password: p})
        });
        var data = await response.json();
        if(data.ok){
            document.getElementById("loginPage").style.display="none";
            document.getElementById("dashboard").style.display="block";

            document.getElementById("trainingStatus").innerHTML="Starting training...";
            fetch("/start-training", {method:"POST"})
                .then(() => checkTrainingStatus());
        } else {
            document.getElementById("loginMsg").innerHTML = "⚠ " + data.message;
        }
    } catch(err){
        document.getElementById("loginMsg").innerHTML = "⚠ Login failed. Try again.";
    }
}

let trainingDone = false;

async function checkTrainingStatus(){
    try{
        const response = await fetch("/training-status");
        const data = await response.json();

        if(!data.finished){
            trainingDone = false;
            document.getElementById("trainingStatus").innerHTML = "Training in progress... " + data.message;
            setTimeout(checkTrainingStatus, 2000);
            return;
        }
        
        trainingDone = true;
        document.getElementById("trainingStatus").innerHTML =
            "Training completed ✔ Ready for prediction!";
    } catch(err){
        document.getElementById("trainingStatus").innerHTML =
            "Status check failed: " + err;
    }
}

var PIPELINE_STEPS = [
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
        description: "Features are projected onto principal components, reducing dimensions while retaining 95% of variance.",
        icon: "📐"
    },
    {
        id: "predict",
        title: "Model Inference",
        description: "The model processes the transformed features and outputs a prediction.",
        icon: "🤖"
    }
];

function showPipelineVisualizer(metadata){
    var container = document.getElementById("process-visualizer");
    var stepsContainer = document.getElementById("pipeline-steps-container");
    stepsContainer.innerHTML = "";
    container.style.display = "block";

    PIPELINE_STEPS.forEach(function(step, index){
        var stepDiv = document.createElement("div");
        stepDiv.className = "pipeline-step";

        var iconSpan = document.createElement("span");
        iconSpan.className = "step-icon";
        iconSpan.textContent = step.icon;

        var contentDiv = document.createElement("div");
        contentDiv.className = "step-content";

        var titleDiv = document.createElement("div");
        titleDiv.className = "step-title";
        titleDiv.textContent = step.title;

        var descP = document.createElement("p");
        descP.className = "step-description";

        // Interpolate dynamic metadata into PCA and inference steps
        var description = step.description;
        if(step.id === "pca" && metadata && metadata.original_features && metadata.pca_components){
            description = "Features are projected onto principal components, reducing from " + metadata.original_features + " to " + metadata.pca_components + " dimensions while retaining 95% of variance.";
        }
        if(step.id === "predict" && metadata && metadata.model_name && metadata.result){
            description = "The " + metadata.model_name + " model processes the transformed features and outputs a prediction: " + metadata.result + ".";
        }
        descP.textContent = description;

        contentDiv.appendChild(titleDiv);
        contentDiv.appendChild(descP);
        stepDiv.appendChild(iconSpan);
        stepDiv.appendChild(contentDiv);
        stepsContainer.appendChild(stepDiv);

        setTimeout(function(){
            stepDiv.classList.add("visible");
        }, index * 500);
    });
}

function hidePipelineVisualizer(){
    var container = document.getElementById("process-visualizer");
    container.style.display = "none";
    var stepsContainer = document.getElementById("pipeline-steps-container");
    stepsContainer.innerHTML = "";
}

async function predictOne(disease, values){
    const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({disease, values})
    });
    const data = await response.json();
    return data;
}

async function predictAll(){
    if(!trainingDone){
        document.getElementById("result").innerHTML =
            "⚠ Please wait until training is completed.";
        hidePipelineVisualizer();
        return;
    }

    // Validate mandatory fields
    var mandatoryFields = [
        {id: "age", name: "Age"},
        {id: "glucose", name: "Glucose"},
        {id: "bmi", name: "BMI"},
        {id: "bp", name: "Blood Pressure"}
    ];
    var missing = [];
    for(var i = 0; i < mandatoryFields.length; i++){
        var field = document.getElementById(mandatoryFields[i].id);
        if(!field.value || field.value.trim() === ""){
            missing.push(mandatoryFields[i].name);
        }
    }
    if(missing.length > 0){
        document.getElementById("result").innerHTML =
            "⚠ Please enter values for: " + missing.join(", ");
        hidePipelineVisualizer();
        return;
    }

    let values = {
        Age: +document.getElementById("age").value,
        Glucose: +document.getElementById("glucose").value,
        BMI: +document.getElementById("bmi").value,
        BloodPressure: +document.getElementById("bp").value,
        Albumin: +document.getElementById("albumin").value,
        Serum_Creatinine: +document.getElementById("creatinine").value,
        TSH: +document.getElementById("tsh").value,
        T3: +document.getElementById("t3").value,
        Total_Bilirubin: +document.getElementById("bilirubin").value,
        Cholesterol: +document.getElementById("cholesterol").value
    };

    document.getElementById("result").innerHTML = "Predicting...";
    hidePipelineVisualizer();

    const order = ["diabetes", "kidney", "heart", "liver", "thyroid"];
    let lines = [];
    let hasError = false;
    let lastResult = "";
    let lastDisease = "";
    let pipelineMetadata = {};

    try{
        for(const disease of order){
            const data = await predictOne(disease, values);
            var msg = data.message || "Error";
            if(msg === "Error" || !data.ok){
                hasError = true;
            } else {
                lastResult = msg;
                lastDisease = disease;
                // Collect metadata from prediction response
                pipelineMetadata = {
                    result: msg,
                    model_name: data.model_name || null,
                    original_features: data.original_features || null,
                    pca_components: data.pca_components || null
                };
            }
            const icon = msg.toLowerCase().startsWith("no") ? "🟢" : "🔴";
            lines.push(`${icon} ${msg}`);
        }

        document.getElementById("result").innerHTML = lines.join("<br>");

        if(!hasError){
            showPipelineVisualizer(pipelineMetadata);
        } else {
            hidePipelineVisualizer();
        }
    } catch(err){
        document.getElementById("result").innerHTML = "❌ Prediction failed: " + err.message;
        hidePipelineVisualizer();
    }
}

function logout(){
    location.reload();
}

</script>

</body>
</html>
"""

def load_prediction_bundle(disease_choice):
    model_path = os.path.join(MODELS_DIR, f"{disease_choice}_model.pkl")
    if not os.path.exists(model_path):
        return None

    return {
        "model": joblib.load(model_path),
        "scaler": joblib.load(os.path.join(MODELS_DIR, f"{disease_choice}_scaler.pkl")),
        "features": joblib.load(os.path.join(MODELS_DIR, f"{disease_choice}_features.pkl")),
        "imputer": joblib.load(os.path.join(MODELS_DIR, f"{disease_choice}_imputer.pkl")),
        "pca": joblib.load(os.path.join(MODELS_DIR, f"{disease_choice}_pca.pkl")),
    }


def predict_web(disease_choice, values):
    bundle = load_prediction_bundle(disease_choice)
    name = DISEASE_DISPLAY_NAMES.get(disease_choice, disease_choice.title())

    if bundle is None:
        return predict_web_demo(disease_choice, values)

    input_data = []
    for feature in bundle["features"]:
        try:
            input_data.append(float(values.get(feature, 0)))
        except (TypeError, ValueError):
            input_data.append(0.0)

    input_array = np.array(input_data).reshape(1, -1)
    input_array = bundle["imputer"].transform(input_array)
    input_array = bundle["scaler"].transform(input_array)

    # Apply PCA transformation if available
    pca_components = None
    if bundle["pca"]:
        input_array = bundle["pca"].transform(input_array)
        pca_components = int(bundle["pca"].n_components_)

    prediction = bundle["model"].predict(input_array)[0]

    # Determine model name from metrics pickle or model type
    model_name = type(bundle["model"]).__name__
    metrics_path = os.path.join(MODELS_DIR, f"{disease_choice}_metrics.pkl")
    if os.path.exists(metrics_path):
        try:
            metrics_data = joblib.load(metrics_path)
            model_name = metrics_data.get("model", model_name)
        except Exception:
            pass

    original_features = len(bundle["features"])

    if int(prediction) == 1:
        message = f"{name} Detected"
    else:
        message = f"No {name}"

    return {
        "ok": True,
        "message": message,
        "model_name": model_name,
        "original_features": original_features,
        "pca_components": pca_components,
    }


# Field names used by the dashboard form (templates/index.html)
DEMO_FEATURES = {
    "diabetes": ["Glucose", "BMI"],
    "kidney": ["BloodPressure", "Albumin"],
    "heart": ["Age", "Cholesterol"],
    "liver": ["Total_Bilirubin", "Albumin"],
    "thyroid": ["TSH", "T3"],
}


def read_demo_value(values, *names):
    lowered = {str(key).strip().lower(): value for key, value in values.items()}
    for name in names:
        try:
            return float(lowered.get(name.lower(), 0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def predict_web_demo(disease_choice, values):
    name = DISEASE_DISPLAY_NAMES.get(disease_choice, disease_choice.title())

    # Demo feature counts per disease (approximate values for demo mode)
    demo_features = {
        "diabetes": 8,
        "kidney": 24,
        "heart": 13,
        "liver": 10,
        "thyroid": 21,
    }

    def _demo_response(detected):
        message = f"{name} Detected" if detected else f"No {name}"
        return {
            "ok": True,
            "message": message,
            "model_name": "Demo (Rule-Based)",
            "original_features": demo_features.get(disease_choice, 10),
            "pca_components": None,
        }

    if disease_choice == "diabetes":
        detected = read_demo_value(values, "Glucose") > 126 or read_demo_value(values, "BMI") > 30
        return _demo_response(detected)

    if disease_choice == "kidney":
        detected = (
            read_demo_value(values, "BloodPressure") > 140
            or read_demo_value(values, "Serum_Creatinine") > 1.5
        )
        return _demo_response(detected)

    if disease_choice == "heart":
        detected = (
            read_demo_value(values, "Age") > 50
            or read_demo_value(values, "Cholesterol") > 200
            or read_demo_value(values, "BloodPressure") > 140
        )
        return _demo_response(detected)

    if disease_choice == "liver":
        detected = (
            read_demo_value(values, "Total_Bilirubin") > 1.2
            or read_demo_value(values, "Albumin") < 3.5
        )
        return _demo_response(detected)

    if disease_choice == "thyroid":
        detected = read_demo_value(values, "TSH") > 4.5 or read_demo_value(values, "T3") < 1.0
        return _demo_response(detected)

    return {"ok": False, "message": "Invalid disease selected."}


WEB_TRAINING_STATUS = {
    "started": False,
    "finished": False,
    "message": "Training not started.",
    "output": "",
}


class WebOutputTee:
    def __init__(self, original):
        self.original = original

    def write(self, text):
        self.original.write(text)
        WEB_TRAINING_STATUS["output"] += text

    def flush(self):
        self.original.flush()


def run_training_for_web():
    WEB_TRAINING_STATUS.update(
        {
            "started": True,
            "finished": False,
            "message": "Main training code is running.",
            "output": "",
        }
    )

    try:
        with contextlib.redirect_stdout(WebOutputTee(sys.stdout)):
            ensure_folders()

            if not os.path.exists(DATA_ROOT):
                print(f"Dataset folder '{DATA_ROOT}' not found. Demo prediction is active.")
                WEB_TRAINING_STATUS.update(
                    {
                        "finished": True,
                        "message": f"Dataset folder '{DATA_ROOT}' not found. Demo prediction is active.",
                    }
                )
                return

            args = argparse.Namespace(
                target_accuracy=TARGET_ACCURACY,
                thyroid_sample=50000,
                predict=False,
                web=False,
                pca=True,
            )

            print("\nMULTI DISEASE PREDICTION SYSTEM")
            print("Efficient training mode started.\n")

            results = []
            for disease, filename in DATASET_FILES.items():
                path = os.path.join(DATA_ROOT, filename)
                if not os.path.exists(path):
                    print(f"Skipping {disease}: missing {path}")
                    continue

                print(f"Training {disease}...")
                result = train_one_dataset(disease, path, args)
                results.append(result)
                print(
                    f"  Best: {result['model']} | Accuracy: {result['accuracy']:.3f} | "
                    f"F1: {result['f1']:.3f} | {result['seconds']:.1f}s"
                )

            if not results:
                print("No datasets were trained.")
                WEB_TRAINING_STATUS.update(
                    {
                        "finished": True,
                        "message": "No datasets were trained. Demo prediction is active.",
                    }
                )
                return

            accuracy_plot = plot_accuracy_summary(results, args.target_accuracy)
            pd.DataFrame(results).drop(columns=["classification_report", "plots"]).to_csv(
                os.path.join(GRAPHS_DIR, "metrics_summary.csv"),
                index=False,
            )

            print_summary(results, args.target_accuracy)
            print(f"\nPlots saved in: {GRAPHS_DIR}")
            print(f"Accuracy summary: {accuracy_plot}")
            print(f"Metrics CSV: {os.path.join(GRAPHS_DIR, 'metrics_summary.csv')}")
            print("\nTraining completed successfully.")

        WEB_TRAINING_STATUS.update(
            {
                "finished": True,
                "message": "Main training completed. Model prediction is active.",
            }
        )
    except Exception as exc:
        WEB_TRAINING_STATUS.update(
            {
                "finished": True,
                "message": f"Training error: {exc}. Demo prediction is active.",
            }
        )
        print(f"Automatic web training error: {exc}")


def start_web_training_once():
    if WEB_TRAINING_STATUS["started"]:
        return

    import threading

    thread = threading.Thread(target=run_training_for_web, daemon=True)
    thread.start()


def run_web_app():
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:
        raise ImportError("Flask is required for web mode. Install it with: pip install flask") from exc

    app = Flask(__name__)

    # User storage file
    USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

    def load_users():
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_users(users):
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)

    @app.route("/")
    def index():
        return WEB_PAGE

    @app.route("/register", methods=["POST"])
    def register_route():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        email = str(payload.get("email", "")).strip()
        password = str(payload.get("password", ""))

        if not username or not email or not password:
            return jsonify({"ok": False, "message": "All fields are required."})

        if len(username) < 3:
            return jsonify({"ok": False, "message": "Username must be at least 3 characters."})

        if len(password) < 4:
            return jsonify({"ok": False, "message": "Password must be at least 4 characters."})

        users = load_users()

        if username.lower() in {u.lower() for u in users.keys()}:
            return jsonify({"ok": False, "message": "Username already exists."})

        import hashlib
        hashed = hashlib.sha256(password.encode()).hexdigest()
        users[username] = {"email": email, "password": hashed}
        save_users(users)

        return jsonify({"ok": True, "message": "Registration successful."})

    @app.route("/login", methods=["POST"])
    def login_route():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))

        if not username or not password:
            return jsonify({"ok": False, "message": "Please enter username and password."})

        users = load_users()

        # Case-insensitive username lookup
        matched_user = None
        for u, data in users.items():
            if u.lower() == username.lower():
                matched_user = data
                break

        if matched_user is None:
            return jsonify({"ok": False, "message": "Invalid username or password."})

        import hashlib
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if matched_user["password"] != hashed:
            return jsonify({"ok": False, "message": "Invalid username or password."})

        return jsonify({"ok": True, "message": "Login successful."})

    @app.route("/start-training", methods=["POST"])
    def start_training_route():
        start_web_training_once()
        return jsonify(WEB_TRAINING_STATUS)

    @app.route("/training-status")
    def training_status_route():
        return jsonify(WEB_TRAINING_STATUS)

    @app.route("/features/<disease_choice>")
    def features(disease_choice):
        disease_choice = disease_choice.strip().lower()
        if disease_choice not in DATASET_FILES:
            return jsonify({"ok": False, "message": "Invalid disease selected."})

        bundle = load_prediction_bundle(disease_choice)
        if bundle is None:
            return jsonify({"ok": True, "features": DEMO_FEATURES[disease_choice]})

        return jsonify({"ok": True, "features": bundle["features"]})

    @app.route("/predict", methods=["POST"])
    def predict_route():
        payload = request.get_json(silent=True) or {}
        disease_choice = str(payload.get("disease", "")).strip().lower()
        values = payload.get("values", {})

        if disease_choice not in DATASET_FILES:
            return jsonify({"ok": False, "message": "Invalid disease selected."})

        return jsonify(predict_web(disease_choice, values))

    @app.route("/api/metrics")
    def api_metrics():
        """Return model metrics data as JSON.

        Query params:
            disease (optional): Filter to a specific disease

        Returns:
            JSON with structure: {"ok": true, "metrics": [...]}
        """
        disease_filter = request.args.get("disease", "").strip().lower() or None

        # Validate disease filter if provided
        if disease_filter and disease_filter not in DATASET_FILES:
            return jsonify({"ok": False, "message": "Invalid disease specified."})

        # Determine which diseases to load
        diseases_to_load = [disease_filter] if disease_filter else list(DATASET_FILES.keys())

        # Try to parse the CSV file
        csv_path = os.path.join(GRAPHS_DIR, "metrics_summary.csv")
        csv_data = {}
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    d = str(row.get("disease", "")).strip().lower()
                    if d in DATASET_FILES:
                        csv_data[d] = {
                            "disease": d,
                            "display_name": DISEASE_DISPLAY_NAMES.get(d, d.title()),
                            "model": str(row.get("model", "Unknown")),
                            "original_features": int(row.get("original_features", 0)),
                            "pca_components": int(row.get("pca_components", 0)) if pd.notna(row.get("pca_components")) else None,
                            "accuracy": round(float(row.get("accuracy", 0)), 4),
                            "precision": round(float(row.get("precision", 0)), 4),
                            "recall": round(float(row.get("recall", 0)), 4),
                            "f1": round(float(row.get("f1", 0)), 4),
                        }
        except Exception:
            csv_data = {}

        # Build metrics list with CSV-first, pickle-fallback strategy
        metrics = []
        for disease in diseases_to_load:
            # Priority 1: CSV data
            if disease in csv_data:
                metrics.append(csv_data[disease])
                continue

            # Priority 2: Pickle file fallback
            pickle_path = os.path.join(MODELS_DIR, f"{disease}_metrics.pkl")
            if os.path.exists(pickle_path):
                try:
                    pkl_data = joblib.load(pickle_path)
                    metrics.append({
                        "disease": disease,
                        "display_name": DISEASE_DISPLAY_NAMES.get(disease, disease.title()),
                        "model": str(pkl_data.get("model", "Unknown")),
                        "original_features": int(pkl_data.get("original_features", 0)),
                        "pca_components": int(pkl_data.get("pca_components", 0)) if pkl_data.get("pca_components") is not None else None,
                        "accuracy": round(float(pkl_data.get("accuracy", 0)), 4),
                        "precision": round(float(pkl_data.get("precision", 0)), 4),
                        "recall": round(float(pkl_data.get("recall", 0)), 4),
                        "f1": round(float(pkl_data.get("f1", 0)), 4),
                    })
                except Exception:
                    pass

        # If filtering by disease and nothing found, report unavailable
        if disease_filter and not metrics:
            return jsonify({"ok": False, "message": f"Metrics data not available for {DISEASE_DISPLAY_NAMES.get(disease_filter, disease_filter.title())}. Model has not been trained yet."})

        # If no metrics found at all
        if not metrics:
            return jsonify({"ok": False, "message": "Metrics data not available. Models have not been trained yet."})

        return jsonify({"ok": True, "metrics": metrics})

    @app.route("/graphs/<filename>")
    def serve_graph(filename):
        """Serve PNG/CSV files from graphs/current/ directory.

        Returns:
            - 200 with appropriate content type on success
            - 400 if path traversal detected
            - 404 if file not found or invalid extension
        """
        from flask import abort, send_from_directory

        # Reject path traversal characters
        if ".." in filename or "/" in filename or "\\" in filename:
            abort(400)

        # Only serve .png or .csv files
        if not (filename.endswith(".png") or filename.endswith(".csv")):
            abort(404)

        # Check file existence
        file_path = os.path.join(GRAPHS_DIR, filename)
        if not os.path.isfile(file_path):
            abort(404)

        # Serve the file with correct content type
        if filename.endswith(".png"):
            return send_from_directory(GRAPHS_DIR, filename, mimetype="image/png")
        else:
            return send_from_directory(GRAPHS_DIR, filename, mimetype="text/csv")

    print("\nLogin page running at: http://127.0.0.1:5000")
    print("Login username: Rachitha")
    print("Login password: 1234\n")
    app.run(host="127.0.0.1", port=5000, debug=False)


def interactive_prediction():

    print("\nENTER PATIENT DETAILS\n")

    patient_data = {}

    fields = [
        "Age",
        "Glucose",
        "BMI",
        "BloodPressure",
        "Albumin",
        "Serum_Creatinine",
        "TSH",
        "T3",
        "Total_Bilirubin",
        "Cholesterol"
    ]

    for field in fields:
        try:
            patient_data[field] = float(input(f"{field}:"))
        except:
            patient_data[field] = 0.0

    print("\n======== PREDICTION RESULT==============\n")

    disease_names = DISEASE_DISPLAY_NAMES
    for disease in DATASET_FILES.keys():
        try:
            model_path = os.path.join(MODELS_DIR, f"{disease}_model.pkl")
            model = joblib.load(model_path)
            scaler = joblib.load(os.path.join(MODELS_DIR, f"{disease}_scaler.pkl"))
            imputer = joblib.load(os.path.join(MODELS_DIR, f"{disease}_imputer.pkl"))
            pca = joblib.load(os.path.join(MODELS_DIR, f"{disease}_pca.pkl"))
            features = joblib.load(os.path.join(MODELS_DIR, f"{disease}_features.pkl"))

            row = []
            for feature in features:
                row.append(patient_data.get(feature, 0))

            row = np.array(row).reshape(1, -1)

            row = imputer.transform(row)
            row = scaler.transform(row)

            # Apply PCA transformation if available
            if pca:
                row = pca.transform(row)

            prediction = model.predict(row)[0]

            if prediction == 1:
                print(f"{disease_names[disease]} DETECTED")
            else:
                print(f"NO {disease_names[disease]} DETECTED")
        except Exception as e:
            print(f"{disease_names[disease]} ERROR : {e}")

    print("\n==================================================")


def main():
    args = parse_args()

    if args.web:
        run_web_app()
        return

    ensure_folders()

    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(
            f"Dataset folder '{DATA_ROOT}' not found. Keep the data folder beside main.py."
        )

    print("\nMULTI DISEASE PREDICTION SYSTEM")
    print("Efficient training mode started.\n")

    results = []
    for disease, filename in DATASET_FILES.items():
        path = os.path.join(DATA_ROOT, filename)
        if not os.path.exists(path):
            print(f"Skipping {disease}: missing {path}")
            continue

        print(f"Training {disease}...")
        result = train_one_dataset(disease, path, args)
        results.append(result)
        print(
            f"  Best: {result['model']} | Accuracy: {result['accuracy']:.3f} | "
            f"F1: {result['f1']:.3f} | {result['seconds']:.1f}s"
        )

    if not results:
        print("No datasets were trained.")
        return

    accuracy_plot = plot_accuracy_summary(results, args.target_accuracy)
    pd.DataFrame(results).drop(columns=["classification_report", "plots"]).to_csv(
        os.path.join(GRAPHS_DIR, "metrics_summary.csv"),
        index=False,
    )

    print_summary(results, args.target_accuracy)
    print(f"\nPlots saved in: {GRAPHS_DIR}")
    print(f"Accuracy summary: {accuracy_plot}")
    print(f"Metrics CSV: {os.path.join(GRAPHS_DIR, 'metrics_summary.csv')}")
    print("\nTraining completed successfully.")

    if args.predict:
        interactive_prediction()


if __name__ == "__main__":
    main()