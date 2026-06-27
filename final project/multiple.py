import argparse
import contextlib
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


# ─────────────────────────────────────────────────────────────────────────────
# WEB PAGE  –  BUG FIXES:
#   1. Added missing closing `}` after the if(!data.started) block
#   2. Removed window.onload that was calling checkTrainingStatus() before login
# ─────────────────────────────────────────────────────────────────────────────
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
    box-sizing:border-box;
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
    white-space:pre-wrap;
    max-height:400px;
    overflow-y:auto;
    border:1px solid #ccc;
    padding:10px;
}
</style>
</head>

<body>

<!-- LOGIN -->
<div id="loginPage">
    <div class="login-box">
        <h2>Login</h2>

        <input type="text" id="username" placeholder="Username">
        <input type="password" id="password" placeholder="Password">

        <button onclick="login()">Login</button>
        <p id="loginMsg"></p>
    </div>
</div>

<!-- DASHBOARD -->
<div id="dashboard">

<h2>Multiple Disease Prediction System</h2>

<div id="trainingStatus">Checking training status...</div>

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

<button onclick="logout()">Logout</button>

</div>

<script>

function login(){
    let u = document.getElementById("username").value;
    let p = document.getElementById("password").value;

    if(u==="Rachitha" && p==="1234"){
        document.getElementById("loginPage").style.display="none";
        document.getElementById("dashboard").style.display="block";
        checkTrainingStatus();   // ✅ Only called AFTER login
    } else {
        document.getElementById("loginMsg").innerHTML="Invalid Login!";
    }
}

async function checkTrainingStatus(){
    document.getElementById("trainingStatus").innerHTML = "Checking training status...";

    try{
        const response = await fetch("/training-status");
        const data = await response.json();

        if(!data.started){
            document.getElementById("trainingStatus").innerHTML =
                "Training not started...";
            setTimeout(checkTrainingStatus, 2000);
            return;
        }  // ✅ FIX: closing brace was missing here — caused if(!data.finished) to be unreachable

        if(!data.finished){
            document.getElementById("trainingStatus").innerHTML =
                "<pre>" + data.output + "</pre>";
            setTimeout(checkTrainingStatus, 2000);
            return;
        }

        document.getElementById("trainingStatus").innerHTML =
            "<pre>" + data.output + "</pre><br>✅ Training Completed";

    } catch(err){
        document.getElementById("trainingStatus").innerHTML =
            "Status check failed: " + err;
    }
}

async function predictOne(disease, values){
    const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({disease, values})
    });
    const data = await response.json();
    return data.message || "Error";
}

async function predictAll(){

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

    const order = ["diabetes", "kidney", "heart", "liver", "thyroid"];
    let lines = [];

    for(const disease of order){
        const msg = await predictOne(disease, values);
        const icon = msg.toLowerCase().startsWith("no") ? "🟢" : "🔴";
        lines.push(`${icon} ${msg}`);
    }

    document.getElementById("result").innerHTML = lines.join("<br>");
}

function logout(){
    location.reload();
}

// ✅ FIX: Removed window.onload that was calling checkTrainingStatus() before login

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
    if bundle["pca"]:
        input_array = bundle["pca"].transform(input_array)

    prediction = bundle["model"].predict(input_array)[0]

    if int(prediction) == 1:
        return {"ok": True, "message": f"{name} Detected"}
    return {"ok": True, "message": f"No {name}"}


# Field names used by the dashboard form
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

    if disease_choice == "diabetes":
        detected = read_demo_value(values, "Glucose") > 126 or read_demo_value(values, "BMI") > 30
        return {"ok": True, "message": f"{name} Detected" if detected else f"No {name}"}

    if disease_choice == "kidney":
        detected = (
            read_demo_value(values, "BloodPressure") > 140
            or read_demo_value(values, "Serum_Creatinine") > 1.5
        )
        return {"ok": True, "message": f"{name} Detected" if detected else f"No {name}"}

    if disease_choice == "heart":
        detected = (
            read_demo_value(values, "Age") > 50
            or read_demo_value(values, "Cholesterol") > 200
            or read_demo_value(values, "BloodPressure") > 140
        )
        return {"ok": True, "message": f"{name} Detected" if detected else f"No {name}"}

    if disease_choice == "liver":
        detected = (
            read_demo_value(values, "Total_Bilirubin") > 1.2
            or read_demo_value(values, "Albumin") < 3.5
        )
        return {"ok": True, "message": f"{name} Detected" if detected else f"No {name}"}

    if disease_choice == "thyroid":
        detected = read_demo_value(values, "TSH") > 4.5 or read_demo_value(values, "T3") < 1.0
        return {"ok": True, "message": f"{name} Detected" if detected else f"No {name}"}

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
            "message": "Training in progress...",
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

    @app.route("/")
    def index():
        return WEB_PAGE

    @app.route("/start-training", methods=["POST"])
    def start_training_route():
        start_web_training_once()
        return jsonify(WEB_TRAINING_STATUS)

    @app.route("/training-status")
    def training_status_route():
        return jsonify(
            started=WEB_TRAINING_STATUS["started"],
            finished=WEB_TRAINING_STATUS["finished"],
            message=WEB_TRAINING_STATUS["message"],
            output=WEB_TRAINING_STATUS["output"],
        )

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

    # Always retrain fresh on every web launch
    start_web_training_once()

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
        "Cholesterol",
    ]

    for field in fields:
        try:
            patient_data[field] = float(input(f"{field}: "))
        except Exception:
            patient_data[field] = 0.0

    print("\n======== PREDICTION RESULT ==============\n")

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