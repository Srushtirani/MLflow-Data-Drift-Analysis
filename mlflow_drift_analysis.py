
import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


os.makedirs("artifacts", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Breast_Cancer_Data_Drift_Analysis")


dataset = load_breast_cancer(as_frame=True)

X = dataset.data.copy()
y = dataset.target.copy()

feature_names = list(X.columns)

print("\nDATASET INFORMATION")
print("-------------------")
print("Dataset: Breast Cancer Wisconsin Diagnostic")
print("Number of records:", len(X))
print("Number of features:", X.shape[1])
print("Features:", feature_names)
print("Target classes:", list(dataset.target_names))
print("\nTarget distribution:")
print(y.value_counts().rename(index={
    0: "malignant",
    1: "benign"
}))
print("\nFirst five records:")
print(X.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\nMODEL EVALUATION")
print("----------------")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=dataset.target_names,
    zero_division=0
))



reference_data = X.copy()


current_data = X.copy()

current_data["mean radius"] += 5.0
current_data["mean texture"] += 5.0

print("\nSimulated drift introduced in:")
print("- mean radius: +5.0")
print("- mean texture: +5.0")




threshold = 0.05
drift_results = []

for feature in feature_names:
    statistic, p_value = ks_2samp(
        reference_data[feature],
        current_data[feature]
    )

    drift_detected = p_value < threshold

    drift_results.append({
        "Feature": feature,
        "KS Statistic": statistic,
        "P-value": p_value,
        "Drift Detected": drift_detected
    })

results_df = pd.DataFrame(drift_results)

print("\nDATA DRIFT RESULTS")
print("------------------")
print(results_df.to_string(index=False))

drift_count = int(results_df["Drift Detected"].sum())
any_drift = drift_count > 0

print("\nFeatures analyzed:", len(feature_names))
print("Features with detected drift:", drift_count)

if any_drift:
    print("Conclusion: Data drift detected.")
else:
    print("Conclusion: No significant data drift detected.")


results_df.to_csv(
    "artifacts/drift_results.csv",
    index=False
)

with open("artifacts/drift_summary.json", "w") as file:
    json.dump({
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "records": len(X),
        "features_analyzed": len(feature_names),
        "drift_type": "Data Drift",
        "detection_method": "Two-sample Kolmogorov-Smirnov test",
        "significance_threshold": threshold,
        "simulated_drift": True,
        "features_with_drift": drift_count,
        "any_drift_detected": any_drift,
        "modified_features": {
            "mean radius": 5.0,
            "mean texture": 5.0
        }
    }, file, indent=4)




fig, axes = plt.subplots(5, 6, figsize=(18, 14))
axes = axes.flatten()

for ax, feature in zip(axes, feature_names):
    ax.hist(reference_data[feature], bins=20)
    ax.set_title(feature, fontsize=8)
    ax.set_xlabel("Value", fontsize=7)
    ax.set_ylabel("Frequency", fontsize=7)

plt.suptitle("Reference Dataset Feature Distributions")
plt.tight_layout()
plt.savefig(
    "artifacts/reference_distributions.png",
    dpi=150
)
plt.close()

fig, axes = plt.subplots(5, 6, figsize=(18, 14))
axes = axes.flatten()

for ax, feature in zip(axes, feature_names):
    ax.hist(
        reference_data[feature],
        bins=20,
        alpha=0.55,
        label="Reference"
    )
    ax.hist(
        current_data[feature],
        bins=20,
        alpha=0.55,
        label="Current"
    )
    ax.set_title(feature, fontsize=8)
    ax.set_xlabel("Value", fontsize=7)
    ax.set_ylabel("Frequency", fontsize=7)
    ax.legend(fontsize=6)

plt.suptitle("Reference vs Current Feature Distributions")
plt.tight_layout()
plt.savefig(
    "artifacts/reference_vs_current.png",
    dpi=150
)
plt.close()


with mlflow.start_run(
    run_name="RandomForest_BreastCancer_Drift"
):

    mlflow.log_param(
        "dataset",
        "Breast Cancer Wisconsin Diagnostic"
    )
    mlflow.log_param(
        "model",
        "RandomForestClassifier"
    )
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Log model performance metrics.
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)

    # Log drift detection configuration and results.
    mlflow.log_param(
        "drift_method",
        "KS two-sample test"
    )
    mlflow.log_param(
        "significance_threshold",
        threshold
    )
    mlflow.log_param("simulated_drift", True)

    mlflow.log_metric(
        "features_analyzed",
        len(feature_names)
    )
    mlflow.log_metric(
        "features_with_drift",
        drift_count
    )

    # Log reports and visualizations.
    mlflow.log_artifact(
        "artifacts/drift_results.csv"
    )
    mlflow.log_artifact(
        "artifacts/drift_summary.json"
    )
    mlflow.log_artifact(
        "artifacts/reference_distributions.png"
    )
    mlflow.log_artifact(
        "artifacts/reference_vs_current.png"
    )

    # Log trained model.
    mlflow.sklearn.log_model(
    sk_model=model,
    name="breast_cancer_random_forest_model",
    serialization_format="pickle"
)

    print("\nMLflow run ID:")
    print(mlflow.active_run().info.run_id)


print("\nExperiment completed successfully.")
print("MLflow database: mlflow.db")
print("Results folder: artifacts/")
print("Open the MLflow UI at http://127.0.0.1:5000")
