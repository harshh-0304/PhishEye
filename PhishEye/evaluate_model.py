"""
PhishEye Model Evaluation & Benchmarking Suite.

Performs stratified k-fold cross-validation, evaluates ROC-AUC, Precision, Recall,
and exports confusion matrix analytics.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def find_dataset():
    candidates = [
        "dataset.csv",
        os.path.join(os.path.dirname(__file__), "dataset.csv"),
        os.path.join("PhishEye", "dataset.csv")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("dataset.csv not found.")


def evaluate():
    dataset_path = find_dataset()
    print(f"Loading evaluation dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    X = df.drop(["index", "Result"], axis=1, errors="ignore")
    y = df["Result"].map({-1: 0, 1: 1})

    model_path = os.path.join(os.path.dirname(dataset_path), "phishing_detector_model.pkl")
    model = joblib.load(model_path)

    print("\nExecuting 5-Fold Stratified Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_roc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"  CV Accuracy: {np.mean(cv_scores) * 100:.2f}% (+/- {np.std(cv_scores) * 100:.2f}%)")
    print(f"  CV ROC-AUC:  {np.mean(cv_roc_scores) * 100:.2f}% (+/- {np.std(cv_roc_scores) * 100:.2f}%)")

    # In-sample predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_roc_auc_mean": float(np.mean(cv_roc_scores)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    }

    out_json = os.path.join(os.path.dirname(dataset_path), "evaluation_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved evaluation metrics to: {out_json}")
    print(f"Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision Score:   {metrics['precision'] * 100:.2f}%")
    print(f"Recall Score:      {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:          {metrics['f1'] * 100:.2f}%")
    print(f"ROC-AUC Score:     {metrics['roc_auc'] * 100:.2f}%")

    return metrics


if __name__ == "__main__":
    evaluate()
