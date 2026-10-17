import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def find_dataset_path():
    """Locate dataset.csv robustly across relative and parent paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, 'dataset.csv'),
        os.path.join(base_dir, '..', 'dataset.csv'),
        'dataset.csv',
        'PhishEye/dataset.csv'
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"dataset.csv not found in candidate locations: {candidates}")


def train_model():
    dataset_path = find_dataset_path()
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print(f"Dataset shape: {df.shape}")
    print(df['Result'].value_counts())

    # --- Data Preparation ---
    X = df.drop(['index', 'Result'], axis=1, errors='ignore')
    y = df['Result']

    # Map -1 to 0 (phishing) and 1 to 1 (safe)
    y = y.map({-1: 0, 1: 1})
    print("\nMapped 'Result' distribution (0=Phishing, 1=Safe):")
    print(y.value_counts())

    # --- Train / Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # --- Model Training ---
    print("\nTraining RandomForestClassifier (n_estimators=100)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Phishing", "Safe"]))

    # --- Save Artifacts ---
    target_dir = os.path.dirname(dataset_path)
    model_path = os.path.join(target_dir, 'phishing_detector_model.pkl')
    features_path = os.path.join(target_dir, 'feature_names.pkl')

    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), features_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved feature names to: {features_path}")

    # Top 10 feature importances
    print("\nTop 10 Feature Importances:")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print(importances.nlargest(10))

    return model, X.columns.tolist()


if __name__ == '__main__':
    train_model()