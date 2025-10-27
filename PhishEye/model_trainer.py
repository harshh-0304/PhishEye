import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Or other classifier of your choice
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- 1. Load and Inspect Data (already done, but good to keep for reference) ---
# Assuming your dataset is named 'phishing_data.csv' and is in a 'data' folder
# Adjust the path if your file name or location is different
df = pd.read_csv('/home/kiyotaka/Downloads/project/PhishEye/dataset.csv')# <--- IMPORTANT: Update this file name if needed

print(df.head())
print(df.info())
print(df.columns)
print(df['Result'].value_counts()) # Check the distribution of your target variable

# --- 2. Data Preparation ---
# Your dataset already has numerical features, so we don't need to define
# new feature extraction functions for URL parsing.

# Define features (X) and target (y)
# We'll drop 'index' as it's just an identifier and 'Result' as it's the target.
X = df.drop(['index', 'Result'], axis=1)
y = df['Result']

# It's good practice to ensure the target variable is 0 or 1.
# Based on your output, Result is -1 or 1. Let's map -1 to 0 (phishing) and 1 to 1 (safe).
# You can reverse this if your interpretation of -1 and 1 is different.
y = y.map({-1: 0, 1: 1})
print("\nMapped 'Result' values:")
print(y.value_counts())

# --- 3. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- 4. Train a Machine Learning Model ---
# Using RandomForestClassifier as it's a good general-purpose model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel training complete.")

# --- 5. Evaluate the Model ---
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Save the Trained Model and Feature Names ---
# It's crucial to save the list of feature names in the correct order,
# as the Streamlit app will need to create input in this exact order.
model_filename = 'phishing_detector_model.pkl'
joblib.dump(model, model_filename)

# Save the feature names separately
feature_names_filename = 'feature_names.pkl'
joblib.dump(X.columns.tolist(), feature_names_filename)

print(f"\nModel saved as {model_filename}")
print(f"Feature names saved as {feature_names_filename}")

# Optional: Print feature importances for reasoning
print("\nFeature Importances (Top 10):")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.nlargest(10))