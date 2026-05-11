import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler

# ---------------- LOAD DATASET ---------------- #

print("Loading dataset...")

df = pd.read_csv("data/processed/sepsis_dataset.csv")

# ---------------- FEATURE ENGINEERING ---------------- #

# Shock Index = Heart Rate / Systolic BP
df["shock_index"] = (
    df["heart_rate"] /
    (df["systolic_bp"] + 1)
)

# Pulse Pressure = Systolic BP - Diastolic BP
df["pulse_pressure"] = (
    df["systolic_bp"] -
    df["diastolic_bp"]
)

# ---------------- SELECT FEATURES ---------------- #

feature_cols = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "shock_index",
    "pulse_pressure",
    "respiratory_rate",
    "spo2",
    "gender",
    "anchor_age"
]

X = df[feature_cols]
y = df["sepsis"]

# ---------------- CLEAN DATA ---------------- #

# replace infinity values
X = X.replace([np.inf, -np.inf], np.nan)

# fill missing values
X = X.fillna(X.mean())

# ---------------- TRAIN TEST SPLIT ---------------- #

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- FEATURE SCALING ---------------- #

# scaling is optional for Random Forest but kept for consistency

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- RANDOM FOREST MODEL ---------------- #

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ---------------- #

print("Making predictions...")

y_pred = model.predict(X_test)

# ---------------- EVALUATION ---------------- #

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- FEATURE IMPORTANCE ---------------- #

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)