import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler

# load dataset
print("Loading dataset...")

df = pd.read_csv("data/processed/sepsis_dataset.csv")

df["shock_index"] = (
    df["heart_rate"] /
    df["systolic_bp"]
)

df["pulse_pressure"] = (
    df["systolic_bp"] -
    df["diastolic_bp"]
)


# select features
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

# replace inf values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# fill missing values
X = X.fillna(X.mean())

# split dataset
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create model
print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# predictions
print("Making predictions...")

y_pred = model.predict(X_test)

coefficients = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_[0]
})

print(coefficients)

# evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))