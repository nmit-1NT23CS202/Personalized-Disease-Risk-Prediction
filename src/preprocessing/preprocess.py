import pandas as pd

from config import (
    PATIENTS_PATH,
    ADMISSIONS_PATH,
    ICUSTAYS_PATH,
    DIAGNOSES_PATH,
    CHARTEVENTS_PATH,
    FINAL_DATASET_PATH,
    PROCESSED_DIR,
    VITAL_SIGNS,
    SEPSIS_CODES
)

# Load patient data

patients = pd.read_csv(
    PATIENTS_PATH,
    usecols=[
        "subject_id",
        "gender",
        "anchor_age"
    ]
)

# Load admission data

admissions = pd.read_csv(
    ADMISSIONS_PATH,
    usecols=[
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime"
    ]
)

# Load ICU stay data

icustays = pd.read_csv(
    ICUSTAYS_PATH,
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime"
    ]
)

# Load diagnosis data

diagnoses = pd.read_csv(
    DIAGNOSES_PATH,
    usecols=[
        "subject_id",
        "hadm_id",
        "icd_code"
    ]
)

# Convert ICD codes to string

diagnoses["icd_code"] = diagnoses["icd_code"].astype(str)

# Create sepsis labels

diagnoses["sepsis"] = diagnoses["icd_code"].apply(
    lambda x: 1 if any(
        x.startswith(code) for code in SEPSIS_CODES
    ) else 0
)

# Group labels by admission

labels = diagnoses.groupby(
    ["subject_id", "hadm_id"]
)["sepsis"].max().reset_index()

# Load vital sign data

chartevents = pd.read_csv(
    CHARTEVENTS_PATH,
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "valuenum"
    ]
)

# Keep required ITEMIDs

chartevents = chartevents[
    chartevents["itemid"].isin(VITAL_SIGNS.keys())
]

# Rename ITEMIDs

chartevents["vital_name"] = chartevents["itemid"].map(VITAL_SIGNS)

# Remove missing values

chartevents = chartevents.dropna(
    subset=["valuenum"]
)

# Convert chart time to datetime

chartevents["charttime"] = pd.to_datetime(
    chartevents["charttime"]
)

# Convert long format to wide format

vitals = chartevents.pivot_table(
    index=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime"
    ],
    columns="vital_name",
    values="valuenum",
    aggfunc="mean"
).reset_index()

# Merge vitals with labels

merged = vitals.merge(
    labels,
    on=["subject_id", "hadm_id"],
    how="left"
)

# Merge patient data

merged = merged.merge(
    patients,
    on="subject_id",
    how="left"
)

# Merge admission data

merged = merged.merge(
    admissions,
    on=["subject_id", "hadm_id"],
    how="left"
)

# Merge ICU stay data

merged = merged.merge(
    icustays,
    on=["subject_id", "hadm_id", "stay_id"],
    how="left"
)

# Fill missing labels

merged["sepsis"] = merged["sepsis"].fillna(0)

# Encode gender

merged["gender"] = merged["gender"].map({
    "M": 0,
    "F": 1
})

# Create processed folder

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Save final dataset

merged.to_csv(
    FINAL_DATASET_PATH,
    index=False
)

print("Dataset created successfully")

print("Final dataset shape:", merged.shape)

print("Saved to:", FINAL_DATASET_PATH)