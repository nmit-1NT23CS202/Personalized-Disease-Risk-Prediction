from pathlib import Path

# Project directories

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

# Raw data files

PATIENTS_PATH = RAW_DIR / "patients.csv"

ADMISSIONS_PATH = RAW_DIR / "admissions.csv"

ICUSTAYS_PATH = RAW_DIR / "icustays.csv"

DIAGNOSES_PATH = RAW_DIR / "diagnoses_icd.csv"

CHARTEVENTS_PATH = RAW_DIR / "chart_events.csv"

# Output file

FINAL_DATASET_PATH = PROCESSED_DIR / "sepsis_dataset.csv"

# ITEMIDs obtained using MIMIC-IV dictionary tables like d_labitems

VITAL_SIGNS = {
    220045: "heart_rate",
    220179: "systolic_bp",
    220180: "diastolic_bp",
    220181: "mean_bp",
    220210: "respiratory_rate",
    223762: "temperature",
    220277: "spo2"
}

# Sepsis ICD codes

SEPSIS_CODES = [
    "99591",
    "99592",
    "A41"
]