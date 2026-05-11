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

# load chart events first to get filtered patient list
print("Loading chart events...")
chunks = []

for chunk in pd.read_csv(
    CHARTEVENTS_PATH,
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "valuenum"
    ],
    chunksize=500000
):

    # keep only required vital signs
    chunk = chunk[
        chunk["itemid"].isin(VITAL_SIGNS.keys())
    ]

    # remove missing values
    chunk = chunk.dropna(subset=["valuenum"])

    # map itemid to vital name
    chunk["vital_name"] = chunk["itemid"].map(VITAL_SIGNS)

    chunks.append(chunk)

chartevents = pd.concat(chunks, ignore_index=True)

# get only relevant patients
filtered_subjects = chartevents["subject_id"].unique()

print("Unique patients:", len(filtered_subjects))

# load patients table filtered
print("Loading patients...")
patients = pd.read_csv(
    PATIENTS_PATH,
    usecols=["subject_id", "gender", "anchor_age"]
)

patients = patients[
    patients["subject_id"].isin(filtered_subjects)
]

# load admissions filtered
print("Loading admissions...")
admissions = pd.read_csv(
    ADMISSIONS_PATH,
    usecols=["subject_id", "hadm_id", "admittime", "dischtime"]
)

admissions = admissions[
    admissions["subject_id"].isin(filtered_subjects)
]

# load icu stays filtered
print("Loading ICU stays...")
icustays = pd.read_csv(
    ICUSTAYS_PATH,
    usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
)

icustays = icustays[
    icustays["subject_id"].isin(filtered_subjects)
]

# load diagnoses filtered
print("Loading diagnoses...")
diagnoses = pd.read_csv(
    DIAGNOSES_PATH,
    usecols=["subject_id", "hadm_id", "icd_code"]
)

diagnoses = diagnoses[
    diagnoses["subject_id"].isin(filtered_subjects)
]

# clean icd codes
diagnoses["icd_code"] = (
    diagnoses["icd_code"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# create sepsis labels
diagnoses["sepsis"] = diagnoses["icd_code"].apply(
    lambda x: 1 if any(
        x.startswith(code) for code in SEPSIS_CODES
    ) else 0
)

# group labels per admission
labels = diagnoses.groupby(
    ["subject_id", "hadm_id"]
)["sepsis"].max().reset_index()

# include all admissions
labels = admissions[
    ["subject_id", "hadm_id"]
].merge(
    labels,
    on=["subject_id", "hadm_id"],
    how="left"
)

labels["sepsis"] = labels["sepsis"].fillna(0).astype(int)

# convert datetime columns
print("Converting datetime...")
chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
icustays["intime"] = pd.to_datetime(icustays["intime"])
icustays["outtime"] = pd.to_datetime(icustays["outtime"])

# aggregate to hourly level
print("Aggregating to hourly...")
chartevents["hour"] = chartevents["charttime"].dt.floor("h")

chartevents = chartevents.groupby(
    ["subject_id", "hadm_id", "stay_id", "hour", "vital_name"]
)["valuenum"].mean().reset_index()

# pivot to wide format
print("Pivoting vitals...")
vitals = chartevents.pivot_table(
    index=["subject_id", "hadm_id", "stay_id", "hour"],
    columns="vital_name",
    values="valuenum"
).reset_index()

vitals.columns.name = None

# rename hour back to charttime
vitals = vitals.rename(columns={"hour": "charttime"})

# merge all tables
print("Merging tables...")
merged = vitals.merge(labels, on=["subject_id", "hadm_id"], how="left")
merged = merged.merge(patients, on="subject_id", how="left")
merged = merged.merge(admissions, on=["subject_id", "hadm_id"], how="left")
merged = merged.merge(icustays, on=["subject_id", "hadm_id", "stay_id"], how="left")

# clean dataset
print("Cleaning dataset...")
merged["sepsis"] = merged["sepsis"].fillna(0).astype(int)

merged["gender"] = merged["gender"].map({"M": 1, "F": 0}).fillna(0)

# sort values
merged = merged.sort_values(by=["subject_id", "charttime"])

# forward + backward fill within each ICU stay
vital_cols = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "respiratory_rate",
    "spo2"
]

merged[vital_cols] = (
    merged.groupby("stay_id")[vital_cols]
    .ffill()
    .bfill()
)
# merged = merged.groupby("stay_id").apply(lambda x: x.ffill().bfill()).reset_index(drop=True)

# save dataset
print("Saving dataset...")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

merged.to_csv(FINAL_DATASET_PATH, index=False)

# final output
print("\nDataset created successfully")
print("\nShape:", merged.shape)
print("\nSepsis distribution:")
print(merged["sepsis"].value_counts())
print("\nSaved to:", FINAL_DATASET_PATH)