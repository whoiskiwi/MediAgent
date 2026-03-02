"""
Agent 1 Data Processing — Disease Symptom Prediction Dataset
Inputs:
  data/raw/disease_symptom_prediction/dataset.csv
  data/raw/disease_symptom_prediction/Symptom-severity.csv
  data/raw/disease_symptom_prediction/symptom_precaution.csv
  data/raw/disease_symptom_prediction/symptom_Description.csv
Outputs:
  data/processed/symptom_classifier_train.jsonl
  data/processed/symptom_classifier_test.jsonl
  data/processed/department_mapping.json
  data/processed/disease_precautions.json   ← used by Agent 3
"""

import pandas as pd
import json
import random
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RAW = _HERE / "raw" / "disease_symptom_prediction"
OUT = _HERE / "processed"

# ── Disease → Department mapping (exact names from CSV) ──────────────────────
DISEASE_MAP = {
    # Cardiology
    "Heart attack":                           "Cardiology",
    "Hypertension":                           "Cardiology",
    "Varicose veins":                         "Cardiology",

    # Neurology
    "Migraine":                               "Neurology",
    "Paralysis (brain hemorrhage)":           "Neurology",
    "(vertigo) Paroymsal  Positional Vertigo":"Neurology",
    "Cervical spondylosis":                   "Neurology",

    # Dermatology
    "Fungal infection":                       "Dermatology",
    "Drug Reaction":                          "Dermatology",
    "Chicken pox":                            "Dermatology",
    "Acne":                                   "Dermatology",
    "Psoriasis":                              "Dermatology",
    "Impetigo":                               "Dermatology",

    # Gastroenterology
    "GERD":                                   "Gastroenterology",
    "Chronic cholestasis":                    "Gastroenterology",
    "Peptic ulcer diseae":                    "Gastroenterology",
    "Gastroenteritis":                        "Gastroenterology",
    "Jaundice":                               "Gastroenterology",
    "hepatitis A":                            "Gastroenterology",
    "Hepatitis B":                            "Gastroenterology",
    "Hepatitis C":                            "Gastroenterology",
    "Hepatitis D":                            "Gastroenterology",
    "Hepatitis E":                            "Gastroenterology",
    "Alcoholic hepatitis":                    "Gastroenterology",
    "Dimorphic hemmorhoids(piles)":           "Gastroenterology",

    # Endocrinology
    "Diabetes":                               "Endocrinology",
    "Hypothyroidism":                         "Endocrinology",
    "Hyperthyroidism":                        "Endocrinology",
    "Hypoglycemia":                           "Endocrinology",

    # Pulmonology
    "Bronchial Asthma":                       "Pulmonology",
    "Tuberculosis":                           "Pulmonology",
    "Pneumonia":                              "Pulmonology",
    "Common Cold":                            "Pulmonology",

    # Infectious Disease
    "AIDS":                                   "Infectious Disease",
    "Malaria":                                "Infectious Disease",
    "Dengue":                                 "Infectious Disease",
    "Typhoid":                                "Infectious Disease",

    # Orthopedics
    "Osteoarthristis":                        "Orthopedics",
    "Arthritis":                              "Orthopedics",

    # Urology
    "Urinary tract infection":                "Urology",

    # General Medicine
    "Allergy":                                "General Medicine",
}

# Diseases that are always Emergency regardless of symptom severity score
ALWAYS_EMERGENCY = {"Heart attack", "Paralysis (brain hemorrhage)"}

# Urgency thresholds based on summed symptom severity weights
# Calibrated against actual score distribution (mean=30, p75=42, p90=53)
URGENCY_THRESHOLDS = {
    "Emergency": 45,  # top ~25% → Emergency
    "Urgent":    20,  # mid ~50% → Urgent
    # below 20  → Routine
}


def load_severity_weights(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Symptom"] = df["Symptom"].str.strip()
    return dict(zip(df["Symptom"], df["weight"]))


def load_precautions(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    result = {}
    for _, row in df.iterrows():
        disease = str(row["Disease"]).strip()
        precautions = [
            str(row[c]).strip()
            for c in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
            if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")
        ]
        result[disease] = precautions
    return result


def compute_urgency(disease: str, row: pd.Series, severity_weights: dict) -> str:
    if disease in ALWAYS_EMERGENCY:
        return "Emergency"

    symptom_cols = [c for c in row.index if c.startswith("Symptom_")]
    total_severity = 0
    for col in symptom_cols:
        symptom = str(row[col]).strip()
        if symptom and symptom != "nan":
            total_severity += severity_weights.get(symptom, 0)

    if total_severity >= URGENCY_THRESHOLDS["Emergency"]:
        return "Emergency"
    elif total_severity >= URGENCY_THRESHOLDS["Urgent"]:
        return "Urgent"
    return "Routine"


def symptoms_to_text(row: pd.Series) -> str:
    symptom_cols = [c for c in row.index if c.startswith("Symptom_")]
    symptoms = [
        str(row[c]).strip().replace("_", " ")
        for c in symptom_cols
        if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")
    ]
    if not symptoms:
        return "The patient has unspecified symptoms."
    return "Patient reports: " + ", ".join(symptoms) + "."


def build_record(row: pd.Series, severity_weights: dict) -> dict | None:
    disease = str(row["Disease"]).strip()
    if disease not in DISEASE_MAP:
        return None
    department = DISEASE_MAP[disease]
    urgency = compute_urgency(disease, row, severity_weights)
    return {
        "instruction": (
            "You are a medical triage assistant. "
            "Classify the patient's symptoms into a medical department and urgency level. "
            "Respond in exactly this format:\n"
            "Department: <department>\nUrgency: <Routine|Urgent|Emergency>"
        ),
        "input": symptoms_to_text(row),
        "output": f"Department: {department}\nUrgency: {urgency}",
        "disease": disease,
        "department": department,
        "urgency": urgency,
    }


def main():
    print("Loading datasets...")
    df = pd.read_csv(RAW / "dataset.csv")
    severity_weights = load_severity_weights(RAW / "Symptom-severity.csv")
    precautions = load_precautions(RAW / "symptom_precaution.csv")

    print(f"  dataset.csv:        {len(df)} rows, {df['Disease'].nunique()} diseases")
    print(f"  Symptom-severity:   {len(severity_weights)} symptoms with weights")
    print(f"  symptom_precaution: {len(precautions)} diseases with precautions")

    records = []
    skipped = set()
    for _, row in df.iterrows():
        rec = build_record(row, severity_weights)
        if rec:
            records.append(rec)
        else:
            skipped.add(str(row["Disease"]).strip())

    if skipped:
        print(f"\nSkipped {len(skipped)} unmapped diseases: {skipped}")

    print(f"\nConverted {len(records)} rows")

    # Train / test split
    random.seed(42)
    random.shuffle(records)
    split = int(len(records) * 0.85)
    train, test = records[:split], records[split:]

    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "symptom_classifier_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    with open(OUT / "symptom_classifier_test.jsonl", "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

    # Save department mapping for reference
    with open(OUT / "department_mapping.json", "w") as f:
        json.dump(DISEASE_MAP, f, indent=2)

    # Save precautions — Agent 3 will use these for pre-visit instructions
    with open(OUT / "disease_precautions.json", "w") as f:
        json.dump(precautions, f, indent=2)

    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Saved to {OUT}/\n")

    # Stats
    dept_counts = Counter(r["department"] for r in records)
    urgency_counts = Counter(r["urgency"] for r in records)

    print("Department distribution:")
    for dept, cnt in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"  {dept:<25} {cnt}")

    print("\nUrgency distribution:")
    for u, cnt in sorted(urgency_counts.items(), key=lambda x: -x[1]):
        print(f"  {u:<12} {cnt}")


if __name__ == "__main__":
    main()
