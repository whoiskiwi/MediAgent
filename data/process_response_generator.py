"""
Agent 3 Data Processing — AI Medical Chatbot Dataset
Input:  HuggingFace: ruslanmv/ai-medical-chatbot  (256,916 dialogues)
        OR manually downloaded parquet/csv placed in data/raw/ai_medical_chatbot/
Output: data/processed/response_generator_train.jsonl
        data/processed/response_generator_test.jsonl

Task:
  Fine-tune LLaMA to generate patient-friendly appointment confirmations
  and pre-visit instructions, grounded in real doctor-patient dialogue patterns.

Strategy:
  - Load 20% sample (≈51k rows) — sufficient for fine-tuning, saves Colab time
  - Convert each dialogue into an instruction record:
      instruction: "You are a medical assistant..."
      input:       appointment context (department, doctor, time, symptoms)
      output:      confirmation + pre-visit instructions (from real dialogue)
  - Filter out very short or low-quality responses
"""

import json
import random
from pathlib import Path


SAMPLE_RATIO = 0.20
MIN_RESPONSE_LEN = 80    # chars — filter out one-liners
MAX_RESPONSE_LEN = 800   # chars — avoid extremely long outputs
MAX_SYMPTOM_LEN = 200    # chars — trim patient question for symptom description

INSTRUCTION_TEMPLATE = (
    "You are a compassionate medical assistant. "
    "A patient has been assigned an appointment. "
    "Write a warm, clear appointment confirmation and practical pre-visit instructions. "
    "Keep the tone professional but reassuring. "
    "Format your response as:\n"
    "Confirmation: <one sentence confirming the appointment>\n"
    "Instructions: <2-4 specific pre-visit instructions>"
)


def _load_doctor_schedules() -> list[dict]:
    """Load generated doctor schedules from JSON (produced by process_appointment_retriever.py)."""
    path = Path(__file__).resolve().parent / "processed" / "doctor_schedules.json"
    if not path.exists():
        raise FileNotFoundError(
            f"doctor_schedules.json not found at {path}. "
            "Run process_appointment_retriever.py first."
        )
    with open(path) as f:
        return json.load(f)


def _load_department_mapping() -> dict[str, str]:
    """Load disease → department mapping (case-insensitive lookup)."""
    path = Path(__file__).resolve().parent / "processed" / "department_mapping.json"
    with open(path) as f:
        raw = json.load(f)
    # Build case-insensitive lookup: lowercase disease name → department
    return {disease.strip().lower(): dept for disease, dept in raw.items()}


def _schedules_by_department(schedules: list[dict]) -> dict[str, list[dict]]:
    """Group schedules by department for efficient lookup."""
    by_dept: dict[str, list[dict]] = {}
    for entry in schedules:
        by_dept.setdefault(entry["department"], []).append(entry)
    return by_dept


def build_synthetic_context(department: str, doctor: str, time_slot: str, urgency: str, symptoms: str) -> str:
    return (
        f"Patient symptoms: {symptoms}\n"
        f"Assigned department: {department}\n"
        f"Doctor: {doctor}\n"
        f"Appointment: {time_slot}\n"
        f"Urgency: {urgency}"
    )


def dialogue_to_record(
    patient_q: str,
    doctor_a: str,
    description: str,
    dept_mapping: dict[str, str],
    schedules_by_dept: dict[str, list[dict]],
    all_schedules: list[dict],
) -> dict | None:
    """
    Convert a raw doctor-patient dialogue pair into an instruction fine-tuning record.

    Uses the Description field (disease name) from the dataset to look up the correct
    department via department_mapping.json, then picks a doctor from that department.
    """
    doctor_a = str(doctor_a).strip()
    if len(doctor_a) < MIN_RESPONSE_LEN or len(doctor_a) > MAX_RESPONSE_LEN:
        return None

    # Look up department from the Description field (disease name)
    desc_key = str(description).strip().lower()
    matched_dept = dept_mapping.get(desc_key, "General Medicine")

    # Pick a doctor from the MATCHED department
    dept_schedules = schedules_by_dept.get(matched_dept)
    if not dept_schedules:
        dept_schedules = schedules_by_dept.get("General Medicine", all_schedules)

    entry = random.choice(dept_schedules)
    department = entry["department"]
    doctor = entry["doctor"]
    time_slot = f"{entry['day']} at {entry['time_slot']}"
    urgency = random.choice(["Routine", "Routine", "Urgent", "Urgent", "Emergency"])  # ~40% Routine, ~40% Urgent, ~20% Emergency

    # Trim patient question to use as symptom description
    symptoms = str(patient_q).strip()[:MAX_SYMPTOM_LEN]

    # Wrap in structured format: Confirmation + Instructions
    # so the model learns the expected output schema
    formatted_output = (
        f"Confirmation: Your appointment with {doctor} in {department} "
        f"has been confirmed for {time_slot}. "
        f"We're here to help with your concerns.\n"
        f"Instructions: {doctor_a}"
    )

    return {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": build_synthetic_context(department, doctor, time_slot, urgency, symptoms),
        "output": formatted_output,
        "department": department,
        "urgency": urgency,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N samples (for quick testing)")
    args = parser.parse_args()

    random.seed(42)

    out = Path(__file__).resolve().parent / "processed"
    out.mkdir(parents=True, exist_ok=True)

    # Load real doctor schedules — no hardcoding
    schedules = _load_doctor_schedules()
    n_doctors = len(set(r["doctor"] for r in schedules))
    n_depts = len(set(r["department"] for r in schedules))
    print(f"Loaded {n_doctors} doctors across {n_depts} departments from doctor_schedules.json")

    # Load department mapping for Description → department lookup
    dept_mapping = _load_department_mapping()
    schedules_by_dept = _schedules_by_department(schedules)
    print(f"Loaded {len(dept_mapping)} disease→department mappings")

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"datasets library not installed: {e}")
        print("Generating a small synthetic sample for local testing instead...")
        records = _generate_synthetic_fallback(schedules_by_dept, n=200)
    else:
        print("Loading AI Medical Chatbot from HuggingFace (this may take a few minutes)...")
        ds = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
        print(f"Loaded {len(ds)} dialogues")

        n_sample = args.limit if args.limit else int(len(ds) * SAMPLE_RATIO)
        n_sample = min(n_sample, len(ds))
        indices = random.sample(range(len(ds)), n_sample)
        sampled = ds.select(indices)
        print(f"Sampled {len(sampled)} rows{' (--limit)' if args.limit else f' ({SAMPLE_RATIO:.0%})'}")

        records = []
        unmatched = 0
        for row in sampled:
            desc = row.get("Description", "")
            rec = dialogue_to_record(
                patient_q=row.get("Patient", row.get("input", "")),
                doctor_a=row.get("Doctor", row.get("output", "")),
                description=desc,
                dept_mapping=dept_mapping,
                schedules_by_dept=schedules_by_dept,
                all_schedules=schedules,
            )
            if rec:
                records.append(rec)
                if rec["department"] == "General Medicine" and desc.strip().lower() not in dept_mapping:
                    unmatched += 1

        print(f"Unmatched descriptions (→ General Medicine): {unmatched}/{len(records)}")

        print(f"Kept {len(records)} records after quality filtering")

    # Train / test split
    random.seed(42)  # re-seed for reproducible split
    random.shuffle(records)
    split = int(len(records) * 0.85)
    train, test = records[:split], records[split:]

    with open(out / "response_generator_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    with open(out / "response_generator_test.jsonl", "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Saved to {out}/")

    # Print department distribution for sanity check
    from collections import Counter
    dept_dist = Counter(r["department"] for r in records)
    print("\n--- Department Distribution ---")
    for dept, cnt in sorted(dept_dist.items(), key=lambda x: -x[1]):
        print(f"  {dept:<25} {cnt:>6}  ({cnt/len(records):.1%})")

    # Print a few sample records
    print("\n--- Sample Records (first 3) ---")
    for r in records[:3]:
        inp_lines = r["input"].split("\n")
        symptoms = inp_lines[0] if inp_lines else ""
        dept_line = inp_lines[1] if len(inp_lines) > 1 else ""
        print(f"  {symptoms[:80]}")
        print(f"  {dept_line}")
        print()


def _generate_synthetic_fallback(schedules_by_dept: dict[str, list[dict]], n: int = 200) -> list[dict]:
    """Minimal synthetic records for testing the pipeline without HuggingFace access."""
    # Symptoms matched to correct departments
    templates = [
        ("I have been having chest pain and shortness of breath for 3 days.", "Cardiology"),
        ("My skin has been very itchy with a rash for a week.", "Dermatology"),
        ("I have had a severe headache and dizziness for two days.", "Neurology"),
        ("I have been experiencing frequent urination and burning sensation.", "Urology"),
        ("I have stomach pain and acid reflux after eating.", "Gastroenterology"),
        ("I have high blood sugar and feel thirsty all the time.", "Endocrinology"),
    ]
    output_template = (
        "Confirmation: Your appointment with {doctor} in {department} has been confirmed "
        "for {time_slot}.\n"
        "Instructions: Please arrive 15 minutes early. Bring a list of current medications. "
        "Note when your symptoms started and any changes."
    )
    all_schedules = [e for entries in schedules_by_dept.values() for e in entries]
    records = []
    for i in range(n):
        symptoms, dept = templates[i % len(templates)]
        dept_entries = schedules_by_dept.get(dept, all_schedules)
        entry = random.choice(dept_entries)
        time_slot = f"{entry['day']} at {entry['time_slot']}"
        output = output_template.format(
            doctor=entry["doctor"],
            department=entry["department"],
            time_slot=time_slot,
        )
        records.append({
            "instruction": INSTRUCTION_TEMPLATE,
            "input": build_synthetic_context(
                entry["department"], entry["doctor"],
                time_slot, "Routine", symptoms,
            ),
            "output": output,
            "department": entry["department"],
            "urgency": "Routine",
        })
    return records


if __name__ == "__main__":
    main()
