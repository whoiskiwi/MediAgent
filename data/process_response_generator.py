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
MIN_RESPONSE_LEN = 80   # chars — filter out one-liners
MAX_RESPONSE_LEN = 800  # chars — avoid extremely long outputs

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


def build_synthetic_context(department: str, doctor: str, time_slot: str, urgency: str, symptoms: str) -> str:
    return (
        f"Patient symptoms: {symptoms}\n"
        f"Assigned department: {department}\n"
        f"Doctor: {doctor}\n"
        f"Appointment: {time_slot}\n"
        f"Urgency: {urgency}"
    )


def dialogue_to_record(patient_q: str, doctor_a: str, schedules: list[dict]) -> dict | None:
    """
    Convert a raw doctor-patient dialogue pair into an instruction fine-tuning record.
    We use the doctor's answer as the target output, and synthesise appointment context
    using real doctor/department/time data from doctor_schedules.json.
    """
    doctor_a = str(doctor_a).strip()
    if len(doctor_a) < MIN_RESPONSE_LEN or len(doctor_a) > MAX_RESPONSE_LEN:
        return None

    # Pick a real schedule entry for realistic context
    entry = random.choice(schedules)
    department = entry["department"]
    doctor = entry["doctor"]
    time_slot = f"{entry['day']} at {entry['time_slot']}"
    urgency = random.choice(["Routine", "Routine", "Urgent"])  # weight towards Routine

    # Trim patient question to use as symptom description
    symptoms = str(patient_q).strip()[:200]

    return {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": build_synthetic_context(department, doctor, time_slot, urgency, symptoms),
        "output": doctor_a,
        "department": department,
        "urgency": urgency,
    }


def main():
    random.seed(42)

    out = Path(__file__).resolve().parent / "processed"
    out.mkdir(parents=True, exist_ok=True)

    # Load real doctor schedules — no hardcoding
    schedules = _load_doctor_schedules()
    n_doctors = len(set(r["doctor"] for r in schedules))
    n_depts = len(set(r["department"] for r in schedules))
    print(f"Loaded {n_doctors} doctors across {n_depts} departments from doctor_schedules.json")

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"datasets library not installed: {e}")
        print("Generating a small synthetic sample for local testing instead...")
        records = _generate_synthetic_fallback(schedules, n=200)
    else:
        print("Loading AI Medical Chatbot from HuggingFace (this may take a few minutes)...")
        ds = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
        print(f"Loaded {len(ds)} dialogues")

        indices = random.sample(range(len(ds)), int(len(ds) * SAMPLE_RATIO))
        sampled = ds.select(indices)
        print(f"Sampled {len(sampled)} rows ({SAMPLE_RATIO:.0%})")

        records = []
        for row in sampled:
            rec = dialogue_to_record(
                patient_q=row.get("Patient", row.get("input", "")),
                doctor_a=row.get("Doctor", row.get("output", "")),
                schedules=schedules,
            )
            if rec:
                records.append(rec)

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


def _generate_synthetic_fallback(schedules: list[dict], n: int = 200) -> list[dict]:
    """Minimal synthetic records for testing the pipeline without HuggingFace access."""
    templates = [
        "I have been having chest pain and shortness of breath for 3 days.",
        "My skin has been very itchy with a rash for a week.",
        "I have had a severe headache and dizziness for two days.",
    ]
    output_template = (
        "Confirmation: Your appointment with {doctor} in {department} has been confirmed "
        "for {time_slot}.\n"
        "Instructions: Please arrive 15 minutes early. Bring a list of current medications. "
        "Note when your symptoms started and any changes."
    )
    records = []
    for i in range(n):
        symptoms = templates[i % len(templates)]
        entry = random.choice(schedules)
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
