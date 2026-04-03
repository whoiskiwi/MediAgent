"""Quick sanity check: verify department matching logic on a small sample.

Usage:
    python -m tests.test_data_processing
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_dept_mapping() -> dict[str, str]:
    with open(DATA_DIR / "department_mapping.json") as f:
        raw = json.load(f)
    return {k.strip().lower(): v for k, v in raw.items()}


def load_schedules_by_dept() -> dict[str, list[dict]]:
    with open(DATA_DIR / "doctor_schedules.json") as f:
        schedules = json.load(f)
    by_dept: dict[str, list[dict]] = {}
    for entry in schedules:
        by_dept.setdefault(entry["department"], []).append(entry)
    return by_dept


def main():
    dept_mapping = load_dept_mapping()
    schedules_by_dept = load_schedules_by_dept()

    print(f"department_mapping: {len(dept_mapping)} diseases")
    print(f"schedules departments: {sorted(schedules_by_dept.keys())}\n")

    # Simulate Description fields from the HuggingFace dataset
    test_cases = [
        # (Description, expected department)
        ("Fungal infection", "Dermatology"),
        ("Heart attack", "Cardiology"),
        ("Migraine", "Neurology"),
        ("GERD", "Gastroenterology"),
        ("Diabetes", "Endocrinology"),
        ("Pneumonia", "Pulmonology"),
        ("Malaria", "Infectious Disease"),
        ("Arthritis", "Orthopedics"),
        ("Urinary tract infection", "Urology"),
        ("Allergy", "General Medicine"),
        # Edge cases
        ("fungal infection", "Dermatology"),       # lowercase
        ("  Migraine  ", "Neurology"),             # whitespace
        ("HEART ATTACK", "Cardiology"),            # uppercase
        ("Something Unknown", "General Medicine"), # unmapped → fallback
        ("", "General Medicine"),                   # empty
    ]

    passed = 0
    failed = 0

    for desc, expected_dept in test_cases:
        desc_key = desc.strip().lower()
        matched = dept_mapping.get(desc_key, "General Medicine")

        # Check doctor is from matched department
        dept_scheds = schedules_by_dept.get(matched)
        has_doctors = dept_scheds is not None and len(dept_scheds) > 0

        ok = matched == expected_dept and has_doctors
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] Description={desc!r:35s} → dept={matched:20s} (expected={expected_dept:20s}, has_doctors={has_doctors})")

    print(f"\n{passed}/{passed + failed} passed")

    if failed:
        print("\nFAILED cases above need attention.")
    else:
        print("\nAll good! Safe to run: python data/process_response_generator.py")


if __name__ == "__main__":
    main()
