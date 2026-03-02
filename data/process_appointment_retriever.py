"""
Agent 2 Data Processing — Doctor Schedule Generator
Input:  data/raw/doctor_schedules/doctors.csv
Output: data/processed/doctor_schedules.json   (DynamoDB seed data)

Logic:
  - Read doctor schedule rows (name, department, day, morning/afternoon ranges) from CSV
  - Generate 1-hour time slots within each range
  - Output is used to seed the DynamoDB table via upload_to_dynamodb.py
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path


def time_to_minutes(t: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def minutes_to_time(m: int) -> str:
    """Convert minutes since midnight to 'HH:MM'."""
    return f"{m // 60:02d}:{m % 60:02d}"


def generate_slots(start: str, end: str) -> list[str]:
    """Generate 1-hour interval slots from start to end (exclusive)."""
    if not start or not end:
        return []
    if not re.fullmatch(r"\d{2}:\d{2}", start) or not re.fullmatch(r"\d{2}:\d{2}", end):
        print(f"WARNING: invalid time format: start='{start}', end='{end}', skipping")
        return []
    s, e = time_to_minutes(start), time_to_minutes(end)
    if s >= e:
        print(f"WARNING: start '{start}' is not before end '{end}', skipping")
        return []
    return [minutes_to_time(t) for t in range(s, e, 60)]


def load_and_generate(csv_path: Path) -> list[dict]:
    """Read CSV and generate all time slot records."""
    REQUIRED = {"doctor", "department", "day", "morning_start", "morning_end", "afternoon_start", "afternoon_end"}
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if missing := REQUIRED - set(reader.fieldnames or []):
            raise ValueError(f"CSV is missing required columns: {missing}")
        for row in reader:
            doctor = row["doctor"].strip()
            department = row["department"].strip()
            day = row["day"].strip()

            morning_slots = generate_slots(
                row["morning_start"].strip(),
                row["morning_end"].strip(),
            )
            afternoon_slots = generate_slots(
                row["afternoon_start"].strip(),
                row["afternoon_end"].strip(),
            )

            for slot in morning_slots + afternoon_slots:
                records.append({
                    "doctor": doctor,
                    "department": department,
                    "day": day,
                    "time_slot": slot,
                    "available": True,
                })

    return records


def main():
    csv_path = Path(__file__).resolve().parent / "raw" / "doctor_schedules" / "doctors.csv"
    out = Path(__file__).resolve().parent / "processed"
    out.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: doctors.csv not found at {csv_path}")
        return

    records = load_and_generate(csv_path)

    with open(out / "doctor_schedules.json", "w") as f:
        json.dump(records, f, indent=2)

    # Summary
    doctors = set(r["doctor"] for r in records)
    departments = set(r["department"] for r in records)
    print(f"Loaded from {csv_path.name}")
    print(f"Generated {len(records)} time slots")
    print(f"Doctors: {len(doctors)} across {len(departments)} departments")
    print(f"Saved to {out}/doctor_schedules.json")

    dept_slots = Counter(r["department"] for r in records)
    print("\nSlots per department:")
    for dept, cnt in sorted(dept_slots.items(), key=lambda x: -x[1]):
        print(f"  {dept:<25} {cnt} slots")


if __name__ == "__main__":
    main()
