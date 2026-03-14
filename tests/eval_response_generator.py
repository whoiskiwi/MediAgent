"""Evaluate Agent 3 (Response Generator) on 6,362 test samples.

Computes:
- Format compliance: % of responses with both Confirmation: and Instructions: fields
- Confirmation present: % non-empty confirmation
- Instructions present: % non-empty instructions
- Average response length (chars)
- Per-department and per-urgency breakdown

Usage:
    python -m tests.eval_response_generator [--limit N]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from schemas import ResponseInput
from agents.response_generator.agent import ResponseGeneratorAgent

TEST_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "response_generator_test.jsonl"


def load_test_data(limit: int | None = None) -> list[dict]:
    with open(TEST_PATH) as f:
        data = [json.loads(line) for line in f]
    if limit:
        data = data[:limit]
    return data


def build_input(sample: dict) -> ResponseInput:
    """Reconstruct ResponseInput from test sample fields."""
    # Extract fields from the input text
    input_text = sample["input"]
    lines = input_text.strip().split("\n")

    patient_text = ""
    department = sample.get("department", "General Medicine")
    urgency = sample.get("urgency", "Routine")
    doctor = "Dr. Unknown"
    time_slot = "Monday at 08:00"

    for line in lines:
        if line.startswith("Patient symptoms:"):
            patient_text = line.replace("Patient symptoms:", "").strip()
        elif line.startswith("Assigned department:"):
            department = line.replace("Assigned department:", "").strip()
        elif line.startswith("Doctor:"):
            doctor = line.replace("Doctor:", "").strip()
        elif line.startswith("Appointment:"):
            time_slot = line.replace("Appointment:", "").strip()
        elif line.startswith("Urgency:"):
            urgency = line.replace("Urgency:", "").strip()

    return ResponseInput(
        patient_text=patient_text,
        department=department,
        doctor=doctor,
        time_slot=time_slot,
        urgency=urgency,
    )


def evaluate(agent: ResponseGeneratorAgent, data: list[dict]) -> dict:
    total = len(data)
    format_ok = 0
    has_confirmation = 0
    has_instructions = 0
    total_length = 0

    dept_counts = defaultdict(lambda: {"total": 0, "format_ok": 0})
    urg_counts = defaultdict(lambda: {"total": 0, "format_ok": 0})

    for i, sample in enumerate(data):
        resp_input = build_input(sample)
        result = agent.generate(resp_input)

        conf = result.confirmation.strip()
        inst = result.instructions.strip()
        full_text = f"{conf} {inst}"

        both_present = bool(conf) and bool(inst)
        if both_present:
            format_ok += 1
        if conf:
            has_confirmation += 1
        if inst:
            has_instructions += 1
        total_length += len(full_text)

        dept = resp_input.department
        urg = resp_input.urgency
        dept_counts[dept]["total"] += 1
        urg_counts[urg]["total"] += 1
        if both_present:
            dept_counts[dept]["format_ok"] += 1
            urg_counts[urg]["format_ok"] += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] format compliance: {format_ok/(i+1):.1%}")

    dept_report = {}
    for d, c in sorted(dept_counts.items()):
        dept_report[d] = {
            "total": c["total"],
            "format_compliance": c["format_ok"] / c["total"] if c["total"] > 0 else 0.0,
        }

    urg_report = {}
    for u, c in sorted(urg_counts.items()):
        urg_report[u] = {
            "total": c["total"],
            "format_compliance": c["format_ok"] / c["total"] if c["total"] > 0 else 0.0,
        }

    return {
        "total": total,
        "format_compliance": format_ok / total,
        "confirmation_rate": has_confirmation / total,
        "instructions_rate": has_instructions / total,
        "avg_response_length": total_length / total,
        "department_report": dept_report,
        "urgency_report": urg_report,
    }


def print_report(results: dict):
    print("\n" + "=" * 60)
    print("RESPONSE GENERATOR EVALUATION REPORT")
    print("=" * 60)
    print(f"Total samples:         {results['total']}")
    print(f"Format compliance:     {results['format_compliance']:.1%}")
    print(f"Confirmation present:  {results['confirmation_rate']:.1%}")
    print(f"Instructions present:  {results['instructions_rate']:.1%}")
    print(f"Avg response length:   {results['avg_response_length']:.0f} chars")

    print("\n--- Per-Department ---")
    print(f"{'Department':<22} {'Compliance':>10} {'Samples':>8}")
    for dept, m in sorted(results["department_report"].items()):
        print(f"{dept:<22} {m['format_compliance']:>10.1%} {m['total']:>8}")

    print("\n--- Per-Urgency ---")
    print(f"{'Urgency':<22} {'Compliance':>10} {'Samples':>8}")
    for urg, m in sorted(results["urgency_report"].items()):
        print(f"{urg:<22} {m['format_compliance']:>10.1%} {m['total']:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N samples")
    args = parser.parse_args()

    print("Loading test data...")
    data = load_test_data(args.limit)
    print(f"Loaded {len(data)} test samples")

    print("Loading model...")
    agent = ResponseGeneratorAgent()

    print("Running evaluation...")
    results = evaluate(agent, data)
    print_report(results)

    # Save results to JSON
    output_path = Path(__file__).resolve().parents[1] / "tests" / "eval_response_generator_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
