"""Evaluate Agent 1 (Symptom Classifier) on 738 test samples.

Computes:
- Overall accuracy (department + urgency both correct)
- Per-department precision / recall / F1
- Per-urgency precision / recall / F1
- Emergency detection recall (target: >95%)

Usage:
    python -m tests.eval_symptom_classifier [--limit N]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from schemas import SymptomInput
from agents.symptom_classifier.agent import SymptomClassifierAgent

TEST_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "symptom_classifier_test.jsonl"


def load_test_data(limit: int | None = None) -> list[dict]:
    with open(TEST_PATH) as f:
        data = [json.loads(line) for line in f]
    if limit:
        data = data[:limit]
    return data


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(agent: SymptomClassifierAgent, data: list[dict]) -> dict:
    dept_tp = defaultdict(int)
    dept_fp = defaultdict(int)
    dept_fn = defaultdict(int)
    urg_tp = defaultdict(int)
    urg_fp = defaultdict(int)
    urg_fn = defaultdict(int)

    correct_both = 0
    correct_dept = 0
    correct_urg = 0
    total = len(data)

    for i, sample in enumerate(data):
        pred = agent.classify(SymptomInput(patient_text=sample["input"]))
        true_dept = sample["department"]
        true_urg = sample["urgency"]
        pred_dept = pred.department
        pred_urg = pred.urgency

        # Department metrics
        if pred_dept == true_dept:
            dept_tp[true_dept] += 1
            correct_dept += 1
        else:
            dept_fp[pred_dept] += 1
            dept_fn[true_dept] += 1

        # Urgency metrics
        if pred_urg == true_urg:
            urg_tp[true_urg] += 1
            correct_urg += 1
        else:
            urg_fp[pred_urg] += 1
            urg_fn[true_urg] += 1

        if pred_dept == true_dept and pred_urg == true_urg:
            correct_both += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] running accuracy: {correct_both/(i+1):.1%}")

    # Build report
    all_depts = sorted(set(list(dept_tp) + list(dept_fn)))
    all_urgs = sorted(set(list(urg_tp) + list(urg_fn)))

    dept_report = {}
    for d in all_depts:
        p, r, f = compute_prf(dept_tp[d], dept_fp[d], dept_fn[d])
        dept_report[d] = {"precision": p, "recall": r, "f1": f, "support": dept_tp[d] + dept_fn[d]}

    urg_report = {}
    for u in all_urgs:
        p, r, f = compute_prf(urg_tp[u], urg_fp[u], urg_fn[u])
        urg_report[u] = {"precision": p, "recall": r, "f1": f, "support": urg_tp[u] + urg_fn[u]}

    emergency_recall = urg_report.get("Emergency", {}).get("recall", 0.0)

    return {
        "total": total,
        "overall_accuracy": correct_both / total,
        "department_accuracy": correct_dept / total,
        "urgency_accuracy": correct_urg / total,
        "emergency_recall": emergency_recall,
        "department_report": dept_report,
        "urgency_report": urg_report,
    }


def print_report(results: dict):
    print("\n" + "=" * 60)
    print("SYMPTOM CLASSIFIER EVALUATION REPORT")
    print("=" * 60)
    print(f"Total samples:       {results['total']}")
    print(f"Overall accuracy:    {results['overall_accuracy']:.1%}")
    print(f"Department accuracy: {results['department_accuracy']:.1%}")
    print(f"Urgency accuracy:    {results['urgency_accuracy']:.1%}")
    print(f"Emergency recall:    {results['emergency_recall']:.1%}  (target: >95%)")

    print("\n--- Per-Department ---")
    print(f"{'Department':<22} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>8}")
    for dept, m in sorted(results["department_report"].items()):
        print(f"{dept:<22} {m['precision']:>6.1%} {m['recall']:>6.1%} {m['f1']:>6.1%} {m['support']:>8}")

    print("\n--- Per-Urgency ---")
    print(f"{'Urgency':<22} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>8}")
    for urg, m in sorted(results["urgency_report"].items()):
        print(f"{urg:<22} {m['precision']:>6.1%} {m['recall']:>6.1%} {m['f1']:>6.1%} {m['support']:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N samples")
    args = parser.parse_args()

    print("Loading test data...")
    data = load_test_data(args.limit)
    print(f"Loaded {len(data)} test samples")

    print("Loading model...")
    agent = SymptomClassifierAgent()

    print("Running evaluation...")
    results = evaluate(agent, data)
    print_report(results)

    # Save results to JSON
    output_path = Path(__file__).resolve().parents[1] / "tests" / "eval_symptom_classifier_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
