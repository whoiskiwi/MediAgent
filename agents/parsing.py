"""Output parsing utilities for all agents.

Separated from agent modules so tests can run without torch/transformers.
"""

import re

from schemas import ClassifierOutput, ResponseOutput

VALID_DEPARTMENTS = {
    "Cardiology",
    "Neurology",
    "Dermatology",
    "Gastroenterology",
    "Endocrinology",
    "Pulmonology",
    "Infectious Disease",
    "Orthopedics",
    "Urology",
    "General Medicine",
}

VALID_URGENCIES = {"Routine", "Urgent", "Emergency"}


def parse_classifier_output(text: str) -> ClassifierOutput:
    dept_match = re.search(r"Department:\s*([^\n]+)", text)
    urg_match = re.search(r"Urgency:\s*([^\n]+)", text)

    department = dept_match.group(1).strip() if dept_match else "General Medicine"
    urgency = urg_match.group(1).strip() if urg_match else "Routine"

    if department not in VALID_DEPARTMENTS:
        department = "General Medicine"
    if urgency not in VALID_URGENCIES:
        urgency = "Routine"

    return ClassifierOutput(department=department, urgency=urgency)


def parse_response_output(text: str) -> ResponseOutput:
    conf_match = re.search(r"Confirmation:\s*(.+?)(?:\n|$)", text)
    inst_match = re.search(r"Instructions:\s*(.+)", text, re.DOTALL)

    confirmation = conf_match.group(1).strip() if conf_match else text
    instructions = inst_match.group(1).strip() if inst_match else ""

    return ResponseOutput(confirmation=confirmation, instructions=instructions)
