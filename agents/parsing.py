"""Output parsing utilities for all agents.

Separated from agent modules so tests can run without torch/transformers.
"""

import re

from schemas import ClassifierOutput, ResponseOutput

# Sentence-level patterns that indicate a medical Q&A sign-off / doctor signature.
# When any of these are found, everything from that point on is discarded.
_SIGNOFF_PATTERNS = re.compile(
    r"("
    r"thanks\s+for\s+(posting|your\s+question)"
    r"|hope\s+i\s+have\s+solved"
    r"|i\s+will\s+be\s+happy\s+to\s+help"
    r"|wishing\s+(you\s+)?good\s+health"
    r"|wish\s+you\s+(good\s+health|well)"
    r"|feel\s+free\s+to\s+contact"
    r"|do\s+not\s+hesitate\s+to\s+contact"
    r"|consult\s+(a\s+)?(?:doctor|physician|specialist|cardiologist)"
    r"|regards[,.]?\s*$"
    r")",
    re.IGNORECASE,
)

# A standalone line that is just a doctor name/credential (e.g. "Dr. Manish Purohit. cardiologist.")
_DOCTOR_LINE_PATTERN = re.compile(
    r"^Dr\.\s+\w[\w\s]+\.\s*(cardiologist|physician|specialist|surgeon|neurologist"
    r"|dermatologist|gastroenterologist|endocrinologist|pulmonologist"
    r"|orthopedist|urologist|internist)?\.?\s*$",
    re.IGNORECASE,
)


_DEFAULT_INSTRUCTIONS = (
    "Please arrive 15 minutes early and bring a valid photo ID. "
    "Bring any relevant medical records or previous test results. "
    "List all current medications before your visit."
)


def _clean_text(text: str) -> str:
    """Remove medical Q&A sign-offs and repeated doctor-signature lines.

    Scans sentence by sentence and stops at the first sign-off / doctor-signature
    sentence.  If the very first sentence is already a sign-off (the model
    reproduced raw training-data), falls back to _DEFAULT_INSTRUCTIONS.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for sentence in sentences:
        if _SIGNOFF_PATTERNS.search(sentence):
            break
        if _DOCTOR_LINE_PATTERN.match(sentence.strip()):
            break
        kept.append(sentence)

    cleaned = " ".join(kept).strip()
    return cleaned if cleaned else _DEFAULT_INSTRUCTIONS

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

    if confirmation:
        confirmation = _clean_text(confirmation)
    if instructions:
        instructions = _clean_text(instructions)

    return ResponseOutput(confirmation=confirmation, instructions=instructions)
