"""
Test agent parsing and utility logic — no model loading, no AWS.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas import normalize_urgency


# ---------------------------------------------------------------------------
# schemas.py utilities
# ---------------------------------------------------------------------------

def test_normalize_urgency_emergency():
    assert normalize_urgency("Emergency") == "Urgent"

def test_normalize_urgency_urgent_unchanged():
    assert normalize_urgency("Urgent") == "Urgent"

def test_normalize_urgency_routine_unchanged():
    assert normalize_urgency("Routine") == "Routine"


# ---------------------------------------------------------------------------
# Agent 1: symptom classifier output parsing
# ---------------------------------------------------------------------------

from agents.symptom_classifier.agent import _parse_output as parse_classifier


@pytest.mark.parametrize("text,expected_dept,expected_urg", [
    ("Department: Cardiology | Urgency: Emergency", "Cardiology", "Emergency"),
    ("Department: Neurology | Urgency: Urgent",     "Neurology",  "Urgent"),
    ("Department: General Medicine | Urgency: Routine", "General Medicine", "Routine"),
])
def test_classifier_parse_valid(text, expected_dept, expected_urg):
    result = parse_classifier(text)
    assert result.department == expected_dept
    assert result.urgency == expected_urg


def test_classifier_parse_fallback_on_garbage():
    """Unrecognisable output should default gracefully."""
    result = parse_classifier("I don't understand the question")
    assert result.department == "General Medicine"
    assert result.urgency == "Routine"


def test_classifier_parse_case_insensitive():
    result = parse_classifier("department: cardiology | urgency: URGENT")
    assert result.department == "Cardiology"
    assert result.urgency == "Urgent"


# ---------------------------------------------------------------------------
# Agent 2: DynamoDB item parsing
# ---------------------------------------------------------------------------

from agents.appointment_retriever.agent import _parse_item


def test_retriever_parse_sk_format():
    item = {"department": "Cardiology", "SK": "Dr. Smith#Monday#09:00"}
    doctor, time_slot = _parse_item(item)
    assert doctor == "Dr. Smith"
    assert time_slot == "Monday 09:00"


def test_retriever_parse_flat_attributes():
    item = {"doctor": "Dr. Lee", "time_slot": "Tuesday 14:00"}
    doctor, time_slot = _parse_item(item)
    assert doctor == "Dr. Lee"
    assert time_slot == "Tuesday 14:00"


def test_retriever_parse_missing_sk_fallback():
    item = {"department": "Cardiology"}
    doctor, time_slot = _parse_item(item)
    assert doctor == "Unknown Doctor"
    assert time_slot == "Unknown Slot"


# ---------------------------------------------------------------------------
# Agent 3: response generator output parsing
# ---------------------------------------------------------------------------

from agents.response_generator.agent import _parse_output as parse_generator


def test_generator_parse_with_separator():
    text = "Confirmation: Your appointment is confirmed.\n---\n• Arrive 15 min early.\n• Bring ID."
    confirmation, instructions = parse_generator(text)
    assert "confirmed" in confirmation
    assert "Arrive" in instructions


def test_generator_parse_no_separator_fallback():
    text = "Confirmation: Your appointment with Dr. Smith has been booked."
    confirmation, instructions = parse_generator(text)
    assert "booked" in confirmation
    assert instructions  # fallback string is non-empty
