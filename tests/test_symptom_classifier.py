"""Unit tests for Symptom Classifier output parsing & validation.

Tests run without torch — parsing logic is in agents.parsing.
"""

import pytest

from schemas import ClassifierOutput
from agents.parsing import (
    parse_classifier_output,
    VALID_DEPARTMENTS,
    VALID_URGENCIES,
)


class TestParseClassifierOutput:
    def test_standard_format(self):
        text = "Department: Cardiology\nUrgency: Emergency"
        result = parse_classifier_output(text)
        assert result.department == "Cardiology"
        assert result.urgency == "Emergency"

    def test_extra_whitespace(self):
        text = "Department:  Neurology \nUrgency:  Urgent "
        result = parse_classifier_output(text)
        assert result.department == "Neurology"
        assert result.urgency == "Urgent"

    def test_extra_text_around(self):
        text = "Based on symptoms:\nDepartment: Dermatology\nUrgency: Routine\nThank you."
        result = parse_classifier_output(text)
        assert result.department == "Dermatology"
        assert result.urgency == "Routine"

    def test_invalid_department_falls_back(self):
        text = "Department: Psychiatry\nUrgency: Routine"
        result = parse_classifier_output(text)
        assert result.department == "General Medicine"

    def test_invalid_urgency_falls_back(self):
        text = "Department: Cardiology\nUrgency: Critical"
        result = parse_classifier_output(text)
        assert result.urgency == "Routine"

    def test_no_match_falls_back(self):
        text = "I don't know"
        result = parse_classifier_output(text)
        assert result.department == "General Medicine"
        assert result.urgency == "Routine"

    def test_all_valid_departments(self):
        for dept in VALID_DEPARTMENTS:
            text = f"Department: {dept}\nUrgency: Routine"
            result = parse_classifier_output(text)
            assert result.department == dept

    def test_all_valid_urgencies(self):
        for urg in VALID_URGENCIES:
            text = f"Department: Cardiology\nUrgency: {urg}"
            result = parse_classifier_output(text)
            assert result.urgency == urg

    def test_returns_classifier_output(self):
        text = "Department: Cardiology\nUrgency: Routine"
        result = parse_classifier_output(text)
        assert isinstance(result, ClassifierOutput)
