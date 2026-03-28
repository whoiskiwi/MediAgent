"""Unit tests for Response Generator output parsing & validation.

Tests run without torch — parsing logic is in agents.parsing.
"""

import pytest

from schemas import ResponseOutput
from agents.parsing import parse_response_output


class TestParseResponseOutput:
    def test_standard_format(self):
        text = (
            "Confirmation: Your appointment with Dr. Smith is on Monday at 09:00.\n"
            "Instructions: Bring your ID. Fast for 8 hours before the visit."
        )
        result = parse_response_output(text)
        assert "Dr. Smith" in result.confirmation
        assert "Bring your ID" in result.instructions

    def test_multiline_instructions(self):
        text = (
            "Confirmation: Appointment confirmed.\n"
            "Instructions: 1. Bring your ID.\n"
            "2. Fast for 8 hours.\n"
            "3. Wear comfortable clothing."
        )
        result = parse_response_output(text)
        assert "Appointment confirmed" in result.confirmation
        assert "comfortable clothing" in result.instructions

    def test_no_structured_format(self):
        text = "Hello, your appointment is confirmed with Dr. Patel on Tuesday."
        result = parse_response_output(text)
        assert "Dr. Patel" in result.confirmation
        assert result.instructions == ""

    def test_empty_string(self):
        result = parse_response_output("")
        assert result.confirmation == ""
        assert result.instructions == ""

    def test_returns_response_output(self):
        text = "Confirmation: Confirmed.\nInstructions: Bring ID."
        result = parse_response_output(text)
        assert isinstance(result, ResponseOutput)
