"""Unit tests for Pydantic schemas and utility functions."""

import pytest
from pydantic import ValidationError

from schemas import (
    AgentState,
    AppointmentOutput,
    AppointmentQuery,
    ClassifierOutput,
    ResponseInput,
    ResponseOutput,
    SymptomInput,
    normalize_urgency,
)


# ---------------------------------------------------------------------------
# SymptomInput
# ---------------------------------------------------------------------------

class TestSymptomInput:
    def test_valid(self):
        s = SymptomInput(patient_text="headache and fever")
        assert s.patient_text == "headache and fever"

    def test_empty_string(self):
        s = SymptomInput(patient_text="")
        assert s.patient_text == ""

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            SymptomInput()


# ---------------------------------------------------------------------------
# ClassifierOutput
# ---------------------------------------------------------------------------

class TestClassifierOutput:
    @pytest.mark.parametrize("urgency", ["Routine", "Urgent", "Emergency"])
    def test_valid_urgencies(self, urgency):
        c = ClassifierOutput(department="Cardiology", urgency=urgency)
        assert c.urgency == urgency

    def test_invalid_urgency(self):
        with pytest.raises(ValidationError):
            ClassifierOutput(department="Cardiology", urgency="Critical")

    def test_department_is_freeform(self):
        c = ClassifierOutput(department="Unknown Dept", urgency="Routine")
        assert c.department == "Unknown Dept"


# ---------------------------------------------------------------------------
# AppointmentQuery / AppointmentOutput
# ---------------------------------------------------------------------------

class TestAppointmentModels:
    def test_query(self):
        q = AppointmentQuery(department="Neurology")
        assert q.department == "Neurology"

    def test_output(self):
        o = AppointmentOutput(doctor="Dr. Smith", time_slot="Monday at 09:00")
        assert o.doctor == "Dr. Smith"


# ---------------------------------------------------------------------------
# ResponseInput / ResponseOutput
# ---------------------------------------------------------------------------

class TestResponseModels:
    def test_valid_input(self):
        r = ResponseInput(
            patient_text="chest pain",
            department="Cardiology",
            doctor="Dr. Chen Wei",
            time_slot="Monday at 08:00",
            urgency="Routine",
        )
        assert r.urgency == "Routine"

    def test_input_accepts_emergency(self):
        """ResponseInput now accepts Emergency (Agent 3 retrained with Emergency data)."""
        r = ResponseInput(
            patient_text="chest pain",
            department="Cardiology",
            doctor="Dr. Chen Wei",
            time_slot="Monday at 08:00",
            urgency="Emergency",
        )
        assert r.urgency == "Emergency"

    def test_input_rejects_invalid_urgency(self):
        with pytest.raises(ValidationError):
            ResponseInput(
                patient_text="chest pain",
                department="Cardiology",
                doctor="Dr. Chen Wei",
                time_slot="Monday at 08:00",
                urgency="Critical",
            )

    def test_output(self):
        r = ResponseOutput(
            confirmation="Your appointment is confirmed.",
            instructions="Bring your ID.",
        )
        assert "confirmed" in r.confirmation


# ---------------------------------------------------------------------------
# normalize_urgency
# ---------------------------------------------------------------------------

class TestNormalizeUrgency:
    def test_emergency_mapped_to_urgent(self):
        assert normalize_urgency("Emergency") == "Urgent"

    def test_routine_unchanged(self):
        assert normalize_urgency("Routine") == "Routine"

    def test_urgent_unchanged(self):
        assert normalize_urgency("Urgent") == "Urgent"


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_partial_state(self):
        state: AgentState = {"patient_text": "headache"}
        assert state["patient_text"] == "headache"

    def test_full_state(self):
        state: AgentState = {
            "patient_text": "headache",
            "department": "Neurology",
            "urgency": "Routine",
            "doctor": "Dr. Smith",
            "time_slot": "Monday at 09:00",
            "confirmation": "Confirmed.",
            "instructions": "Bring ID.",
        }
        assert len(state) == 7
