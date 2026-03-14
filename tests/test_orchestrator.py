"""
Test LangGraph orchestrator flow.
Each agent node is mocked — verifies state is passed and merged correctly.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_classifier(department="Cardiology", urgency="Emergency"):
    def mock_classifier(state):
        return {**state, "department": department, "urgency": urgency}
    return mock_classifier


def _make_retriever(doctor="Dr. Smith", time_slot="Monday 09:00"):
    def mock_retriever(state):
        return {**state, "doctor": doctor, "time_slot": time_slot}
    return mock_retriever


def _make_generator(confirmation="Confirmed.", instructions="Arrive early."):
    def mock_generator(state):
        return {**state, "confirmation": confirmation, "instructions": instructions}
    return mock_generator


def test_state_flows_through_all_nodes():
    with (
        patch("orchestrator.graph.run_classifier", side_effect=_make_classifier()),
        patch("orchestrator.graph.run_retriever", side_effect=_make_retriever()),
        patch("orchestrator.graph.run_generator", side_effect=_make_generator()),
    ):
        from orchestrator.graph import run_pipeline
        result = run_pipeline("chest pain")

    assert result["patient_text"] == "chest pain"
    assert result["department"] == "Cardiology"
    assert result["urgency"] == "Emergency"
    assert result["doctor"] == "Dr. Smith"
    assert result["time_slot"] == "Monday 09:00"
    assert result["confirmation"] == "Confirmed."
    assert result["instructions"] == "Arrive early."


def test_patient_text_preserved_throughout():
    """patient_text must survive all three node transformations."""
    with (
        patch("orchestrator.graph.run_classifier", side_effect=_make_classifier()),
        patch("orchestrator.graph.run_retriever", side_effect=_make_retriever()),
        patch("orchestrator.graph.run_generator", side_effect=_make_generator()),
    ):
        from orchestrator.graph import run_pipeline
        result = run_pipeline("my unique symptom text")

    assert result["patient_text"] == "my unique symptom text"


def test_different_departments_routed():
    with (
        patch("orchestrator.graph.run_classifier",
              side_effect=_make_classifier(department="Neurology", urgency="Urgent")),
        patch("orchestrator.graph.run_retriever", side_effect=_make_retriever()),
        patch("orchestrator.graph.run_generator", side_effect=_make_generator()),
    ):
        from orchestrator.graph import run_pipeline
        result = run_pipeline("I have a severe headache")

    assert result["department"] == "Neurology"
    assert result["urgency"] == "Urgent"
