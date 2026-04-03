"""
Test FastAPI endpoints.
LLM models and DynamoDB are mocked — no GPU or AWS credentials required.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MOCK_STATE = {
    "patient_text": "I have chest pain",
    "department": "Cardiology",
    "urgency": "Emergency",
    "doctor": "Dr. Smith",
    "time_slot": "Monday 09:00",
    "confirmation": "Your appointment with Dr. Smith has been confirmed.",
    "instructions": "Please arrive 15 minutes early.",
}


@pytest.fixture
def client():
    with patch("orchestrator.graph.run_pipeline", return_value=MOCK_STATE):
        from main import app
        return TestClient(app)


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_query_returns_all_agents(client):
    r = client.post("/api/v1/query", json={"symptom": "I have chest pain"})
    assert r.status_code == 200
    body = r.json()
    assert "agent1" in body
    assert "agent2" in body
    assert "agent3" in body


def test_query_agent1_fields(client):
    r = client.post("/api/v1/query", json={"symptom": "I have chest pain"})
    a1 = r.json()["agent1"]
    assert a1["department"] == "Cardiology"
    assert a1["urgency"] == "Emergency"


def test_query_agent2_fields(client):
    r = client.post("/api/v1/query", json={"symptom": "I have chest pain"})
    a2 = r.json()["agent2"]
    assert a2["doctor"] == "Dr. Smith"
    assert a2["time_slot"] == "Monday 09:00"


def test_query_agent3_fields(client):
    r = client.post("/api/v1/query", json={"symptom": "I have chest pain"})
    a3 = r.json()["agent3"]
    assert "confirmation" in a3
    assert "instructions" in a3


def test_query_missing_body(client):
    r = client.post("/api/v1/query", json={})
    assert r.status_code == 422  # Pydantic validation error
