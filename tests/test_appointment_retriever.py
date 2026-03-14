"""Unit tests for Appointment Retriever agent.

Tests use a mocked DynamoDB table to avoid AWS dependency.
"""

import pytest
from unittest.mock import MagicMock, patch

from schemas import AppointmentOutput, AppointmentQuery
from agents.appointment_retriever.agent import AppointmentRetrieverAgent


@pytest.fixture
def mock_agent():
    """Create an agent with a mocked DynamoDB table."""
    with patch("agents.appointment_retriever.agent.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        agent = AppointmentRetrieverAgent()
        agent._mock_table = mock_table
        yield agent


class TestRetrieve:
    def test_returns_earliest_slot(self, mock_agent):
        mock_agent._mock_table.query.return_value = {
            "Items": [
                {"doctor": "Dr. B", "day": "Wednesday", "time_slot": "10:00", "available": True},
                {"doctor": "Dr. A", "day": "Monday", "time_slot": "08:00", "available": True},
                {"doctor": "Dr. C", "day": "Monday", "time_slot": "09:00", "available": True},
            ]
        }
        result = mock_agent.retrieve(AppointmentQuery(department="Cardiology"))
        assert result.doctor == "Dr. A"
        assert result.time_slot == "Monday at 08:00"

    def test_no_available_slots(self, mock_agent):
        mock_agent._mock_table.query.return_value = {"Items": []}
        result = mock_agent.retrieve(AppointmentQuery(department="Cardiology"))
        assert result.doctor == "No available doctor"
        assert result.time_slot == "No available slot"

    def test_single_slot(self, mock_agent):
        mock_agent._mock_table.query.return_value = {
            "Items": [
                {"doctor": "Dr. X", "day": "Friday", "time_slot": "16:00", "available": True},
            ]
        }
        result = mock_agent.retrieve(AppointmentQuery(department="Neurology"))
        assert result.doctor == "Dr. X"
        assert result.time_slot == "Friday at 16:00"

    def test_returns_appointment_output(self, mock_agent):
        mock_agent._mock_table.query.return_value = {
            "Items": [
                {"doctor": "Dr. Y", "day": "Tuesday", "time_slot": "11:00", "available": True},
            ]
        }
        result = mock_agent.retrieve(AppointmentQuery(department="Dermatology"))
        assert isinstance(result, AppointmentOutput)
